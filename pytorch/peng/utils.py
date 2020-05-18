#!/usr/bin/env python3

import argparse
import numpy as np
import torch
import torch.nn as nn

import time, math

def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    # return torch.sum(torch.stack([torch.sum(x*y) for (x, y) in zip(xs, ys)]))
    return sum([torch.sum(x*y) for (x, y) in zip(xs, ys)])
def group_add(params, update, alpha=1):
    """
    params = params + update*alpha
    :param params: list of variable
    :param update: list of data
    :return:
    """
    for i,p in enumerate(params):
        params[i].data.add_(update[i]*alpha) 
    return params

def hessian_clip_cr(grads_fun, grads_data, params, delta, vs, weight=0.0):
    """ compute the optimal solution of CR subproblem along one direction vs
            min_{a > 0} a* g^Tv + a^2/2 * v^THv  + sigma/3 a^3 ||v||^3
            a = (-v^THv + sqrt((v^THv)^2 - 4\sigma ||v||^3 g^Tv))/(2*\sigma*||v||^3)
    :param grads_fun: Hessian
    :param grads_data: gradient
    :param params: model parameters
    :param delta: cubic regularization sigma = 1/delta
    :param vs: CR direction
    :param weight: H + weight* I
    :return: sub_opt CR solution and model value
    
    """
    sigma = 1.0/delta
    Hvs = torch.autograd.grad(grads_fun, params, grad_outputs=vs, only_inputs=True, retain_graph=True)
    Hvs = [hd.detach() + weight*d for hd, d in zip(Hvs, vs)]
    vHv = group_product(Hvs, vs)
    vnorm = math.sqrt(group_product(vs,vs))
    gv = group_product(grads_data, vs)
    alpha = ( -vHv + math.sqrt((vHv)**2 + 4*sigma*(vnorm**3)*abs(gv)))/2.0/(sigma*(vnorm**3))
    #print('alpha:', alpha.item())
    if gv > 0:
        vs = [-v*alpha for v in vs]
    else:
        vs = [v *alpha for v in vs]
    m = vHv*0.5*alpha*alpha - abs(gv)*alpha + sigma/3.0 * (alpha**3)*(vnorm**3)
    return vs, m.item()
    

def hessian_clip(grads_fun, grads_data, params, delta, vs, weight=0.0):
    """ compute the optimal solution of TR subproblem along one direction vs

    :param grads_fun: Hessian
    :param grads_data: gradient
    :param params: model parameters
    :param delta: TR radius
    :param vs: TR direction
    :param weight: H + weight* I
    :return: sub_opt TR solution and model value
    
    """
    Hvs = torch.autograd.grad(grads_fun, params, grad_outputs=vs, only_inputs=True, retain_graph=True)


    #for i in Hvs: 
    #    print( i.shape )

    Hvs = [hd.detach() + weight*d for hd, d in zip(Hvs, vs)]
    #print( 'Nomr of vs: ', math.sqrt(  group_product( vs, vs ) ) )
    #print( 'Norm of Hvs: ', math.sqrt( group_product( Hvs, Hvs ) ) )
    vHv = group_product(Hvs, vs)
    #print( 'vHv: ', vHv.item () )
    vnorm = math.sqrt(group_product(vs,vs))
    if vHv < 0:
        #print('NC')
        vs = [v*delta/vnorm  for v in vs]
        #print( 'Grad * vs: ', group_product(grads_data, vs).item () )
        #print( 'vHv term: ', vHv *0.5 *delta*delta/vnorm/vnorm )
        m = vHv *0.5 *delta*delta/vnorm/vnorm - group_product(grads_data, vs)
        #print('alpha:', delta/vnorm)
        return vs, m.item()
    else:
        gv = group_product(vs, grads_data)
        #print( ' grad-vec ', gv )
        alpha =  gv/(vHv + 1e-6)
        alpha = min(alpha.item(), delta/(vnorm+1e-16))
        #print('alpha:', alpha)
        # print(alpha, delta, vnorm)
        vs = [v * alpha for v in vs]
        m = vHv *0.5 *alpha*alpha - gv *alpha
        return vs, m.item()



def cg_steihaug(grads_fun, grads_data, params, delta, weight=0.0, max_iters=25, tol=1e-6, p0=None):
    
    # init
    if p0 == None:
        zs = [0.0*p.data for p in params]
        rs = [g+0.0for g in grads_data]
        ds = [-g+0.0 for g in grads_data]
    else:
        zs = p0
        Hzs = torch.autograd.grad(grads_fun, params, grad_outputs=zs, only_inputs=True, retain_graph=True)
        Hzs = [hd.data + weight*d for hd, d in zip(Hzs, zs)]
        rs = [hz + g for hz, g in zip(Hzs, grads_data)]
        ds = [-r+0.0 for r in rs]
        
    for i in range(max_iters):
        Hd = torch.autograd.grad(grads_fun, params, grad_outputs=ds, only_inputs=True, retain_graph=True)
        Hd = [hd.data + weight*d for hd, d in zip(Hd, ds)]
        dHd = group_product(Hd, ds)
        # print(dHd, group_product(ds,ds), group_product(Hd,Hd))
        if dHd <= 0.0:
            #print (dHd)
            ac = group_product(ds, ds)
            bc = 2 * group_product(zs, ds)
            cc = group_product(zs, zs) - delta*delta
            tau = (-bc + math.sqrt(bc*bc - 4*ac*cc))/(2*ac+1e-6)
            flag = "Negative Curvature"
            ps = [z + tau*d+0.0 for (z, d) in zip(zs, ds)]
            break
        
        rnorm_square = group_product(rs, rs)
        alpha = rnorm_square/(dHd + 1e-6)
        zs_next = [z + alpha*d+0.0 for z,d in zip(zs, ds)]
        zs_next_norm = math.sqrt(group_product(zs_next, zs_next))
        if zs_next_norm >= delta:
            ac = group_product(ds, ds)
            bc = 2 * group_product(zs, ds)
            cc = group_product(zs, zs) - delta*delta
            tau = (-bc + math.sqrt(bc*bc - 4*ac*cc))/(2*ac+1e-6)
            flag = "Hit Boundary"
            ps = [z + tau*d+0.0 for (z, d) in zip(zs, ds)]
            break
        zs = [z+0.0 for z in zs_next]
        rs = [r + alpha*hd+0.0 for r, hd in zip(rs, Hd)]
        rnext_norm_square = group_product(rs, rs)
        if rnext_norm_square < tol:
            flag = "Small Residue"
            ps = [z+ 0.0 for z in zs]
            break
        beta = rnext_norm_square/(rnorm_square+1e-6)
        ds = [-r + beta*d+0.0 for r, d in zip(rs, ds)]
    if i == max_iters - 1:
        flag = "Maximum Iterations"
        ps = [z+0.0 for z in zs]
    Hp = torch.autograd.grad(grads_fun, params, grad_outputs=ps, only_inputs=True, retain_graph=True)
    Hp = [hp.data+0.0 for hp in Hp]
    m_obj = 0.5*group_product(Hp, ps) + group_product(ps, grads_data)+ 0.5*weight*group_product(ps,ps)
    return ps, m_obj, i+1, flag






def test(model, test_loader, loss_f, cuda=True):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if cuda:
            data, target = data.type(torch.DoubleTensor).cuda(), target.type(torch.LongTensor).cuda()
        output = model(data)
        test_loss += loss_f(output, target).item()*data.size()[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    acc = 1.0 * correct.item() / len(test_loader.dataset)
    model.train()
    return test_loss, acc
