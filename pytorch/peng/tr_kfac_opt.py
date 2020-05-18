import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import *

class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

    
# https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L87
def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))

# TODO: In order to make this code faster:
# 1) Implement _extract_patches as a single cuda kernel
# 2) Compute QR decomposition in a separate process
# 3) Actually make a general KFAC optimizer so it fits PyTorch


def _extract_patches(x, kernel_size, stride, padding):
    if padding[0] + padding[1] > 0:
        x = F.pad(x, (padding[1], padding[1], padding[0],
                      padding[0])).data  # Actually check dims
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
    x = x.view(
        x.size(0), x.size(1), x.size(2),
        x.size(3) * x.size(4) * x.size(5))
    return x


def compute_cov_a(a, classname, layer_info, fast_cnn):
    batch_size = a.size(0)

    if classname == 'Conv2d':
        if fast_cnn:
            a = _extract_patches(a, *layer_info)
            a = a.view(a.size(0), -1, a.size(-1))
            a = a.mean(1)
        else:
            a = _extract_patches(a, *layer_info)
            a = a.view(-1, a.size(-1)).div_(a.size(1)).div_(a.size(2))
            #print( 'Z Long form size: ', a[ 0, : ] )
            #exit ()
    elif classname == 'AddBias':
        is_cuda = a.is_cuda
        a = torch.ones(a.size(0), 1)
        if is_cuda:
            a = a.cuda()

    return a.t() @ (a / batch_size)


def compute_cov_g(g, classname, layer_info, fast_cnn):
    batch_size = g.size(0)
    #print( g.size () )
    #print( 'Raw Lambda: ', math.sqrt( group_product( g, g) ) )
    
    if classname == 'Conv2d':
        if fast_cnn:
            g = g.view(g.size(0), g.size(1), -1)
            g = g.sum(-1)
        else:
            g = g.transpose(1, 2).transpose(2, 3).contiguous()
            g = g.view(-1, g.size(-1)).mul_(g.size(1)).mul_(g.size(2))
    elif classname == 'AddBias':
        g = g.view(g.size(0), g.size(1), -1)
        g = g.sum(-1)

    g_ = g * batch_size
    return g_.t().mm(g_ / g.size(0))


def update_running_stat(aa, m_aa, momentum):
    # m_aa = m_aa * momentum + aa * (1 - momentum)
    # Do the trick to keep aa unchanged and not create any additional tensors
    m_aa *= momentum / (1 - momentum)
    m_aa += aa
    m_aa *= (1 - momentum)



class SplitBias(nn.Module):
    def __init__(self, module):
        super(SplitBias, self).__init__()
        self.module = module
        self.add_bias = AddBias(module.bias.data)
        self.module.bias = None

    def forward(self, input):
        x = self.module(input)
        x = self.add_bias(x)
        return x

SubProblems = {'tr': hessian_clip, 'cr': hessian_clip_cr}

class KFACOptimizer(optim.Optimizer):
    def __init__(self,
                 model,
                 scheme='tr',
                 momentum=0,
                 check_grad=0,
                 stat_decay=0.99,
                 kl_clip=0.001,
                 damping=1e-2,
                 weight_decay=0,
                 fast_cnn=False,
                 Ts=1,
                 Tf=10,
                 max_delta=100,
                 min_delta=1e-6,
                ):
        defaults = dict()

        def split_bias(module):
            for mname, child in module.named_children():
                if hasattr(child, 'bias'):
                    module._modules[mname] = SplitBias(child)
                else:
                    split_bias(child)

        # split_bias(model)

        super(KFACOptimizer, self).__init__(model.parameters(), defaults)

        self.known_modules = {'Linear', 'Conv2d', 'AddBias'}

        self.modules = []
        self.grad_outputs = {}

        self.model = model
        self._prepare_model()

        self.steps = 0

        self.m_aa, self.m_gg = {}, {}
        self.Q_a, self.Q_g = {}, {}
        self.d_a, self.d_g = {}, {}
        self.aa_inv, self.gg_inv = {},{}
        # self.m_a, self.m_g = {}, {}
        self.vs = []
        self.momentums = []
        
        self.momentum = momentum
        self.check_grad = check_grad
        self.stat_decay = stat_decay
        
        self.acc_stats = True

        self.kl_clip = kl_clip # for TR radius or CR parameter
        self.hessian_clip = SubProblems[scheme]
        
        self.damping = damping
        self.weight_decay = weight_decay

        self.fast_cnn = fast_cnn

        self.Ts = Ts
        self.Tf = Tf
        
        self.max_delta = max_delta
        self.min_delta = min_delta
        
        self.cur_loss = None
        self.next_loss = None
        self.q_model_change = None



    def _save_input(self, module, input):
        if self.model.training and torch.is_grad_enabled() and self.steps % self.Ts == 0:
            classname = module.__class__.__name__
            layer_info = None
            if classname == 'Conv2d':
                layer_info = (module.kernel_size, module.stride,
                              module.padding)
            tmp = input[0].clone()
            aa = compute_cov_a(input[0].data, classname, layer_info,
                               self.fast_cnn)
            # assert(torch.norm(tmp.data - input[0].data) == 0)
            aa = aa.detach()
            # self.m_a[module] = a.clone()
            # Initialize buffers
            if self.steps == 0:
                self.m_aa[module] = aa.clone()
            
            update_running_stat(aa, self.m_aa[module], self.stat_decay)

    def _save_grad_output(self, module, grad_input, grad_output):
        if self.model.training and self.acc_stats:
            classname = module.__class__.__name__
            layer_info = None
            if classname == 'Conv2d':
                layer_info = (module.kernel_size, module.stride,
                              module.padding)
            tmp = grad_output[0].clone()
            gg = compute_cov_g(grad_output[0].data, classname, layer_info,
                               self.fast_cnn)
            # assert(torch.norm(tmp.data - grad_output[0].data) == 0)
            gg = gg.detach()
            # self.m_g[module] = g.clone()
            # Initialize buffers
            if self.steps == 0:
                self.m_gg[module] = gg.clone()

            update_running_stat(gg, self.m_gg[module], self.stat_decay)

    def _prepare_model(self):
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                assert not ((classname in ['Linear', 'Conv2d']) and module.bias is not None), \
                                    "You must have a bias as a separate layer"

                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)

    def step(self, closure=None):
        # Add weight decay
        if self.weight_decay > 0:
            for p in self.model.parameters():
                p.grad.data.add_(self.weight_decay, p.data)
                # p.grad.add_(self.weight_decay, p)
        if closure:
            self.cur_loss = closure()
        
        updates = {}
        for i, m in enumerate(self.modules):
            assert len(list(m.parameters())) == 1, "Can handle only one parameter at the moment"
            classname = m.__class__.__name__
            p = next(m.parameters())

            la = self.damping + self.weight_decay
            #print( 'Z[i] :', math.sqrt( group_product( self.m_aa[m], self.m_aa[m] ) ) )
            #print( 'D[i] :', math.sqrt( group_product( self.m_gg[m], self.m_gg[m] ) ) )

            if self.steps % self.Tf == 0:
                # My asynchronous implementation exists, I will add it later.
                # Experimenting with different ways to this in PyTorch.
                # self.d_a[m], self.Q_a[m] = torch.symeig(
                #     self.m_aa[m], eigenvectors=True)
                # self.d_g[m], self.Q_g[m] = torch.symeig(
                #     self.m_gg[m], eigenvectors=True)
                # self.d_a[m].mul_((self.d_a[m] > 1e-6).float())
                # self.d_g[m].mul_((self.d_g[m] > 1e-6).float())
                if p.is_cuda:
                    self.aa_inv[m] = torch.inverse(self.m_aa[m] + la*torch.eye(self.m_aa[m].shape[0]).type( torch.cuda.DoubleTensor ) )
                    self.gg_inv[m] = torch.inverse(self.m_gg[m] + la*torch.eye(self.m_gg[m].shape[0]).type( torch.cuda.DoubleTensor ))
                else:
                    self.aa_inv[m] = torch.inverse(self.m_aa[m] + la*torch.eye(self.m_aa[m].shape[0]).type (torch.cuda.DoubleTensor))
                    self.gg_inv[m] = torch.inverse(self.m_gg[m] + la*torch.eye(self.m_gg[m].shape[0]).type( torch.cuda.DoubleTensor ))

            if classname == 'Conv2d':
                p_grad_mat = p.grad.data.view(p.grad.data.size(0), -1)
            else:
                p_grad_mat = p.grad.data
            # v1 = self.Q_g[m].t() @ p_grad_mat @ self.Q_a[m]
            # v2 = v1 / (
            #     self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) + la)
            # v = self.Q_g[m] @ v2 @ self.Q_a[m].t()
            #v = self.gg_inv[m] @ p_grad_mat @ self.aa_inv[m]
            v = self.gg_inv[m] @ p_grad_mat @ self.aa_inv[m]
            v = v.view(p.grad.data.size())
            #print( math.sqrt( group_product(self.gg_inv[m], self.gg_inv[m] ) ), math.sqrt( group_product(p_grad_mat, p_grad_mat  ) ), math.sqrt( group_product(self.aa_inv[m], self.aa_inv[m]  ) ) )
            updates[p] = v

        vs = []
        grads_fun = []
        grads_data = []
        for p in self.model.parameters():
            vs.append(updates[p])
            grads_fun.append(p.grad)
            grads_data.append(p.grad.data)
        self.vs = vs
        #print( 'grad norm', math.sqrt( group_product( grads_data, grads_data ) ) )
        #print( 'weights norm', math.sqrt( group_product( self.model.parameters (), self.model.parameters () ) ) )
        #print( 'norm of vs (x_kfac): ', math.sqrt( group_product( vs, vs ) ) )

        assert(group_product(vs, grads_data) >=0)
        vs, self.q_model_change = self.hessian_clip(grads_fun, grads_data, list(self.model.parameters()), self.kl_clip, vs, self.weight_decay)
        if self.check_grad != 0:
            gs, q_model_change_grad = self.hessian_clip(grads_fun, grads_data, list(self.model.parameters()), self.kl_clip, grads_data, self.weight_decay)
            #print( 'V Model Change: ', self.q_model_change )
            #print( 'G Model Change: ',q_model_change_grad )
				
            if self.q_model_change > q_model_change_grad:
                vs = gs
                self.q_model_change = q_model_change_grad
                #print("grad")
        if len(self.momentums)>0 and self.momentum != 0:
            # momentums, q_model_change_mom = self.hessian_clip(grads_fun, grads_data, list(self.model.parameters()), self.kl_clip, self.momentums, self.weight_decay)
            # if self.q_model_change > q_model_change_mom:
            #     vs = momentums
            #     self.q_model_change = q_model_change_mom
            #     print("momentum")
            for v, mom in zip(vs, self.momentums):
                v.data.add_(self.momentum, mom)
        self.momentums = []
        for i, p in enumerate(self.model.parameters()):
            v = vs[i]
            # p.grad.data.copy_(v)
            p.data.add_(-1.0, v)
            self.momentums.append(vs[i])

        # self.optim.step()
        if closure:
            self.next_loss = closure()
            rho = (self.next_loss - self.cur_loss)/(self.q_model_change-1e-16)
            if rho > 0.75:
                self.kl_clip = min(self.max_delta, 2.0*self.kl_clip)
            if rho < 0.25:
                self.kl_clip = max(self.min_delta, self.kl_clip/2.0)
            # a safeguard
            if rho < 1e-4 or self.next_loss > 10*self.cur_loss:
                #print("rejection")
                for i, p in enumerate(self.model.parameters()):
                    p.data.add_(1.0, vs[i])
                #print(self.cur_loss, closure())
            #print(rho, self.kl_clip, self.next_loss, self.cur_loss, self.q_model_change)
            #print( '%4.10e  %4.10e  %4.10e  %4.10e %4.10e  %6s  %3.6e' %(self.cur_loss, self.next_loss, rho, self.q_model_change, math.sqrt( group_product( self.model.parameters (), self.model.parameters () ) ), 'kfac', self.kl_clip) )

				
        self.steps += 1
