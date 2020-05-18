import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    """Swish Function: 
    Applies the element-wise function :math:`f(x) = x / ( 1 + exp(-x))`

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::

        >>> m = Swish()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """

    def forward(self, input):
        return input * torch.sigmoid(input)

    def __repr__(self):
        return self.__class__.__name__ + '()'


## Initialization

def normal_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant(m.bias, 0.0)
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None: 
            nn.init.constant(m.bias, 0.0)

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                nn.init.constant(m.bias, 0)

        
def zeros_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # m.weight.data.normal_(0.0, 0.02)
        m.weight.data.fill_(0.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    if isinstance(m, nn.Linear):
        m.weight.data.fill_(0.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

def ones_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # m.weight.data.normal_(0.0, 0.02)
        m.weight.data.fill_(1.)
        if m.bias is not None:
            m.bias.data.fill_(1.)
    if isinstance(m, nn.Linear):
        m.weight.data.fill_(1.)
        if m.bias is not None:
            m.bias.data.fill_(1.)


######## MODELS ############

class QAlexNetS(nn.Module):
    """
    QAlexNet with Swish, Batch Normalization
    """
    def __init__(self, num_classes=10):
        super(QAlexNetS, self).__init__()
        # what is kernel_size in Conv2d,
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2, bias=False),# 32x32x3 -> 32x32x64
            Swish(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding =1 ),# 32x32x64 -> 16x16x64
            nn.Conv2d(64, 64, kernel_size=5, stride =1, padding=2, bias=False),# 16x16x64 -> 16x16x64
            Swish(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),# 16x16x64 -> 8x8x64
        )
        self.classifier = nn.Sequential(
            nn.Linear(64*8*8, 384, bias=False),
            Swish(),
            nn.Linear(384,192, bias=False),
            Swish(),
            nn.Linear(192, num_classes, bias=False),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x 

class QAlexNetSb(nn.Module):
    """
    QAlexNet with Swish, Batch Normalization
    """
    def __init__(self, num_classes=10):
        super(QAlexNetSb, self).__init__()
        # what is kernel_size in Conv2d,
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2, bias=False),# 32x32x3 -> 32x32x64
            Swish(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding =1 ),# 32x32x64 -> 16x16x64
            nn.BatchNorm2d(64, affine=False),
            nn.Conv2d(64, 64, kernel_size=5, stride =1, padding=2, bias=False),# 16x16x64 -> 16x16x64
            Swish(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),# 16x16x64 -> 8x8x64
            nn.BatchNorm2d(64, affine=False),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64*8*8, 384, bias=False),
            Swish(),
            nn.Linear(384,192, bias=False),
            Swish(),
            nn.Linear(192,num_classes, bias=False),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x 





## VGG
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGGb(nn.Module):
    def __init__(self, vgg_name, num_class=10):
        super(VGGb, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_class, bias=False)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
        # return F.log_softmax(out)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False),
                           nn.BatchNorm2d(x, affine=False),
                           Swish()]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(self, vgg_name, num_class=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_class, bias=False)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
        # return F.log_softmax(out)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False),
                           Swish()]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)











