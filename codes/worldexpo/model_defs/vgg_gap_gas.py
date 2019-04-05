import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math


__all__ = [
    'VGG', 'VGG_GAP_GAS', 'vgg16', 'vgg_gap_gas',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class VGG_GAP_GAS(nn.Module):

    def __init__(self, model_vgg16, input_size):
        super(VGG_GAP_GAS, self).__init__()

        self.tmp = list(model_vgg16.features.children());
        for i in xrange(8) :
            self.tmp.pop();

        # replace relu layers with prelu
        self.replace_relu_with_prelu();

        self.features_1 = nn.Sequential(*self.tmp);
        self.features_2 = nn.AvgPool2d(kernel_size=input_size, stride=input_size); # 45x80
        self.classifier = nn.Linear(in_features=512, out_features=1);

    def replace_relu_with_prelu(self) :
        id_relu = [1,3,6,8,11,13,15,18,20,22];
        for i in id_relu :
            self.tmp[i] = nn.PReLU(self.tmp[i-1].out_channels);


    def forward(self, x):
        x = self.features_1(x);
        x = x.abs();
        count = self.features_2(x);
        count = count.view(count.size(0), -1);
        count = self.classifier(count);
        x = x.view(x.size(0), x.size(1), -1);
        x = x.mul(torch.autograd.Variable(self.classifier.weight.data.unsqueeze(2)));
        x = x.sum(1);
        x = x.abs();
        max_, _ = x.data.max(1);
        x.data.div_(max_.unsqueeze(1).expand_as(x));

        return count, x;


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'GAS': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, ],
}


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model


def vgg_gap_gas(pretrained, input_size) :
    """VGG16-layer model (configuration "GAS") upto 4-3 layer and GAP-
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = vgg16(pretrained=pretrained)
    model = VGG_GAP_GAS(model, input_size=input_size);
    return model
