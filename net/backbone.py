import torch.nn as nn
import torchvision


def init_weights(module, init_method='He'):
    for _, m in module.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if init_method == 'He':
                nn.init.kaiming_normal_(m.weight.data)
            elif init_method == 'Xavier':
                nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, val=0)


def frozen_layer(module):
    for params in module.parameters():
        params.required_grad = False


def unfrozen_layer(module):
    for params in module.parameters():
        params.required_grad = True


class BaseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Module()
        self.avgpool = nn.Module()
        self.classifier = nn.Module()

    def forward(self, x):
        N = x.size(0)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(N, -1)
        x = self.classifier(x)
        return {'logits': x}

    def extract_features(self, x, vector=True):
        N = x.size(0)
        x = self.features(x)
        if vector:
            x = self.avgpool(x)
            x = x.view(N, -1)
        return x


class ResNet(BaseNet):
    def __init__(self, arch='resnet18', n_classes=200, pretrained=True, fc_init='He'):
        super().__init__()
        self.arch = arch
        self._pretrained = pretrained
        self._n_classes = n_classes

        assert arch in torchvision.models.__dict__.keys(), f'{arch} is not supported!'

        resnet = torchvision.models.__dict__[self.arch](pretrained=self._pretrained)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.feat_dim = resnet.fc.in_features
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Linear(in_features=self.feat_dim, out_features=self._n_classes)

        if self._pretrained:
            init_weights(self.classifier, init_method=fc_init)


class VGGNet(BaseNet):
    def __init__(self, arch='vgg16', n_classes=200, pretrained=True, fc_init='He'):
        super().__init__()
        self.arch = arch
        self._pretrained = pretrained
        self._n_classes = n_classes

        assert arch in torchvision.models.__dict__.keys(), f'{arch} is not supported!'

        vgg = torchvision.models.__dict__[self.arch](pretrained=self._pretrained)
        self.features = vgg.features
        self.feat_dim = 512
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
            *list(vgg.classifier.children())[:-1],
            nn.Linear(in_features=vgg.classifier[-1].in_features, out_features=self._n_classes)
        )

        if self._pretrained:
            init_weights(self.classifier, init_method=fc_init)
