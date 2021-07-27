import torch
import torch.nn as nn
import torchvision
from net.backbone import init_weights, frozen_layer, unfrozen_layer


class ChannelPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = lambda x: x.mean(dim=1, keepdim=True)
        self.maxpool = lambda x: x.max(dim=1, keepdim=True)[0]

    def forward(self, x):
        return torch.cat((self.maxpool(x), self.avgpool(x)), dim=1)  # N, 2, h, w


class SpatialAttention(nn.Module):
    def __init__(self, in_planes=2, out_planes=1, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=out_planes),
            nn.ReLU()
        )
        self.channel_pool = ChannelPool()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv_block(self.channel_pool(x)))  # size: N, 1, h_x, h_w


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes, reduction_ratio=4):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_planes, out_features=int(in_planes // reduction_ratio)),
            nn.ReLU(),
            nn.Linear(in_features=int(in_planes // reduction_ratio), out_features=out_planes),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        N = x.size(0)
        avg_pool = self.avgpool(x).view(N, -1)
        max_pool = self.maxpool(x).view(N, -1)
        channel_att = self.mlp(avg_pool) + self.mlp(max_pool)
        channel_att = self.sigmoid(channel_att)
        channel_att = channel_att.unsqueeze(2).unsqueeze(3)
        return channel_att  # size: N, out_planes, 1, 1


class ClarResNet(nn.Module):
    def __init__(self, arch='resnet18', r=2, n_classes=200, pretrained=True, fc_init='He'):
        super().__init__()
        assert arch in torchvision.models.__dict__.keys(), f'{arch} is not supported!'
        self.arch = arch
        self._pretrained = pretrained
        self._n_classes = n_classes

        resnet = torchvision.models.__dict__[self.arch](pretrained=self._pretrained)
        self.input_stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.stage1 = resnet.layer1  # 64
        self.stage2 = resnet.layer2  # 128
        self.stage3 = resnet.layer3  # 256
        self.stage4 = resnet.layer4  # 512

        self.feat_dim = resnet.fc.in_features
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Linear(in_features=int(self.feat_dim*1.5), out_features=self._n_classes)

        # attention module
        self.spatial_gate = SpatialAttention(in_planes=2, out_planes=1, kernel_size=3, stride=2, padding=1)                  # extract from stage 3
        self.channel_gate = ChannelAttention(in_planes=self.feat_dim, out_planes=int(self.feat_dim*0.5), reduction_ratio=r)  # extract from stage 4

        if self._pretrained:
            init_weights(self.classifier, init_method=fc_init)
        init_weights(self.spatial_gate, init_method=fc_init)
        init_weights(self.channel_gate, init_method=fc_init)

    def forward(self, x):
        N = x.size(0)   # assume x.size: N, 3, 128, 128
        x = self.input_stem(x)  # N, _, 64, 64
        x = self.stage1(x)      # N, 64, 32, 32
        x = self.stage2(x)      # N, 128, 16, 16
        x3 = self.stage3(x)     # N, 256, 8, 8
        x4 = self.stage4(x3)    # N, 512, 4, 4

        x3, x4 = self.clar(x3, x4)
        x = torch.cat([self.avgpool(x3), self.avgpool(x4)], dim=1)
        x = x.view(N, -1)
        x = self.classifier(x)
        return {'logits': x}

    def clar(self, xl, xh):
        spatial_attention_mask = self.spatial_gate(xl)  # (N, 1, 4, 4), spatial attention from lower lever layer
        xh = xh + xh * spatial_attention_mask
        channel_attention_mask = self.channel_gate(xh)  # (N, 256, 1, 1), channel attention from higher level layer (SEN)
        xl = xl + xl * channel_attention_mask.expand_as(xl)
        return xl, xh
