from net.backbone import VGGNet, ResNet
from net.clar_backbone import ClarResNet


def build_model(arch, n_classes=200, pretrained=True, fc_init='He', reduction_factor=2):
    if arch.startswith('vgg'):
        model = VGGNet(arch, n_classes, pretrained, fc_init)
    elif arch.startswith('resnet'):
        model = ResNet(arch, n_classes, pretrained, fc_init)
    elif arch.startswith('clar-resnet'):
        model = ClarResNet(arch.split('-')[1], reduction_factor, n_classes, pretrained, fc_init)
    else:
        raise AssertionError(f'{arch} arch is not supported!')
    return model
