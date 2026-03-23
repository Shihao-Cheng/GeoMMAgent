# Minimal DeepLabV3+ with Xception backbone (from DeepLabV3Plus-Pytorch).

from .utils import IntermediateLayerGetter
from ._deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3
from .backbone import xception


def _segm_xception(
    name,
    backbone_name,
    num_classes,
    output_stride,
    pretrained_backbone,
):
    if output_stride == 8:
        replace_stride_with_dilation = [False, False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation = [False, False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = xception.xception(
        pretrained="imagenet" if pretrained_backbone else False,
        replace_stride_with_dilation=replace_stride_with_dilation,
    )

    inplanes = 2048
    low_level_planes = 128

    if name == "deeplabv3plus":
        return_layers = {"conv4": "out", "block1": "low_level"}
        classifier = DeepLabHeadV3Plus(
            inplanes, low_level_planes, num_classes, aspp_dilate
        )
    elif name == "deeplabv3":
        return_layers = {"conv4": "out"}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)
    else:
        raise ValueError(name)

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = DeepLabV3(backbone, classifier)
    return model


def deeplabv3plus_xception(
    num_classes=21,
    output_stride=8,
    pretrained_backbone=True,
):
    return _segm_xception(
        "deeplabv3plus",
        "xception",
        num_classes,
        output_stride=output_stride,
        pretrained_backbone=pretrained_backbone,
    )
