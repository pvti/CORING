from typing import Any, Callable, List, Optional
import torch
import torch.nn as nn

from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.keypoint_rcnn import KeypointRCNN
from torchvision.models._utils import _ovewrite_value_param
from torchvision.models.detection.backbone_utils import (
    _validate_trainable_layers,
    BackboneWithFPN,
)
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock, LastLevelMaxPool

import sys

sys.path.append("..")
from main.models.imagenet.resnet import resnet_50, ResNet50


def fasterrcnn_resnet50_fpn(
    *,
    weights=None,
    num_classes: Optional[int] = None,
    weights_backbone=None,
    compress_rate=[0.0] * 53,
    trainable_backbone_layers: Optional[int] = None,
    **kwargs: Any,
) -> FasterRCNN:
    """
    Simplified from https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py

    Faster R-CNN model with a ResNet-50-FPN backbone from the `Faster R-CNN: Towards Real-Time Object
    Detection with Region Proposal Networks <https://arxiv.org/abs/1506.01497>`__
    paper.

    .. betastatus:: detection module

    Args:
        weights (:class:`~torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background)
        weights_backbone (:class:`~torchvision.models.ResNet50_Weights`, optional): The
            pretrained weights for the backbone.
        trainable_backbone_layers (int, optional): number of trainable (not frozen) layers starting from
            final block. Valid values are between 0 and 5, with 5 meaning all backbone layers are
            trainable. If ``None`` is passed (the default) this value is set to 3.
        **kwargs: parameters passed to the ``torchvision.models.detection.faster_rcnn.FasterRCNN``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights
        :members:
    """

    if weights is not None:
        weights_backbone = None
        num_classes = 91  # always use COCO-2017
    elif num_classes is None:
        num_classes = 91

    is_trained = weights is not None or weights_backbone is not None
    trainable_backbone_layers = _validate_trainable_layers(
        is_trained, trainable_backbone_layers, 5, 3
    )

    backbone = resnet_50(compress_rate=compress_rate)
    if weights_backbone is not None:
        ckpt = torch.load(weights_backbone)
        backbone.load_state_dict(ckpt["state_dict"])
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)

    model = FasterRCNN(backbone, num_classes=num_classes, **kwargs)

    if weights is not None:
        ckpt = torch.load(weights)
        model.load_state_dict(ckpt["model"])

    return model


def maskrcnn_resnet50_fpn(
    *,
    weights=None,
    num_classes: Optional[int] = None,
    weights_backbone=None,
    compress_rate=[0.0] * 53,
    trainable_backbone_layers: Optional[int] = None,
    **kwargs: Any,
) -> MaskRCNN:
    """
    Simplified from https://github.com/pytorch/vision/blob/main/torchvision/models/detection/mask_rcnn.py

    Mask R-CNN model with a ResNet-50-FPN backbone from the `Mask R-CNN
    <https://arxiv.org/abs/1703.06870>`_ paper.

    .. betastatus:: detection module

    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.

    The behavior of the model changes depending on if it is in training or evaluation mode.

    During training, the model expects both the input tensors and targets (list of dictionary),
    containing:

        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box
        - masks (``UInt8Tensor[N, H, W]``): the segmentation binary masks for each instance

    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses for both the RPN and the R-CNN, and the mask loss.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows, where ``N`` is the number of detected instances:

        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the predicted labels for each instance
        - scores (``Tensor[N]``): the scores or each instance
        - masks (``UInt8Tensor[N, 1, H, W]``): the predicted masks for each instance, in ``0-1`` range. In order to
          obtain the final segmentation masks, the soft masks can be thresholded, generally
          with a value of 0.5 (``mask >= 0.5``)

    For more details on the output and on how to plot the masks, you may refer to :ref:`instance_seg_output`.

    Mask R-CNN is exportable to ONNX for a fixed batch size with inputs images of fixed size.

    Example::

        >>> model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
        >>>
        >>> # optionally, if you want to export the model to ONNX:
        >>> torch.onnx.export(model, x, "mask_rcnn.onnx", opset_version = 11)

    Args:
        weights (:class:`~torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background)
        weights_backbone (:class:`~torchvision.models.ResNet50_Weights`, optional): The
            pretrained weights for the backbone.
        trainable_backbone_layers (int, optional): number of trainable (not frozen) layers starting from
            final block. Valid values are between 0 and 5, with 5 meaning all backbone layers are
            trainable. If ``None`` is passed (the default) this value is set to 3.
        **kwargs: parameters passed to the ``torchvision.models.detection.mask_rcnn.MaskRCNN``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/detection/mask_rcnn.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights
        :members:
    """

    if weights is not None:
        weights_backbone = None
        num_classes = 91
    elif num_classes is None:
        num_classes = 91

    is_trained = weights is not None or weights_backbone is not None
    trainable_backbone_layers = _validate_trainable_layers(
        is_trained, trainable_backbone_layers, 5, 3
    )

    backbone = resnet_50(compress_rate=compress_rate)
    if weights_backbone is not None:
        ckpt = torch.load(weights_backbone)
        backbone.load_state_dict(ckpt["state_dict"])
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)

    model = MaskRCNN(backbone, num_classes=num_classes, **kwargs)

    if weights is not None:
        ckpt = torch.load(weights)
        model.load_state_dict(ckpt["model"])

    return model


def keypointrcnn_resnet50_fpn(
    *,
    weights=None,
    num_classes: Optional[int] = None,
    num_keypoints: Optional[int] = None,
    weights_backbone=None,
    compress_rate=[0.0] * 53,
    trainable_backbone_layers: Optional[int] = None,
    **kwargs: Any,
) -> KeypointRCNN:
    """
    Simplified from https://github.com/pytorch/vision/blob/main/torchvision/models/detection/keypoint_rcnn.py

    Constructs a Keypoint R-CNN model with a ResNet-50-FPN backbone.

    .. betastatus:: detection module

    Reference: `Mask R-CNN <https://arxiv.org/abs/1703.06870>`__.

    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.

    The behavior of the model changes depending on if it is in training or evaluation mode.

    During training, the model expects both the input tensors and targets (list of dictionary),
    containing:

        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box
        - keypoints (``FloatTensor[N, K, 3]``): the ``K`` keypoints location for each of the ``N`` instances, in the
          format ``[x, y, visibility]``, where ``visibility=0`` means that the keypoint is not visible.

    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses for both the RPN and the R-CNN, and the keypoint loss.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows, where ``N`` is the number of detected instances:

        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the predicted labels for each instance
        - scores (``Tensor[N]``): the scores or each instance
        - keypoints (``FloatTensor[N, K, 3]``): the locations of the predicted keypoints, in ``[x, y, v]`` format.

    For more details on the output, you may refer to :ref:`instance_seg_output`.

    Keypoint R-CNN is exportable to ONNX for a fixed batch size with inputs images of fixed size.

    Example::

        >>> model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
        >>>
        >>> # optionally, if you want to export the model to ONNX:
        >>> torch.onnx.export(model, x, "keypoint_rcnn.onnx", opset_version = 11)

    Args:
        weights (:class:`~torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights`
            below for more details, and possible values. By default, no
            pre-trained weights are used.
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int, optional): number of output classes of the model (including the background)
        num_keypoints (int, optional): number of keypoints
        weights_backbone (:class:`~torchvision.models.ResNet50_Weights`, optional): The
            pretrained weights for the backbone.
        trainable_backbone_layers (int, optional): number of trainable (not frozen) layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable. If ``None`` is
            passed (the default) this value is set to 3.

    .. autoclass:: torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights
        :members:
    """

    if weights is not None:
        weights_backbone = None
        num_classes = 2
        num_keypoints = 17
    else:
        if num_classes is None:
            num_classes = 2
        if num_keypoints is None:
            num_keypoints = 17

    is_trained = weights is not None or weights_backbone is not None
    trainable_backbone_layers = _validate_trainable_layers(
        is_trained, trainable_backbone_layers, 5, 3
    )

    backbone = resnet_50(compress_rate=compress_rate)
    if weights_backbone is not None:
        ckpt = torch.load(weights_backbone)
        backbone.load_state_dict(ckpt["state_dict"])
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
    model = KeypointRCNN(backbone, num_classes, num_keypoints=num_keypoints, **kwargs)

    if weights is not None:
        ckpt = torch.load(weights)
        model.load_state_dict(ckpt["model"])

    return model


def _resnet_fpn_extractor(
    backbone: ResNet50,
    trainable_layers: int,
    returned_layers: Optional[List[int]] = None,
    extra_blocks: Optional[ExtraFPNBlock] = None,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
) -> BackboneWithFPN:
    """
    Modified from https://github.com/pytorch/vision/blob/main/torchvision/models/detection/backbone_utils.py
    """

    # select layers that won't be frozen
    if trainable_layers < 0 or trainable_layers > 5:
        raise ValueError(
            f"Trainable layers should be in the range [0,5], got {trainable_layers}"
        )
    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][
        :trainable_layers
    ]
    if trainable_layers == 5:
        layers_to_train.append("bn1")
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    if min(returned_layers) <= 0 or max(returned_layers) >= 5:
        raise ValueError(
            f"Each returned layer should be in the range [1,4]. Got {returned_layers}"
        )
    return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}

    in_channels_list = []
    in_channels_list.append(backbone.layer1[-1].conv3.out_channels)
    in_channels_list.append(backbone.layer2[-1].conv3.out_channels)
    in_channels_list.append(backbone.layer3[-1].conv3.out_channels)
    in_channels_list.append(backbone.layer4[-1].conv3.out_channels)

    out_channels = 256
    return BackboneWithFPN(
        backbone,
        return_layers,
        in_channels_list,
        out_channels,
        extra_blocks=extra_blocks,
        norm_layer=norm_layer,
    )
