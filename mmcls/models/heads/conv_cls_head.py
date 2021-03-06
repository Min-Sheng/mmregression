import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS
from .cls_head import ClsHead


@HEADS.register_module()
class ConvClsHead(ClsHead):
    """Convolution regressor head.

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels in the output.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Kaiming', layer='Conv2d').
    """
    def __init__(self,
                 num_classes,
                 in_channels,
                 init_cfg=dict(type='Kaiming', layer='Conv2d'),
                 *args,
                 **kwargs):
        super(ConvClsHead, self).__init__(init_cfg=init_cfg, *args, **kwargs)
        
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv = nn.Conv2d(self.in_channels, self.num_classes, kernel_size=1, stride=1, padding=0)

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        return x

    def simple_test(self, x, softmax=True, post_process=True):
        """Inference without augmentation.

        Args:
            x (tuple[Tensor]): The input features.
                Multi-stage inputs are acceptable but only the last stage will
                be used to classify. The shape of every item should be
                ``(num_samples, in_channels, *)``.
            softmax (bool): Whether to softmax the classification score.
            post_process (bool): Whether to do post processing the
                inference results. It will convert the output to a list.

        Returns:
            Tensor | list: The inference results.

                - If no post processing, the output is a tensor with shape
                  ``(num_samples, *, num_channels)``.
                - If post processing, the output is a multi-dimentional list of
                  float and the dimensions are ``(num_samples, *, num_channels)``.
        """
        x = self.pre_logits(x)
        cls_score = self.conv(x)

        if softmax:
            pred = F.softmax(cls_score, dim=1)

        else:
            pred = cls_score

        pred = torch.moveaxis(pred, 1, -1)

        if post_process:
            return self.post_process(pred)
        else:
            return pred

    def forward_train(self, x, gt_label, **kwargs):
        x = self.pre_logits(x)
        cls_score = self.conv(x)
        cls_score = torch.moveaxis(cls_score, 1, -1)
        cls_score = cls_score.reshape(-1, self.num_classes)
        gt_label = gt_label.view(-1)
        losses = self.loss(cls_score, gt_label, **kwargs)
        return losses
