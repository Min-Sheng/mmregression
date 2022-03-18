# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

from ..builder import HEADS, build_loss
from ..utils import is_tracing
from .base_head import BaseHead


@HEADS.register_module()
class RegHead(BaseHead):
    """regression head.

    Args:
        loss (dict): Config of regression loss.
    """

    def __init__(self,
                 loss=dict(type='MSELoss', loss_weight=1.0),
                 init_cfg=None):
        super(RegHead, self).__init__(init_cfg=init_cfg)

        assert isinstance(loss, dict)

        self.compute_loss = build_loss(loss)

    def loss(self, reg_score, gt_label, **kwargs):
        losses = dict()
        # compute loss
        loss = self.compute_loss(
            reg_score, gt_label, **kwargs)
        losses['loss'] = loss
        return losses

    def forward_train(self, reg_score, gt_label, **kwargs):
        if isinstance(reg_score, tuple):
            reg_score = reg_score[-1]
        losses = self.loss(reg_score, gt_label, **kwargs)
        return losses

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        return x

    def simple_test(self, reg_score, post_process=True):
        """Inference without augmentation.

        Args:
            reg_score (tuple[Tensor]): The input regression score logits.
                Multi-stage inputs are acceptable but only the last stage will
                be used to regress. The shape of every item should be
                ``(num_samples, in_channels, *)``.
            post_process (bool): Whether to do post processing the
                inference results. It will convert the output to a list.

        Returns:
            Tensor | list: The inference results.

                - If no post processing, the output is a tensor with shape
                  ``(num_samples, num_channels, *)``.
                - If post processing, the output is a multi-dimentional list of
                  float and the dimensions are ``(num_samples, num_channels, *)``.
        """
        if isinstance(reg_score, tuple):
            pred = reg_score[-1]

        if post_process:
            return self.post_process(pred)
        else:
            return pred

    def post_process(self, pred):
        on_trace = is_tracing()
        if torch.onnx.is_in_onnx_export() or on_trace:
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred
