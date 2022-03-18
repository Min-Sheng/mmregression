import torch.nn as nn

from ..builder import HEADS
from .reg_head import RegHead


@HEADS.register_module()
class ConvRegHead(RegHead):
    """Convolution regressor head.

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels in the output.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Kaiming', layer='Conv2d').
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 init_cfg=dict(type='Kaiming', layer='Conv2d'),
                 *args,
                 **kwargs):
        super(ConvRegHead, self).__init__(init_cfg=init_cfg, *args, **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        return x

    def simple_test(self, x, post_process=True):
        """Inference without augmentation.

        Args:
            x (tuple[Tensor]): The input features.
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
        x = self.pre_logits(x)
        x = self.conv(x)
        pred = self.sigmoid(x)

        if post_process:
            return self.post_process(pred)
        else:
            return pred

    def forward_train(self, x, gt_label, **kwargs):
        x = self.pre_logits(x)
        x = self.conv(x)
        pred = self.sigmoid(x)
        losses = self.loss(pred, gt_label, **kwargs)
        return losses
