# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from ..builder import NECKS


@NECKS.register_module()
class AdaptiveAveragePooling(nn.Module):
    """Adaptive Pooling neck.

    Args:

        dim (int): Dimensions of each sample channel, can be one of {1, 2, 3}.
            Default: 2
        output_size (int | tuple):  output size,
            If dim equals to 1: output_size is a single integer.
            Else, if output_size is a tuple of integers with the length of dim.
            Default: (5, 4)
    """

    def __init__(self, dim=2, output_size=(5,4)):
        super(AdaptiveAveragePooling, self).__init__()
        assert dim in [1, 2, 3], 'AdaptiveAveragePooling dim only support ' \
            f'{1, 2, 3}, get {dim} instead.'
        if dim == 1:
            assert isinstance(output_size, int)
            self.aap = nn.AdaptiveAvgPool1d(output_size)
        elif dim == 2:
            assert isinstance(output_size, tuple)
            assert len(output_size) == 2
            self.aap = nn.AdaptiveAvgPool2d(output_size)
        else:
            assert isinstance(output_size, tuple)
            assert len(output_size) == 3
            self.aap = nn.AdaptiveAvgPool3d(output_size)

        self.output_size = output_size

    def init_weights(self):
        pass

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            outs = tuple([self.aap(x) for x in inputs])
        elif isinstance(inputs, torch.Tensor):
            outs = self.aap(inputs)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs
