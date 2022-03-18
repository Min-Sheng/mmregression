# Copyright (c) OpenMMLab. All rights reserved.
from .aap import AdaptiveAveragePooling
from .gap import GlobalAveragePooling
from .gem import GeneralizedMeanPooling
from .hr_fuse import HRFuseScales

__all__ = [
        'GlobalAveragePooling', 'GeneralizedMeanPooling', 
        'HRFuseScales','AdaptiveAveragePooling'
    ]
