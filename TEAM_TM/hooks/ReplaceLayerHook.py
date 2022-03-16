# Copyright (c) OpenMMLab. All rights reserved.
import numbers
from math import cos, pi

import mmcv
from mmcv.runner.hooks.hook import HOOKS, Hook

@HOOKS.register_module()
class ReplaceLayerHook(Hook):
    """Replace from Nth layer in MMCV.

    Args:
        by_epoch (bool): LR changes epoch by epoch
        warmup (string): Type of warmup used. It can be None(use no warmup),
            'constant', 'linear' or 'exp'
        warmup_iters (int): The number of iterations or epochs that warmup
            lasts
        warmup_ratio (float): LR used at the beginning of warmup equals to
            warmup_ratio * initial_lr
        warmup_by_epoch (bool): When warmup_by_epoch == True, warmup_iters
            means the number of epochs that warmup lasts, otherwise means the
            number of iteration that warmup lasts
    """

    def __init__(self,
                 work_dir,
                 cfg_path,
                 which_layer_name,
                 new_layers_amount,
                 new_layers_filter_size,
                 results_dir_name='filter_hook_results'):



        self.work_dir = work_dir
        self.cfg_path = cfg_path
        self.which_layer_name = which_layer_name
        self.new_layers_amount = new_layers_amount
        self.new_layers_filter_size = new_layers_filter_size
        self.results_dir_name = results_dir_name


        print('HI')