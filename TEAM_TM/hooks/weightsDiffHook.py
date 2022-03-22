from mmcv.runner import Hooks, Hook
import torch
from datetime import datetime
import numpy as np
from numpy import linalg as LA


@HOOKS.register_module()
class ShowWeightsDiffHook(Hook):

    def __init__(self,cfg_path='' , show_weights_from_layers=['']  , print_model=False):

        self.cfg_path = cfg_path
        self.layer=show_weights_from_layers
        self.print_model=print_model

        self.first_layer_weights = []
        self.first_layer_norm = -1
        self.last_layer_weights = []
        self.first_epoch=-1





    def after_train_epoch(self,runner):
        def flat_vector(weights):
            flat_weights = []
            for i in weights:
                for curr in i :
                    flat_weights.extend(list(curr.flat))
            return flat_weights

        def get_norm(flat_weights):
            norm =LA.norm(flat_weights)
            return norm

        def subtruct_vectors(curr_vec, last_vec):
            return np.subtract(curr_vec,last_vec)

        if self.first_epoch==-1:
            self.first_epoch=runner.epoch

        if self.print_model==True:
            print(runner.model.module)


        if self.layer!=['']:
            for freeze_layer in self.freeze_layers:
                layers=eval('runner.model.module.'+ self.layer)

                weights = layers.weight.cpu().data.numpy()
                curr_flatten_weights =flat_vector(weights)
                if(self.first_layer_norm==-1):
                    curr_norm = get_norm(curr_flatten_weights)
                    self.first_layer_norm=curr_norm
                    self.first_layer_weights=curr_flatten_weights





                diff_curr_from_last_layer_weights=subtruct_vectors(curr_flatten_weights,self.last_layer_weights)
                norm_of_diff_from_last_epoch = get_norm(diff_curr_from_last_layer_weights)
                differnce_from_last_epoch = 100*(norm_of_diff_from_last_epoch/self.first_layer_norm)

                diff_curr_from_earlier_layer_weights = subtruct_vectors(curr_flatten_weights,self.first_layer_norm)

                norm_of_diff_from_earlier_epoch = get_norm(diff_curr_from_earlier_layer_weights)

                differnce_from_earlier_epoch = 100* (norm_of_diff_from_earlier_epoch / self.first_layer_norm)

                tmp_str = f'Layer:{self.layer}, Epoch: {runner.epoch} , Difference from precious epoch: %.7f' %differnce_from_last_epoch + f'%' + f'Difference from epoch{self.first_epoch}:%.7f' %differnce_from_earlier_epoch+f'%'


                print(tmp_str)

                self.last_layer_weights=curr_flatten_weights




