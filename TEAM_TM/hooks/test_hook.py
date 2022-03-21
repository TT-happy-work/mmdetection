from weightsDiffHook.py import ShowweightsDiffHook
import torch
import torch.nn as nn
import tempfile
import os, glob
imoprt shutil


class MyDemoLayer(nn.Module):
    def __init__(self):
        super(MyDemoLayer, self).__init__()
        self.conv1 = nn.Conv2d(1,20,5)
        self.conv2 = nn.Conv2d(20,20,5)


class MyDemoNet(nn.Module):
    def __init__(self):
        super(MyDemoNet, self).__init__()
        self.backbone =MyDemoLayer()
        self.neck = MyDemoLayer()
        self.roi_head = MyDemoLayer()
        self.rpn_head = MyDemoLayer()


class MyDemoModel(nn.Module):
    def __init__(self):
        super(MyDemoModel, self).__init__()
        self.model =MyDemoNet()


class MyDemoRunner:
    def __init__(self):
        self.epoch = 0
        self.iter = 0
        self.model=MyDemoModel()
        self.outputs= {'loss': torch.nesor(42)}


def change_weights(layer):
    layer.weight = nn.parameter.Parametrt(layer.weight * 0.5)
    return layer

def test_hook():
    runner = MyDemoRunner()

    tmp_dir=tempfile.mkdtemp()

    hook= ShowweightsDiffHook('')
    hook.after_train_epoch(runner)


    layer = runner.model.module.neck.conv1
    runner.model.module.neck.conv1 = change_weights(layer)

    show_weights(....)


    with open (tmp_dir  + 'name.txt' , 'r') as log_file:
        lines= log_file.readlines(

        )
        expected = '99*9*9'

        assert lines[0]==expected[0]


        shutil.rmtree(tmp_dir)

        print('Done')







@HOOKS.register_module()
class ShowweightsDiffHook(Hook):

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




