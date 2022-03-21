from mmcv.runner import Hooks, Hook

@HOOKS.register_module()
class FreezeHook(Hook):

    def __init__(self,cfg_path='' , freeze_layers=['']  , print_model=False):

        self.cfg_path = cfg_path
        self.freeze_layers=freeze_layers
        self.print_model=print_model

    def before_train_iter(self,runner):

        if runner.print_model==True:
            print(runner.model.module)

        if self.freeze_layers!=['']:
            for freeze_layer in self.freeze_layers:
                layers=eval('runner.model.module.'+ freeze_layer+'.parameters()')
                for layer in layers:
                    if 'requires_grad' in layer.__dir__():
                        layer.requires_grad=False
                    if 'training' in layer.__dir__():
                        layer.training = False