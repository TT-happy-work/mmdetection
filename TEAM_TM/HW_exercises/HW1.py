import os
import torch
import matplotlib.pyplot as plt
import mmcv
from mmdet.apis import init_detector, inference_detector

assert torch.cuda.is_available(), 'Restart Computer, Cuda is not available'
cfg_file_path = 'HW1_config.py'

example_images = ['/home/tamarbo/datasets/car_damage/val/1.jpg',
                  '/home/tamarbo/datasets/car_damage/val/22.jpg',
                  '/home/tamarbo/datasets/car_damage/val/3.jpg',
                  '/home/tamarbo/datasets/car_damage/val/42.jpg']

nice_demo_img_path = {0:'/home/tamarbo/datasets/coco/images/train2017/000000000488.jpg',
                      1:'/home/tamarbo/datasets/coco/images/train2017/000000000540.jpg',
                      2:'/home/tamarbo/datasets/coco/images/train2017/000000000544.jpg',  # baseball game
                      3:'/home/tamarbo/datasets/coco/images/train2017/000000000625.jpg',  # freesbie girls
                      4:'/home/tamarbo/datasets/coco/images/train2017/000000000673.jpg'}  # surfboards

demo_img_path = nice_demo_img_path[3]

conv_layers = [0, 2, 61]
which_kernels = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

cfg = mmcv.Config.fromfile(cfg_file_path)
mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
im = mmcv.imread(demo_img_path)

class SaveFilters():
    def __init__(self,
                 work_dir,
                 cfg_path,
                 which_layers,
                 which_kernels,
                 results_dir_name='save_filters_results'):

        self.cfg_path = cfg_path
        self.which_layers = which_layers
        self.which_kernels = which_kernels
        self.results_dir_name = os.path.join(work_dir, results_dir_name)
        self.kernels = []
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        for k in self.which_kernels:
            self.kernels.append(module.weight[k, 2, :, :])
            self.outputs.append(module_out)

    def __clear__(self):
        self.kernels = []


# Before init of model
filters_before_load = SaveFilters(cfg.work_dir, cfg_file_path, conv_layers, which_kernels)
model = init_detector(cfg, device='cuda:0')
# hack to be able to .numpy() even layers which had requires_grad = True
for param in model.parameters():
    param.requires_grad = False
l_counter=0
for layer in model.modules():
    if isinstance(layer, torch.nn.modules.conv.Conv2d):
        if l_counter in conv_layers:
            print(layer)
            layer.register_forward_hook(filters_before_load)
        l_counter+=1
result = inference_detector(model, im)

# After init of model
model = init_detector(cfg, cfg.resume_from, device='cuda:0')
filters_after_load = SaveFilters(cfg.work_dir, cfg_file_path, conv_layers, which_kernels)
# hack to be able to .numpy() even layers which had requires_grad = True
for param in model.parameters():
    param.requires_grad = False

l_counter=0
for layer in model.modules():
    if isinstance(layer, torch.nn.modules.conv.Conv2d):
        if l_counter in conv_layers:
            layer.register_forward_hook(filters_after_load)
        l_counter+=1
inference_detector(model, im)


# plot org im
plt.figure()
plt.imshow(im, cmap=plt.cm.gray)
plt.savefig(os.path.join(cfg.work_dir, 'im_freesbie.jpg'))#+str(i)+'.jpg'))
plt.close()

#plot kernels
for f in [0,1]:
    fig1 = plt.figure(f)
    if f==0:
        title = 'Kernels: Before init model: of layers: [' + str(conv_layers[0]) + ', ' + str(conv_layers[1]) + ', ' + str(conv_layers[2]) + ']'
        which_hook = filters_before_load
    else:
        title = 'Kernels: After init model, of layers: [' + str(conv_layers[0]) + ', ' + str(conv_layers[1]) + ', ' + str(conv_layers[2]) + ']'
        which_hook = filters_after_load
    fig1.suptitle(title, fontsize=18)
    for sub_p in range(1, len(which_kernels)*len(conv_layers)+1):
        if sub_p<len(which_kernels)+1:
            this_layer = conv_layers[0]
        else:
            this_layer = conv_layers[1]
        this_kernel = which_kernels[(sub_p % len(which_kernels)) - 1]
        # plot kernels
        fig1.add_subplot(len(conv_layers), len(which_kernels), sub_p)
        im1 = which_hook.kernels[sub_p-1].cpu().numpy()
        plt.imshow(im1, cmap=plt.cm.gray) # cmap='viridis'
        plt.axis('off')
        plt.title('Kernel: ' + str(this_kernel), fontsize=5)
        if not os.path.exists(which_hook.results_dir_name): os.mkdir(which_hook.results_dir_name)
        plt.savefig(os.path.join(which_hook.results_dir_name,  title + '.jpg'))



#plot outputs
for f in [2,3]:
    fig2 = plt.figure(f)
    if f==2:
        title = 'Outputs: Before init model: of layers: [' + str(conv_layers[0]) + ', ' + str(conv_layers[1]) + ', ' + str(conv_layers[2]) + ']'
        which_hook = filters_before_load
    else:
        title = 'Outputs: After init model: of layers: [' + str(conv_layers[0]) + ', ' + str(conv_layers[1]) + ', ' + str(conv_layers[2]) + ']'
        which_hook = filters_after_load
    fig2.suptitle(title, fontsize=18)
    for sub_p in range(1, len(which_kernels)*len(conv_layers)+1):
        if sub_p<len(which_kernels)+1:
            this_layer = conv_layers[0]
        else:
            this_layer = conv_layers[1]
        this_kernel = which_kernels[(sub_p % len(which_kernels)) - 1]
        #plot outputs
        fig2.add_subplot(len(conv_layers), len(which_kernels), sub_p)
        im2 = which_hook.outputs[sub_p - 1][0, 5, :, :].cpu().numpy()
        #im2 = (im2-im2.min())/(im2.max()-im2.min())
        plt.imshow(im2, cmap=plt.cm.gray)  # cmap='viridis'
        plt.axis('off')
        plt.title('Kernel: ' + str(this_kernel), fontsize=5)
        if not os.path.exists(which_hook.results_dir_name): os.mkdir(which_hook.results_dir_name)
        plt.savefig(os.path.join(which_hook.results_dir_name, title + '.jpg'))
filters_before_load.__clear__()
filters_after_load.__clear__()