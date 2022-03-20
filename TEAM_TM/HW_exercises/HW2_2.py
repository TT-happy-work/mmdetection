import os
import torch
import matplotlib.pyplot as plt
import mmcv
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import cv2
import numpy as np


def show_objectnesses(img, featuremaps):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # mmcv turns img into bgr

    for scale in range(0,5):
        fig, axes = plt.subplots(2, 2, figsize= (16,9))
        axes[0,0].imshow(img)
        axes[0,0].set_title('Original image')

        axes[0,1].imshow(img, alpha=0.9)
        featuremap = featuremaps[scale][0].cpu().numpy()
        axes[0,1].imshow(cv2.resize(featuremap, (640,425)), alpha=0.5)
        axes[0,1].set_title('Scale: '+str(scale) + ', Anchor: 0')

        axes[1, 0].imshow(img, alpha=0.9)
        featuremap = featuremaps[scale][1].cpu().numpy()
        axes[1, 0].imshow(cv2.resize(featuremap, (640, 425)), alpha=0.5)
        axes[1, 0].set_title('Scale: ' + str(scale) + ', Anchor: 1')

        axes[1,1].imshow(img, alpha=0.9)
        featuremap = featuremaps[scale][2].cpu().numpy()
        axes[1,1].imshow(cv2.resize(featuremap, (640,425)), alpha=0.5)
        axes[1,1].set_title('Scale: '+str(scale) + ', Anchor: 2')

        plt.savefig(os.path.join(cfg.work_dir,'feature_results_of_scale_' + str(scale)+'.png'), bbox_inches='tight')


class GetLayerObjectness():
    def __init__(self,
                 work_dir,
                 cfg_path,
                 results_dir_name='save_objectness_results'):

        self.cfg_path = cfg_path
        self.which_fpn_scale = 0
        self.results_dir_name = os.path.join(work_dir, results_dir_name)
        self.objectnesses = dict()


    def __call__(self, module, module_in, module_out):
        self.objectnesses[self.which_fpn_scale] = []
        for anchor in range(0,3):
            # print('anchor: ', anchor, self.which_fpn_scale)
            self.objectnesses[self.which_fpn_scale].append(module_out[0, anchor, :, :])
        self.which_fpn_scale += 1

    def __clear__(self):
        self.objectnesses = dict()


cfg_file_path = 'HW2_config.py'
#which_kernels = list(range(0, 255, 50))
which_kernels = list(range(0, 255, 35))
nice_demo_img_path = {0:'/home/tamarbo/datasets/coco/images/train2017/000000000488.jpg',  # baseball players
                      1:'/home/tamarbo/datasets/coco/images/train2017/000000000540.jpg',  # airplane
                      2:'/home/tamarbo/datasets/coco/images/train2017/000000000544.jpg',  # baseball game
                      3:'/home/tamarbo/datasets/coco/images/train2017/000000000625.jpg',  # freesbie girls
                      4:'/home/tamarbo/datasets/coco/images/train2017/000000000673.jpg'}  # surfboards

demo_img_path = nice_demo_img_path[3]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available(): print('Restart Computer, Cuda is not available')
cfg = mmcv.Config.fromfile(cfg_file_path)
mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))

#Show original image
plt.figure('Org img')
plt.imshow(plt.imread(demo_img_path))
plt.savefig(os.path.join(cfg.work_dir, 'org_img.jpg'))

model = init_detector(cfg, cfg.resume_from, device='cuda:0')
objectness = GetLayerObjectness(cfg.work_dir, cfg_file_path, results_dir_name='save_objectness_results')

# Define which layer gets the hook
layer_4_objectness = model.rpn_head.rpn_cls
layer_4_objectness.register_forward_hook(objectness)

# Perform the inference+hook
im = mmcv.imread(demo_img_path)
result = inference_detector(model, demo_img_path)
# Show org image with its detections
out = os.path.join(cfg.work_dir, 'imgs', os.path.basename(cfg_file_path))
show_result_pyplot(model, im, result, score_thr=0.1)
# Show HEATMAPS
heatmaps = objectness.objectnesses
show_objectnesses(im, heatmaps)
objectness.__clear__()