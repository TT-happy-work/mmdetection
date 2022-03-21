import os
import torch
import matplotlib.pyplot as plt
import mmcv
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import cv2
import numpy as np
from tools.train import main as mmdet_train
from tools.test import main as mmdet_test


cfg_file_path = 'medium1_config.py'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available(): print('Restart Computer, Cuda is not available')
cfg = mmcv.Config.fromfile(cfg_file_path)
mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))

##Train
mmdet_train()

# ##Test
# mmdet_test()
#
# model =
# # Perform the inference
# nice_demo_img_path = {0:'/home/tamarbo/datasets/coco/images/train2017/000000000488.jpg',  # baseball players
#                       1:'/home/tamarbo/datasets/coco/images/train2017/000000000540.jpg',  # airplane
#                       2:'/home/tamarbo/datasets/coco/images/train2017/000000000544.jpg',  # baseball game
#                       3:'/home/tamarbo/datasets/coco/images/train2017/000000000625.jpg',  # freesbie girls
#                       4:'/home/tamarbo/datasets/coco/images/train2017/000000000673.jpg'}  # surfboards
#
# demo_img_path = nice_demo_img_path[3]
# im = mmcv.imread(demo_img_path)
# result = inference_detector(model, demo_img_path)
#
# # Show org image with its detections
# out = os.path.join(cfg.work_dir, 'imgs', os.path.basename(cfg_file_path))
# show_result_pyplot(model, im, result, score_thr=0.1)
