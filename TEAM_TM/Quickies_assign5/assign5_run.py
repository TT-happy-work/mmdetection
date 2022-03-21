import sys
import os
from tools.train import main as mmdet_train
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import torch
import mmcv
from mmcv import Config
import matplotlib.pyplot as plt



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device=="cpu":
    print("Please restart computer, no CUDA available device")

nice_demo_img_path = {0:'/home/tamarbo/datasets/coco/images/train2017/000000000488.jpg',
                      1:'/home/tamarbo/datasets/coco/images/train2017/000000000540.jpg',
                      2:'/home/tamarbo/datasets/coco/images/train2017/000000000544.jpg',  # baseball game
                      3:'/home/tamarbo/datasets/coco/images/train2017/000000000625.jpg',  # freesbie girls
                      4:'/home/tamarbo/datasets/coco/images/train2017/000000000673.jpg'}  # surfboards
demo_img_path = nice_demo_img_path[3]
sys.path.append(os.getcwd())

archs = {
            'faster'          : '/home/tamarbo/PycharmProjects/mmdetection/TEAM_TM/configs/config_faster_coco.py',
            'retinanet'       : '/home/tamarbo/PycharmProjects/mmdetection/TEAM_TM/configs/config_retinanet_coco.py',
            'yolo3_mobilenet' : '/home/tamarbo/PycharmProjects/mmdetection/TEAM_TM/configs/config_yolo3mobilenet_coco.py',
            'yolo3_darknet'   : '/home/tamarbo/PycharmProjects/mmdetection/TEAM_TM/configs/config_yolo3darknet_coco.py',
            # 'detr'            : '/home/tamarbo/PycharmProjects/mmdetection/TEAM_TM/configs/yolo3_carDamage_config.py'
        }

for arch in archs.keys():
    print(arch)
    cfg = Config.fromfile(archs[arch])
    sys.argv.append(cfg)
    model = init_detector(cfg, cfg.resume_from, device='cuda:0')
    # plt.imshow(plt.imread(demo_img_path))
    # plt.show()
    im = mmcv.imread(demo_img_path)
    result = inference_detector(model, im)
    show_result_pyplot(model, im, result, score_thr=0.2)      # , out_file=out)










