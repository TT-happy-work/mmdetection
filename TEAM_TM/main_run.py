import sys
import os
import os.path as osp
from tools.train import main as mmdet_train
from mmdet.apis import inference_detector, show_result_pyplot
import torch
import mmcv
from mmcv import Config




assert torch.cuda.is_available(), 'Restart Computer, Cuda is not available'


arch = 'faster'   # 'faster' /  'yolo'
if arch=='faster':
    cfg_file_path = '/home/tamarbo/PycharmProjects/mmdetection/TEAM_TM/configs/faster_carDamage_config.py'
elif arch=='yolo':
    cfg_file_path = '/home/tamarbo/PycharmProjects/mmdetection/TEAM_TM/configs/yolo3_carDamage_config.py'
elif arch=='detr':
    cfg_file_path = '/home/tamarbo/PycharmProjects/mmdetection/TEAM_TM/configs/yolo3_carDamage_config.py'

cfg = Config.fromfile(cfg_file_path)
sys.path.append(os.getcwd())
sys.argv.append(cfg_file_path)

##Train
mmdet_train()


##Short Inference
# imgs = cfg.example_images
#
#
# for ima in imgs:
#     im = mmcv.imread(ima)
#     result = inference_detector(model, im)
#     print('after inference_detector Is cuda available:', torch.cuda.is_available())
#     out = osp.join(cfg.work_dir, 'imgs', osp.basename(ima))
#     try:
#         show_result_pyplot(model, im, result, score_thr=0.2)# , out_file=out)
#         pass
#     except:
#         pass

##Test








