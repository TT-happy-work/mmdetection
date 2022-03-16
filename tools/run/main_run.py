import os.path as osp
import mmcv
from mmcv import Config
import copy
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector, set_random_seed
import torch
import time
from mmdet.utils import collect_env, get_root_logger#, get_git_hash
from mmdet import __version__
import os


## Organization
print('Is cuda available:', torch.cuda.is_available())
#cfg_file = 'configs/kiti_config.py'
cfg_file = '/home/tamarbo/PycharmProjects/mmdetection/tamarbo/configs/car_damage_config.py'
cfg = Config.fromfile(cfg_file)

# create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

# dump config
cfg.dump(os.path.join(cfg.work_dir, cfg_file.split('/')[-1]))
# init the logger before other steps
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

if cfg.resume_from is not None:
    cfg.resume_from = cfg.resume_from
if cfg.gpu_ids is not None:
    cfg.gpu_ids = cfg.gpu_ids
else:
    cfg.gpu_ids = range(1) if cfg.gpus is None else range(cfg.gpus)

# init the meta dict to record some important information such as
# environment info and seed, which will be logged
meta = dict()
# log env info
env_info_dict = collect_env()
print('after env_info_dict Is cuda available:', torch.cuda.is_available())

env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
dash_line = '-' * 60 + '\n'
logger.info('Environment info:\n' + dash_line + env_info + '\n' +
            dash_line)
meta['env_info'] = env_info
meta['config'] = cfg.pretty_text
# log some basic info
logger.info(f'Config:\n{cfg.pretty_text}')

# set random seeds
if cfg.seed is not None:
    logger.info(f'Set random seed to {cfg.seed}, '
                f'deterministic: {cfg.deterministic}')
    set_random_seed(cfg.seed, deterministic=cfg.deterministic)
cfg.seed = cfg.seed
meta['seed'] = cfg.seed
meta['exp_name'] = osp.basename(cfg_file)

##Model and Datasets
model = build_detector(
    cfg.model,
    train_cfg=cfg.get('train_cfg'),
    test_cfg=cfg.get('test_cfg'))

# Call ShowFiltersHook

model.init_weights()

# Call ShowFiltersHook

datasets = [build_dataset(cfg.data.train)]

##Pipeline
if len(cfg.workflow) == 2:
    val_dataset = copy.deepcopy(cfg.data.val)
    val_dataset.pipeline = cfg.data.train.pipeline
    datasets.append(build_dataset(val_dataset))
if cfg.checkpoint_config is not None:
    # save mmdet version, config file content and class names in
    # checkpoints as meta data
    cfg.checkpoint_config.meta = dict(
        mmdet_version=__version__ ,#+ get_git_hash()[:7],
        CLASSES=datasets[0].CLASSES)
# add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

#Handle Hooks
if 'custom_hooks' in cfg:
    for custom_hook in cfg['custom_hooks']:



##Train
train_detector(
    model,
    datasets,
    cfg,
    validate=(not cfg.no_validate),
    timestamp=timestamp,
    meta=meta)

##Short Inference
imgs = cfg.example_images
model.cfg = cfg

for ima in imgs:
    im = mmcv.imread(ima)
    result = inference_detector(model, im)
    print('after inference_detector Is cuda available:', torch.cuda.is_available())
    out = osp.join(cfg.work_dir, 'imgs', osp.basename(ima))
    try:
        show_result_pyplot(model, im, result, score_thr=0.2)# , out_file=out)
        pass
    except:
        pass

##Test








