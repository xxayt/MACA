import os
import yaml
from yacs.config import CfgNode as CN


_C = CN()

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 128
_C.DATA.DATA_PATH = ''
_C.DATA.DATASET = ''
_C.DATA.IMG_SIZE = 224

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.TYPE = 'ViT'
_C.MODEL.NAME = ''
_C.MODEL.RESUME = ''
_C.MODEL.NUM_CLASSES = 100
_C.MODEL.PRETRAINED = ''

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 300

# 使用文件更新
def _update_config_from_file(config, cfg_file):
    config.defrost()  # 解冻参数，可以修改
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()  # 冻结参数，不能更新


# 使用args参数更新
def update_config(config, args):
    _update_config_from_file(config, args.cfg)
    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)  # 可以直接从一个list中读取参数

    # merge from specific arguments
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if args.dataset is not None:
        config.DATA.DATASET = args.dataset

    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.pretrain is not None:
        config.MODEL.PRETRAINED = args.pretrain
    
    if args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    if args.output:
        config.OUTPUT = args.output

    if args.epochs is not None:
        config.TRAIN.EPOCHS = args.epochs
    
    
    # set local rank for distributed training
    config.LOCAL_RANK = args.local_rank
    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)
    config.freeze()  # 冻结参数，不能更新


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()  # 读取预定义的一些值
    update_config(config, args)  # 可以使用yaml文件中的值更新，以及直接使用args中的参数更新
    return config


################### For Inferencing ####################
def update_inference_config(config, args):
    _update_config_from_file(config, args.cfg)
    config.defrost()
    
    config.freeze()


def get_inference_config(cfg_path):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_inference_config(config, cfg_path)
    return config

################### For Inferencing ####################
