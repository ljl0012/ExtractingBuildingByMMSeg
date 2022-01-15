# Check Pytorch installation
import torch, torchvision
import mmseg
import mmcv

import torch
# import segmentation_models_pytorch as smp
import numpy as np
# import numba as nb
# import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
# from tqdm import tqdm
from PIL import Image
import cv2
import random
import os
import gc

from mmcv.parallel.distributed_deprecated import MMDistributedDataParallel
from torchvision import transforms as tfs
import os.path as osp
from scipy.io import loadmat
import csv
from PIL import Image
from torch.nn.parallel import DistributedDataParallel

from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmcv import Config

# DATAPATH = "../input/competition-for-ml-data-class/data/"


config_file = '/media/home/jianlong.li/ljl/mmseg2buildingExtract/mmsegmentation/configs/swin/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K.py'
# !mkdir ./checkpoints
# !wget https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K_20210531_125459-429057bf.pth -P checkpoints

# split train/val set randomly

# train_list = [osp.splitext(filename)[0] for filename in mmcv.scandir(
#     osp.join('/media/home/jianlong.li/ljl/data/Extractbuildingdata/data/train','label'), suffix='.tif')]
# val_list = [osp.splitext(filename)[0] for filename in mmcv.scandir(
#     osp.join('/media/home/jianlong.li/ljl/data/Extractbuildingdata/data/val','label'), suffix='.tif')]
# with open(osp.join('/media/home/jianlong.li/ljl/data/Extractbuildingdata/data/train/train.txt'), 'w') as f:
#   # select first 4/5 as train set
#   train_length = int(len(train_list))
#   f.writelines(line + '\n' for line in train_list)
# with open(osp.join( '/media/home/jianlong.li/ljl/data/Extractbuildingdata/data/val/val.txt'), 'w') as f:
#   # select last 1/5 as train set
#   f.writelines(line + '\n' for line in val_list)
#
# train_list = [osp.splitext(filename)[0] for filename in mmcv.scandir(
#     osp.join('/media/home/jianlong.li/ljl/UGS/UGSet2/train','label'), suffix='.tif')]
# val_list = [osp.splitext(filename)[0] for filename in mmcv.scandir(
#     osp.join('/media/home/jianlong.li/ljl/UGS/UGSet2/val','label'), suffix='.tif')]
# test_list = [osp.splitext(filename)[0] for filename in mmcv.scandir(
#     osp.join('/media/home/jianlong.li/ljl/UGS/UGSet2/test/test','label'), suffix='.tif')]
# with open(osp.join('/media/home/jianlong.li/ljl/UGS/UGSet2/train/train.txt'), 'w') as f:
#   # select first 4/5 as train set
#   train_length = int(len(train_list))
#   f.writelines(line +'.tif'+ '\n' for line in train_list)
# with open(osp.join( '/media/home/jianlong.li/ljl/UGS/UGSet2/val/val.txt'), 'w') as f:
#   # select last 1/5 as train set
#   f.writelines(line +'.tif'+ '\n' for line in val_list)
# with open(osp.join('/media/home/jianlong.li/ljl/UGS/UGSet2/test/test/test.txt'), 'w') as f:
#   # select last 1/5 as train set
#   f.writelines(line +'.tif'+ '\n' for line in val_list)

cfg = Config.fromfile(config_file)

classes = ('buildind', 'background')
palette = [[0, 0, 0], [255, 255, 255]]


# @DATASETS.register_module()
# class myDataset(CustomDataset):
#   CLASSES = classes
#   PALETTE = palette
#
#   def __init__(self, split, **kwargs):
#       super().__init__(img_suffix='.tif', seg_map_suffix='.tif',
#                        split=split, **kwargs)
#       assert osp.exists(self.img_dir) and self.split is not None

from mmseg.apis import set_random_seed

# Since we use ony one GPU, BN is used instead of SyncBN
# cfg.norm_cfg = dict(type='SyncLN', requires_grad=True)
# cfg.model.backbone.norm_cfg = cfg.norm_cfg
# cfg.model.decode_head.norm_cfg = cfg.norm_cfg
# cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
# modify num classes of the model in decode/auxiliary head
cfg.model.decode_head.num_classes = 2
cfg.model.auxiliary_head.num_classes = 2

# Modify dataset type and path
cfg.dataset_type = 'myDataset'
cfg.data_root = '/media/home/jianlong.li/ljl/data/Extractbuildingdata'

cfg.data.samples_per_gpu = 4
cfg.data.workers_per_gpu=2

cfg.img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
cfg.crop_size = (384, 384)
cfg.train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
#     dict(type='Resize', img_scale=(320, 240), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=cfg.crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **cfg.img_norm_cfg),
    dict(type='Pad', size=cfg.crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

cfg.test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(384, 384),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **cfg.img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


cfg.data.train.type = cfg.dataset_type
cfg.data.train.data_root =''
cfg.data.train.img_dir = '/media/home/jianlong.li/ljl/data/Extractbuildingdata/data/train/image'
cfg.data.train.ann_dir = '/media/home/jianlong.li/ljl/data/Extractbuildingdata/data/train/label'
cfg.data.train.pipeline = cfg.train_pipeline
cfg.data.train.split = '/media/home/jianlong.li/ljl/data/Extractbuildingdata/data/train/train.txt'

cfg.data.val.type = cfg.dataset_type
cfg.data.val.data_root = ''
cfg.data.val.img_dir = '/media/home/jianlong.li/ljl/data/Extractbuildingdata/data/val/image'
cfg.data.val.ann_dir = '/media/home/jianlong.li/ljl/data/Extractbuildingdata/data/val/label'
cfg.data.val.pipeline = cfg.test_pipeline
cfg.data.val.split = '/media/home/jianlong.li/ljl/data/Extractbuildingdata/data/val/val.txt'

cfg.data.test.type = cfg.dataset_type
cfg.data.test.data_root = ''
cfg.data.test.img_dir = '/media/home/jianlong.li/ljl/data/Extractbuildingdata/data/val/image'
cfg.data.test.ann_dir = '/media/home/jianlong.li/ljl/data/Extractbuildingdata/data/val/label'
cfg.data.test.pipeline = cfg.test_pipeline
cfg.data.test.split = '/media/home/jianlong.li/ljl/data/Extractbuildingdata/data/val/val.txt'

# We can still use the pre-trained Mask RCNN model though we do not need to
# use the mask branch
cfg.load_from =  '/media/home/jianlong.li/ljl/data/Extractbuildingdata/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K_20210531_125459-429057bf.pth'

# Set up working dir to save files and logs.
cfg.work_dir = '/media/home/jianlong.li/ljl/data/Extractbuildingdata/work/tutorial'

cfg.runner.max_iters = 10
cfg.log_config.interval = 10
cfg.evaluation.interval = 10
cfg.checkpoint_config.interval = 10

# Set seed to facitate reproducing the result
cfg.seed = 0
set_random_seed(0, deterministic=False)
# cfg.gpus = 2
cfg.gpu_ids = range(1)


# Let's have a look at the final config used for training
print(f'Config:\n{cfg.pretty_text}')

# Build the dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_segmentor(
    cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
# torch.distributed.init_process_group(backend="nccl")
# model = MMDistributedDataParallel(model.cuda())
# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES
model.PALETTE=datasets[0].PALETTE
# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_segmentor(model, datasets, cfg, distributed=False, validate=True,
                meta=dict(CLASSES=model.CLASSES,PALETTE=[[0, 0, 0], [255, 255, 255]]))