import torch, torchvision
import mmseg
import mmcv

import torch
import numpy as np
import numba as nb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm
from PIL import Image
import cv2
import random
import os
import gc
from torchvision import transforms as tfs
import os.path as osp
from scipy.io import loadmat
import csv
from PIL import Image

from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

from mmseg.apis import set_random_seed
from mmcv import Config

from mmcv.runner import load_checkpoint
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette



config_file = 'myconfig_swin.py'
cfg = Config.fromfile(config_file)
# print(checkpoint['meta']['CLASSES'])
testImageList = [i.path for i in os.scandir('/media/home/jianlong.li/ljl/data/Extractbuildingdata/data/val/image')]
testLabelList = [i.path for i in os.scandir('/media/home/jianlong.li/ljl/data/Extractbuildingdata/data/val/label')]

colors = ['white', 'blue']
cmap = mpl.colors.ListedColormap(colors)

# model = init_segmentor(cfg,  '/media/home/jianlong.li/ljl/data/Extractbuildingdata/work/tutorial/iter_90000.pth', device='cuda:0')
# # print(checkpoint['meta']['PALETTE'])
# with torch.no_grad():
#     for i in range(0, len(testImageList), 25):
#         img2 = cv2.imread(testImageList[i], -1)
#         img = mmcv.imread(testImageList[i])
#         lb = cv2.imread(testLabelList[i])[:, :, 0] / 255
#         lb = lb.astype(int)
#
#         result = inference_segmentor(model, testImageList[i])
#         result = torch.tensor(np.array(result))
#         result.squeeze_()
#         result = result.numpy()
#
#         plt.figure(figsize=(10, 30))
#         plt.subplot(1, 3, 1)
#         plt.imshow(img)
#         plt.subplot(1, 3, 2)
#         plt.imshow(lb)
#         plt.subplot(1, 3, 3)
#         plt.imshow(result)
#         plt.show()
#         if i > 1000:
#             break


# 计算dice系数
def dice_coef(pred, label):
    assert (pred.shape == label.shape)
    batchSize = pred.shape[0]
    pred, label = pred.view(batchSize, -1), label.view(batchSize, -1)
    TP = torch.sum(pred * label, dim=1).float()
    return 2 * TP / (torch.sum(pred, dim=1).float() + torch.sum(label, dim=1).float() + 0.00000001)


# 编码
def encodePixel(binaryMap):
    """
    把一张预测结果编码成平台要求的上传格式
    """
    # 输入必须为[h, w]型状的二值预测结果
    assert len(binaryMap.shape) == 2
    binaryMap = binaryMap.reshape(-1)
    totalPixNum = binaryMap.shape[0]
    encodedStr = ""
    flag = 0
    count = 0
    for i in range(totalPixNum):
        if (binaryMap[i] == 1) and (flag == 0) and (i < totalPixNum - 1):
            encodedStr += str(i + 1)
            encodedStr += " "
            flag = 1
            count += 1
        elif (binaryMap[i] == 0) and (flag == 1):
            encodedStr += str(count)
            encodedStr += " "
            count = 0
            flag = 0
        elif (binaryMap[i] == 1) and (flag == 1) and (i < totalPixNum - 1):
            count += 1
        elif (binaryMap[i] == 1) and (flag == 1) and (i == totalPixNum - 1):
            encodedStr += str(count)
            encodedStr += " "
            count = 0
            flag = 0
        elif (binaryMap[i] == 1) and (flag == 0) and (i == totalPixNum - 1):
            encodedStr += str(i + 1)
            encodedStr += " 1 "

    return encodedStr[:-1]

def inference(model,save_path, device=torch.device('cpu:0')):
    """
    使用model预测data_loader中的数据
    """

    data_path="/media/home/jianlong.li/ljl/data/Extractbuildingdata/data/test/image"

    imgList = [i.path for i in os.scandir(data_path) if i.name.split('.')[-1]]
    imgList.sort()
    imgNameList = [i.split('/')[-1] for i in imgList]
    fnList, encodedPixelList = ["ID"], ["Prediction"]

    for i in tqdm(range(len(imgNameList ))):
        result=inference_segmentor(model,  data_path+'/'+imgNameList[i])
        result = torch.tensor(np.array(result))
        result.squeeze_()
        result = result.cpu().numpy()
        fnList.append(imgNameList[i])
        encodedPixelList.append(encodePixel(result))

    pred2submit = np.array(list(zip(fnList, encodedPixelList)))
    np.savetxt(save_path, pred2submit, delimiter=",", fmt="%s")
    print("prediction saved to %s" % save_path)


model = init_segmentor(cfg, '/media/home/jianlong.li/ljl/data/Extractbuildingdata/work/tutorial/iter_70000.pth', device='cuda:1')
inference(model,  save_path="/media/home/jianlong.li/ljl/data/Extractbuildingdata/work/tutorial/prediction70000.csv", device=torch.device('cuda:1'))
model = init_segmentor(cfg, '/media/home/jianlong.li/ljl/data/Extractbuildingdata/work/tutorial/iter_80000.pth', device='cuda:1')
inference(model,  save_path="/media/home/jianlong.li/ljl/data/Extractbuildingdata/work/tutorial/prediction80000.csv", device=torch.device('cuda:1'))
model = init_segmentor(cfg, '/media/home/jianlong.li/ljl/data/Extractbuildingdata/work/tutorial/iter_90000.pth', device='cuda:1')
inference(model,  save_path="/media/home/jianlong.li/ljl/data/Extractbuildingdata/work/tutorial/prediction90000.csv", device=torch.device('cuda:1'))