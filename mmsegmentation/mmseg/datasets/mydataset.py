import os.path as osp

import mmcv
import numpy as np
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset

classes = ('buildind', 'background')
palette = [[0, 0, 0], [255, 255, 255]]
@DATASETS.register_module()
class myDataset(CustomDataset):
  CLASSES = classes
  PALETTE = palette

  def __init__(self, split, **kwargs):
      super().__init__(img_suffix='.tif', seg_map_suffix='.tif',
                       split=split, **kwargs)
      assert osp.exists(self.img_dir) and self.split is not None