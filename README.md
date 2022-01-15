# ExtractingBuildingByMMSeg
This repository is for Competition for ML_data class. Based on mmsegmentatoin,mainly using swin transformer to completed the competition.

Two ways of training are provided here, non-distributed and distributed training. If you want to use distributed training, you can directly check the code in test.py

If you want to use distributed training on a linux server, you can do the following：
First make dist_train.sh available under linux
```latex
chmod 777 ./mmsegmentation/tools/dist_train.sh
vi ./mmsegmentation/tools/dist_train.sh
set ff=unix
```
Next, you can use the following commands for distributed training. We provide two kinds of configuration files, biet and swin, so you can configure the configuration files yourself as needed, and to support both BEiT and ConvNeXt networks, we modify and add to the mmsegmentation source code to make sure it can be used directly
```latex
nohup ./mmsegmentation/tools/dist_train.sh ./mine/myconfig_swin.py 4 > hehe.log 2>&1 &
```
You can also manually select the desired GPU
```latex
CUDA_VISIBLE_DEVICES=2,3 ./mmsegmentation/tools/dist_train.sh ./mine/myconfig_biet.py 2
```

