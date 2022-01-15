# ExtractingBuildingByMMSeg
This repository is for Competition for ML_data class. Based on mmsegmentatoin,mainly using swin transformer to completed the competition.

```latex
chmod 777 ./mmsegmentation/tools/dist_train.sh
vi ./mmsegmentation/tools/dist_train.sh
set ff=unix
}
```

```latex
nohup ./mmsegmentation/tools/dist_train.sh ./mine/myconfig_swin.py 4 > hehe.log 2>&1 &
}
```

```latex
CUDA_VISIBLE_DEVICES=2,3 ./mmsegmentation/tools/dist_train.sh ./mine/myconfig_biet.py 2
}
```

