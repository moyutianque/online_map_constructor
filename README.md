# Semantic map constructor

This project is a distilled map constructor from object goal navigation paper.

The main modules of constructing a 2D/3D semantic map from simulation environment are designed

Main mathmatical calculation rely on pytorch


## Data preparation

```
data
├── scene_datasets
│   ├── mp3d
│   └── gibson
└─── vlnce
    ├── train
    │   ├── train.json.gz
    │   └── train_gt.json.gz
    ├── val_seen
    │   └── ...

```
scene_datasets -> ```/staging/leuven/stg_00095/simulator_scenes```

vlnce -> ```/data/leuven/335/vsc33595/dataset/vln_task/R2R_VLNCE_v1-2_preprocessed```


## Generation test


```bash
python run.py --vis_stepwise  # if want to have stepwise plot with agent arrow
python run.py  # if only want the overall map after explore
```