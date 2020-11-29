### Paper - [**Domain Adaptive Knowledge Distillation for Driving Scene Semantic Segmentation**](https://arxiv.org/abs/2011.08007)

Please cite our paper if you find it useful.

```
@article{kothandaraman2020unsupervised,
  title={Unsupervised Domain Adaptive Knowledge Distillation for Semantic Segmentation},
  author={Kothandaraman, Divya and Nambiar, Athira and Mittal, Anurag},
  journal={arXiv preprint arXiv:2011.08007},
  year={2020}
}
```

Table of Contents
=================

  * [Paper - <a href="https://arxiv.org/abs/2011.08007" rel="nofollow"><strong>Domain Adaptive Knowledge Distillation for Driving Scene Semantic Segmentation</strong></a>](#paper---Domain-Adaptive-Knowledge-Distillation-for-Driving-Scene-Semantic-Segmentation)
  * [**Repo Details and Contents**](#repo-details-and-contents)
     * [Code structure](#code-structure)
     * [Datasets](#datasets)
     * [Dependencies](#dependencies)
  * [**Our network**](#our-network)
  * [**Acknowledgements**](#acknowledgements)

## Repo Details and Contents
Python version: 3.7

### Code structure
'dataset' folder - Contains dataloaders, list of train and validation images <br>
'model' folder - Contains code for the network architectures <br>
'utils' folder - Additional functions <br>
eval_cs.py - Evaluation script for cityscapes <br>
train_gta2cs_multi_drnd38, train_gta2cs_multi_drnd22 - Training script for teacher and undistilled student networks <br>
train_gta2cs_ts_multi.py - Training script for our domain adaptive knowledge distillation network

### Datasets
* [**CityScapes**](https://www.cityscapes-dataset.com/) 
* [**Berkeley Deep Drive**](https://bdd-data.berkeley.edu/) 
* [**GTA5**](https://download.visinf.tu-darmstadt.de/data/from_games/) 

### Dependencies
pytorch <br>
numpy <br>
scipy <br>
matplotlib <br>

## Our network

<p align="center">
<img src="figures/network.png">
</p>

## Acknowledgements

This code is heavily borrowed from [**AdaptSegNet**](https://github.com/wasidennis/AdaptSegNet)
