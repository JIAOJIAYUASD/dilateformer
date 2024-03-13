# DilateFormer

Official PyTorch implementation of IEEE Transaction on Multimedia 2023 paper “DilateFormer: Multi-Scale Dilated Transformer for Visual Recognition” .
[[paper]](https://arxiv.org/abs/2302.01791) 
[[Project Page]](https://isee-ai.cn/~jiaojiayu/DilteFormer.html)


We currenent release the pytorch version code for:

- [x] ImageNet-1K training
- [x] ImageNet-1K pre-trained weights

## ImageNet-1K pre-trained weights
Baidu Netdisk Link: [[ckpt]](https://pan.baidu.com/s/1DTKScF5G0Cbq-jaJrxeb4A?pwd=q4mu)
Extracted code：q4mu

Google drive Link: [[ckpt]](https://drive.google.com/drive/folders/1r8PDAQyccI6lKMIuaejin-AI1VW16Fvb?usp=sharing)

## Image classification

Our repository is built base on the [DeiT](https://github.com/facebookresearch/deit) repository, but we add some useful features:

1. Calculating accurate FLOPs and parameters with [fvcore](https://github.com/facebookresearch/fvcore) (see [check_model.py](check_model.py)).
2. Auto-resuming.
3. Saving best models and backup models.
4. Generating training curve (see [generate_tensorboard.py](generate_tensorboard.py)).

### Installation


- Install PyTorch 1.7.0+ and torchvision 0.8.1+

  ```shell
  conda install -c pytorch pytorch torchvision
  ```

- Install other packages

  ```shell
  pip install timm==0.5.4
  pip install fvcore
  ```

### Training

Simply run the training scripts as followed,  and take dilateformer_tiny as example:

```shell
bash dist_train.sh dilateformer_tiny [other prams]
```

If the training was interrupted abnormally, you can simply rerun the script for auto-resuming. Sometimes the checkpoint may not be saved properly, you should set the resumed model via `--reusme ${work_path}/ckpt/backup.pth`.



### Generate curves

You can generate the training curves as followed:

```shell
python3 generate_tensoboard.py
```

Note that you should install `tensorboardX`.

### Calculating FLOPs and Parameters

You can calculate the FLOPs and parameters via:

```shell
python3 check_model.py
```

## Acknowledgement

This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models) library and the [DeiT](https://github.com/facebookresearch/deit) repository.

## Citation
If you use this code for a paper, please cite:

DilateFormer
```
@article{jiao2023dilateformer,
title = {DilateFormer: Multi-Scale Dilated Transformer for Visual Recognition},
author = {Jiao, Jiayu and Tang, Yu-Ming and Lin, Kun-Yu and Gao, Yipeng and Ma, Jinhua and Wang, Yaowei and Zheng, Wei-Shi},
journal = {{IEEE} Transaction on Multimedia},
year = {2023}
}
```
