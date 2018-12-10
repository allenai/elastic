# Elastic
This repo contains a PyTorch implementation of Elastic. It is compatible with PyTorch 1.0-stable, PyTorch 1.0-preview and PyTorch 0.4.1. All released models are exactly the models evaluated in the paper.
## ImageNet
We prepare our data following https://github.com/pytorch/examples/tree/master/imagenet

Pretrained models available at 
```
for a in resnext50 resnext50_elastic resnext101 resnext101_elastic dla60x dla60x_elastic dla102x se_resnext50_elastic densenet201 densenet201_elastic; do
   wget http://ai2-vision.s3.amazonaws.com/elastic/imagenet_models/"$a".pth.tar
done
```
### Testing
```
python classify.py /path/to/imagenet/ --evaluate --resume /path/to/model.pth.tar
```
### Training
```
python classify.py /path/to/imagenet/
```
### Multi-processing distributed training in Docker (recommended):
We train all the models in docker containers: https://docs.nvidia.com/deeplearning/dgx/pytorch-release-notes/rel_18.07.html

You may need to follow instructions in the link above to install [docker](https://www.docker.com/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) if you haven't done so.

After pulling the docker image, we run a docker container:
```
nvidia-docker run -it -e NVIDIA_VISIBLE_DEVICES=0,1 --ipc=host --rm -v /path/to/code:/path/to/code -v /path/to/imagenet:/path/to/imagenet nvcr.io/nvidia/pytorch:18.07-py3
```
Then run this training script inside the docker container.
```
python -m apex.parallel.multiproc docker_classify.py /path/to/imagenet
```
## MSCOCO
We extract data into this structure and use python cocoapi to load data: https://github.com/cocodataset/cocoapi
```
/path/to/mscoco/annotations/instances_train2014.json
/path/to/mscoco/annotations/instances_val2014.json
/path/to/mscoco/train2014
/path/to/mscoco/val2014
```
Pretrained models available at 
```
for a in resnext50 resnext50_elastic resnext101 resnext101_elastic dla60x dla60x_elastic densenet201 densenet201_elastic; do
   wget http://ai2-vision.s3.amazonaws.com/elastic/coco_models/coco_"$a".pth.tar
done
```
### Testing
```
python multilabel_classify.py /path/to/mscoco --resume /path/to/model.pth.tar --evaluate
```
### Finetuning or resume training
```
python multilabel_classify.py /path/to/mscoco --resume /path/to/model.pth.tar
```
## PASCAL VOC semantic segmentation
We prepare PASCAL VOC data following https://github.com/chenxi116/DeepLabv3.pytorch

Pretrained models available at
```
for a in resnext50 resnext50_elastic resnext101 resnext101_elastic dla60x dla60x_elastic; do
   wget http://ai2-vision.s3.amazonaws.com/elastic/pascal_models/deeplab_"$a"_pascal_v3_original_epoch50.pth
done
```
### Testing
Models should be put at data/deeplab_*.pth
```
CUDA_VISIBLE_DEVICES=0 python segment.py --exp original
```
### Finetuning or resume training
All PASCAL VOC semantic segmentation models are trained on one GPU.
```
CUDA_VISIBLE_DEVICES=0 python segment.py --exp my_exp --train --resume /path/to/model.pth.tar
```
## Note
Distributed training maintains batchnorm statistics on each GPU/worker/process without synchronization, which leads to different performances on different GPUs. At the end of each epoch, our distributed script reports averaged performance (top-1, top-5) by evaluating the whole validation set on all GPUs, and saves the model on the first GPU (throws away models on other GPUs). As a result, evaluating the saved model after training leads to slightly (<0.1%) different (could be either better or worse) numbers. In the paper, we reported the average performances for all models. Averaging batchnorm statistics before evaluation may lead to marginally better numbers.

## Credits
ImageNet training script is modified from https://github.com/pytorch/pytorch

ImageNet distributed training script is modified from https://github.com/NVIDIA/apex

Pascal segmentation code is modified from https://github.com/chenxi116/DeepLabv3.pytorch

ResNext model is modified form https://github.com/last-one/tools

DLA models are modified from https://github.com/ucbdrive/dla

DenseNet model is modified from https://github.com/csrhddlam/pytorch-checkpoint

