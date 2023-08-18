# Downstream tasks training scripts

This folder contains reference training scripts for Faster/Mask/Keypoint-RCNN-ResNet50-FPN for object detection, segmentation and keypoint detection.

To execute the example commands below you must install the following:

```
cython
pycocotools
matplotlib
```

You must modify the following flags:

`--data-path=/path/to/coco/dataset`

`--nproc_per_node=<number_of_gpus_available>`

Weights backbone is adopted from the classification task, therefore it should be the compressed resnet50 with coressponding pruning ratio.

As [recommended](https://github.com/pytorch/vision/blob/87d54c4e583207e7b003d6b59f1e7f49167f68f1/references/detection/train.py#L85) by `torchvision`, default learning rate and batch size values go along with 8xV100. Please modify them to match with your numbers of gpus, *e.g.,* `--nproc_per_node=1 --lr 0.02 -b 2`.

### Faster R-CNN ResNet-50 FPN
```
torchrun --nproc_per_node=8 train.py --dataset coco --model fasterrcnn_resnet50_fpn --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3 --weights-backbone resnet50.pt -cpr [0.]+[0.1]*3+[0.35]*16
```

### Mask R-CNN
```
torchrun --nproc_per_node=8 train.py --dataset coco --model maskrcnn_resnet50_fpn --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3 --weights-backbone resnet50.pt -cpr [0.]+[0.1]*3+[0.35]*16
```


### Keypoint R-CNN
```
torchrun --nproc_per_node=8 train.py --dataset coco_kp --model keypointrcnn_resnet50_fpn --epochs 46 --lr-steps 36 43 --aspect-ratio-group-factor 3 --weights-backbone resnet50.pt -cpr [0.]+[0.1]*3+[0.35]*16
```
