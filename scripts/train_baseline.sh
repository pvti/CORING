python train.py --arch VGG16 --output checkpoint/baseline/vgg/vgg16.pth --num_epochs 500 --wandb_name vgg16_train &&

python train.py --arch Resnet18 --output checkpoint/baseline/resnet/resnet18.pth --num_epochs 500 --wandb_name resnet18_train
