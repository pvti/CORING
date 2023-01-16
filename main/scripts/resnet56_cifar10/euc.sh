for cpr in [0.]+[0.18]*29 [0.]+[0.12]*2+[0.4]*27 [0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9
do
    python main/main.py --arch resnet_56 --job_dir result/hrankplus/resnet_56/standard --use_pretrain --pretrain_dir checkpoint/cifar/cifar10/resnet_56.pt --criterion Euclide_dis --strategy min_sum --compress_rate $cpr --epochs 300 --lr_decay_step '150,225' --batch_size 128
done
