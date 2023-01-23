#round 1
for shot in 5 10 15
do
    for cpr in [0.]+[0.18]*29 [0.]+[0.12]*2+[0.4]*27 [0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9
    do
    python main/main_kshot.py --arch resnet_56 --job_dir kshot/round1 --pretrain_dir checkpoint/cifar/cifar10/resnet_56.pt --criterion Euclide_dis --strategy min_sum --compress_rate $cpr --shot $shot --epochs 400 --lr_decay_step '150,225' --weight_decay 0.005 --batch_size 256
    done
done

#round 2
for shot in 5 10 15
do
    for cpr in [0.]+[0.18]*29 [0.]+[0.12]*2+[0.4]*27 [0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9
    do
        python main/main_kshot.py --arch resnet_56 --job_dir kshot/round2 --pretrain_dir checkpoint/cifar/cifar10/resnet_56.pt --criterion Euclide_dis --strategy min_sum --compress_rate $cpr --shot $shot --epochs 400 --lr_decay_step '150,225' --weight_decay 0.006 --batch_size 256
    done
done

#round 3
for shot in 5 10 15
do
    for cpr in [0.]+[0.18]*29 [0.]+[0.12]*2+[0.4]*27 [0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9
    do
        python main/main_kshot.py --arch resnet_56 --job_dir kshot/round3 --pretrain_dir checkpoint/cifar/cifar10/resnet_56.pt --criterion Euclide_dis --strategy min_sum --compress_rate $cpr --shot $shot --epochs 400 --lr_decay_step '150,225' --weight_decay 0.005 --batch_size 128
    done
done
