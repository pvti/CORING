#round 5
for shot in 1
do
    for cpr in [0.]+[0.05]+[0.1]*2+[0.15]*2+[0.2]*2 [0.]+[0.1]+[0.25]*2+[0.25]*2+[0.3]*2 [0.]+[0.2]+[0.3]*2+[0.4]*2+[0.5]*2
    do
    python main/main_lr.py --arch mobilenet_v2 --job_dir mobilenetv2CIFAR/kshot/round5 --pretrain_dir checkpoint/cifar/cifar10/mobilenet_v2.pt --criterion VBD_dis --strategy min_sum --compress_rate $cpr --shot $shot --epochs 400 --weight_decay 0.005 --batch_size 256
    done
done &

#round 6
for shot in 1
do
    for cpr in [0.]+[0.05]+[0.1]*2+[0.15]*2+[0.2]*2 [0.]+[0.1]+[0.25]*2+[0.25]*2+[0.3]*2 [0.]+[0.2]+[0.3]*2+[0.4]*2+[0.5]*2
    do
    python main/main_lr.py --arch mobilenet_v2 --job_dir mobilenetv2CIFAR/kshot/round6 --pretrain_dir checkpoint/cifar/cifar10/mobilenet_v2.pt --criterion VBD_dis --strategy min_sum --compress_rate $cpr --shot $shot --epochs 400 --weight_decay 0.001 --batch_size 256 --gpu 1
    done
done
