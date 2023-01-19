#round 1
for cpr in [0.]+[0.2]*12+[0.]+[0.2]*12+[0.]+[0.2]*12 [0.]+[0.3]*12+[0.1]+[0.3]*12+[0.1]+[0.3]*12 [0.]+[0.4]*12+[0.]+[0.4]*12+[0.]+[0.4]*12
do
    python main/main.py --arch densenet_40 --job_dir round1 --use_pretrain --pretrain_dir checkpoint/cifar/cifar10/densenet_40.pt --criterion Euclide_dis --strategy min_sum --compress_rate $cpr --epochs 300 --lr_decay_step '150,225' --weight_decay 0.002 --batch_size 256
done
&&

#round 2
for cpr in [0.]+[0.2]*12+[0.]+[0.2]*12+[0.]+[0.2]*12 [0.]+[0.3]*12+[0.1]+[0.3]*12+[0.1]+[0.3]*12 [0.]+[0.4]*12+[0.]+[0.4]*12+[0.]+[0.4]*12
do
    python main/main.py --arch densenet_40 --job_dir round2 --use_pretrain --pretrain_dir checkpoint/cifar/cifar10/densenet_40.pt --criterion Euclide_dis --strategy min_sum --compress_rate $cpr --epochs 300 --lr_decay_step '150,225' --weight_decay 0.003 --batch_size 256
done
&&

#round 3
for cpr in [0.]+[0.2]*12+[0.]+[0.2]*12+[0.]+[0.2]*12 [0.]+[0.3]*12+[0.1]+[0.3]*12+[0.1]+[0.3]*12 [0.]+[0.4]*12+[0.]+[0.4]*12+[0.]+[0.4]*12
do
    python main/main.py --arch densenet_40 --job_dir round3 --use_pretrain --pretrain_dir checkpoint/cifar/cifar10/densenet_40.pt --criterion Euclide_dis --strategy min_sum --compress_rate $cpr --epochs 300 --lr_decay_step '150,225' --weight_decay 0.002 --batch_size 128
done
