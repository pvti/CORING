#round 7
for shot in 1 5 10 15
do
    for cpr in [0.21]*7+[0.75]*5 [0.3]*7+[0.75]*5 [0.45]*7+[0.78]*5
    do
        python main/main_lr.py --arch vgg_16_bn --job_dir kshot/round7 --pretrain_dir checkpoint/cifar/cifar10/vgg_16_bn.pt --criterion cosine_sim --strategy min_sum --compress_rate $cpr --shot $shot --epochs 400 --lr_type cos --weight_decay 0.005 --batch_size 256
    done
done &

for shot in 1 5 10 15
do
    for cpr in [0.21]*7+[0.75]*5 [0.3]*7+[0.75]*5 [0.45]*7+[0.78]*5
    do
        python main/main_lr.py --arch vgg_16_bn --job_dir kshot/round7 --pretrain_dir checkpoint/cifar/cifar10/vgg_16_bn.pt --criterion Euclide_dis --strategy min_sum --compress_rate $cpr --shot $shot --epochs 400 --lr_type cos --weight_decay 0.005 --batch_size 256 --gpu 1
    done
done &

for shot in 1 5 10 15
do
    for cpr in [0.21]*7+[0.75]*5 [0.3]*7+[0.75]*5 [0.45]*7+[0.78]*5
    do
        python main/main_lr.py --arch vgg_16_bn --job_dir kshot/round7 --pretrain_dir checkpoint/cifar/cifar10/vgg_16_bn.pt --criterion VBD_dis --strategy min_sum --compress_rate $cpr --shot $shot --epochs 400 --lr_type cos --weight_decay 0.005 --batch_size 256 --gpu 2
    done
done
