#round 1
for cpr in [0.21]*7+[0.75]*5 [0.3]*7+[0.75]*5 [0.45]*7+[0.78]*5
do
    python main/main_kshot.py --arch vgg_16_bn --job_dir kshot/round1 --use_pretrain --pretrain_dir checkpoint/cifar/cifar10/vgg_16_bn.pt --criterion VBD_dis --strategy min_sum --compress_rate $cpr --epochs 400 --weight_decay 0.005 --batch_size 256 --gpu 1
done

#round 2
for cpr in [0.21]*7+[0.75]*5 [0.3]*7+[0.75]*5 [0.45]*7+[0.78]*5
do
    python main/main_kshot.py --arch vgg_16_bn --job_dir kshot/round2 --use_pretrain --pretrain_dir checkpoint/cifar/cifar10/vgg_16_bn.pt --criterion VBD_dis --strategy min_sum --compress_rate $cpr --epochs 400 --weight_decay 0.006 --batch_size 256 --gpu 1
done

#round 3
for cpr in [0.21]*7+[0.75]*5 [0.3]*7+[0.75]*5 [0.45]*7+[0.78]*5
do
    python main/main_kshot.py --arch vgg_16_bn --job_dir kshot/round3 --use_pretrain --pretrain_dir checkpoint/cifar/cifar10/vgg_16_bn.pt --criterion VBD_dis --strategy min_sum --compress_rate $cpr --epochs 400 --weight_decay 0.005 --batch_size 128 --gpu 1
done
