#round 1
#for cpr in [0.]+[0.2]*12+[0.]+[0.2]*12+[0.]+[0.2]*12 [0.]+[0.3]*12+[0.1]+[0.3]*12+[0.1]+[0.3]*12 [0.]+[0.4]*12+[0.]+[0.4]*12+[0.]+[0.4]*12
#do
#    python main/main.py --arch densenet_40 --job_dir round1 --use_pretrain --pretrain_dir checkpoint/cifar/cifar10/densenet_40.pt --criterion VBD_dis --strategy min_sum --compress_rate $cpr --epochs 300 --lr_decay_step '150,225' --weight_decay 0.002 --batch_size 256 --gpu 1
#done

#round 2
for cpr in [0.]+[0.2]*12+[0.]+[0.2]*12+[0.]+[0.2]*12 [0.]+[0.3]*12+[0.1]+[0.3]*12+[0.1]+[0.3]*12 [0.]+[0.4]*12+[0.]+[0.4]*12+[0.]+[0.4]*12
do
    python main/main.py --arch densenet_40 --job_dir round2 --use_pretrain --pretrain_dir checkpoint/cifar/cifar10/densenet_40.pt --criterion VBD_dis --strategy min_sum --compress_rate $cpr --epochs 300 --lr_decay_step '150,225' --weight_decay 0.003 --batch_size 256 --gpu 1
done

#round 3
for cpr in [0.]+[0.2]*12+[0.]+[0.2]*12+[0.]+[0.2]*12 [0.]+[0.3]*12+[0.1]+[0.3]*12+[0.1]+[0.3]*12 [0.]+[0.4]*12+[0.]+[0.4]*12+[0.]+[0.4]*12
do
    python main/main.py --arch densenet_40 --job_dir round3 --use_pretrain --pretrain_dir checkpoint/cifar/cifar10/densenet_40.pt --criterion VBD_dis --strategy min_sum --compress_rate $cpr --epochs 300 --lr_decay_step '150,225' --weight_decay 0.002 --batch_size 128 --gpu 1
done

#round 4
for cpr in [0.]+[0.08]*6+[0.09]*6+[0.08]*26
do
    python main/main.py --arch densenet_40 --job_dir round4 --use_pretrain --pretrain_dir checkpoint/cifar/cifar10/densenet_40.pt --criterion VBD_dis --strategy min_sum --compress_rate $cpr --epochs 400 --lr_decay_step '150,225' --weight_decay 0.002 --batch_size 256 --gpu 1
done
