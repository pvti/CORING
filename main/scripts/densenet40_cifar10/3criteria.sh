#round 6
for cpr in [0.]+[0.17]*5+[0.16]*7+[0.18]*1+[0.17]*25 [0.]+[0.25]*5+[0.28]*8+[0.32]*15+[0.34]*10 [0.]+[0.08]*6+[0.09]*6+[0.08]*26
do
    python main/main_lr.py --shot 1 --calib 0 --arch densenet_40 --job_dir round6 --pretrain_dir checkpoint/cifar/cifar10/densenet_40.pt --criterion cosine_sim --lr_type cos --strategy min_sum --compress_rate $cpr --epochs 400 --lr_decay_step '150,250' --weight_decay 0.005 --batch_size 256
done &
for cpr in [0.]+[0.2]*12+[0.]+[0.2]*12+[0.]+[0.2]*12 [0.]+[0.4]*12+[0.]+[0.4]*12+[0.]+[0.4]*12 [0.]+[0.3]*12+[0.1]+[0.3]*12+[0.1]+[0.3]*12
do
    python main/main_lr.py --shot 1 --calib 0 --arch densenet_40 --job_dir round6 --pretrain_dir checkpoint/cifar/cifar10/densenet_40.pt --criterion cosine_sim --lr_type cos --strategy min_sum --compress_rate $cpr --epochs 400 --lr_decay_step '150,250' --weight_decay 0.005 --batch_size 256
done &

for cpr in [0.]+[0.17]*5+[0.16]*7+[0.18]*1+[0.17]*25 [0.]+[0.25]*5+[0.28]*8+[0.32]*15+[0.34]*10 [0.]+[0.08]*6+[0.09]*6+[0.08]*26
do
    python main/main_lr.py --shot 1 --calib 0 --arch densenet_40 --job_dir round6 --pretrain_dir checkpoint/cifar/cifar10/densenet_40.pt --criterion Euclide_dis --lr_type cos --strategy min_sum --compress_rate $cpr --epochs 400 --lr_decay_step '150,250' --weight_decay 0.005 --batch_size 256 --gpu 2
done &
for cpr in [0.]+[0.2]*12+[0.]+[0.2]*12+[0.]+[0.2]*12 [0.]+[0.4]*12+[0.]+[0.4]*12+[0.]+[0.4]*12 [0.]+[0.3]*12+[0.1]+[0.3]*12+[0.1]+[0.3]*12
do
    python main/main_lr.py --shot 1 --calib 0 --arch densenet_40 --job_dir round6 --pretrain_dir checkpoint/cifar/cifar10/densenet_40.pt --criterion Euclide_dis --lr_type cos --strategy min_sum --compress_rate $cpr --epochs 400 --lr_decay_step '150,250' --weight_decay 0.005 --batch_size 256 --gpu 2
done &

for cpr in [0.]+[0.17]*5+[0.16]*7+[0.18]*1+[0.17]*25 [0.]+[0.25]*5+[0.28]*8+[0.32]*15+[0.34]*10 [0.]+[0.08]*6+[0.09]*6+[0.08]*26
do
    python main/main_lr.py --shot 1 --calib 0 --arch densenet_40 --job_dir round6 --pretrain_dir checkpoint/cifar/cifar10/densenet_40.pt --criterion VBD_dis --lr_type cos --strategy min_sum --compress_rate $cpr --epochs 400 --lr_decay_step '150,250' --weight_decay 0.005 --batch_size 256 --gpu 1
done &
for cpr in [0.]+[0.2]*12+[0.]+[0.2]*12+[0.]+[0.2]*12 [0.]+[0.4]*12+[0.]+[0.4]*12+[0.]+[0.4]*12 [0.]+[0.3]*12+[0.1]+[0.3]*12+[0.1]+[0.3]*12
do
    python main/main_lr.py --shot 1 --calib 0 --arch densenet_40 --job_dir round6 --pretrain_dir checkpoint/cifar/cifar10/densenet_40.pt --criterion VBD_dis --lr_type cos --strategy min_sum --compress_rate $cpr --epochs 400 --lr_decay_step '150,250' --weight_decay 0.005 --batch_size 256 --gpu 1
done
