#round 1
for shot in 5 10 15
do
    for cpr in [0.]+[0.17]*5+[0.16]*7+[0.18]*1+[0.17]*25 [0.]+[0.25]*5+[0.28]*8+[0.32]*15+[0.34]*10 [0.]+[0.08]*6+[0.09]*6+[0.08]*26
    do
    python main/main_kshot.py --arch densenet_40 --job_dir kshot/round1 --pretrain_dir checkpoint/cifar/cifar10/densenet_40.pt --criterion VBD_dis --strategy min_sum --compress_rate $cpr --shot $shot --epochs 400 --lr_decay_step '150,250' --weight_decay 0.005 --batch_size 256
    done
done