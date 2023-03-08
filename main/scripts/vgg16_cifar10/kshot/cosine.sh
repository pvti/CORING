#round 7
for shot in 1 5 10 15
do
    for cpr in [0.05]*7+[0.2]*5 [0.5]*2+[0.65]*5+[0.8]*5
    do
        python main/main_lr.py --arch vgg_16_bn --job_dir kshot/round7 --pretrain_dir checkpoint/cifar/cifar10/vgg_16_bn.pt --criterion cosine_sim --strategy min_sum --compress_rate $cpr --shot $shot --epochs 400 --lr_type cos --weight_decay 0.005 --batch_size 256
    done
done
