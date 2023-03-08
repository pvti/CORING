#round 1
#for cpr in [0.]+[0.2]*12+[0.]+[0.2]*12+[0.]+[0.2]*12 [0.]+[0.3]*12+[0.1]+[0.3]*12+[0.1]+[0.3]*12 [0.]+[0.4]*12+[0.]+[0.4]*12+[0.]+[0.4]*12
#do
#    python main/main.py --arch densenet_40 --job_dir round1 --use_pretrain --pretrain_dir checkpoint/cifar/cifar10/densenet_40.pt --criterion VBD_dis --strategy min_sum --compress_rate $cpr --epochs 300 --lr_decay_step '150,225' --weight_decay 0.002 --batch_size 256 --gpu 1
#done

#round 2
# for cpr in [0.]+[0.2]*12+[0.]+[0.2]*12+[0.]+[0.2]*12 [0.]+[0.3]*12+[0.1]+[0.3]*12+[0.1]+[0.3]*12 [0.]+[0.4]*12+[0.]+[0.4]*12+[0.]+[0.4]*12
# do
#     python main/main.py --arch densenet_40 --job_dir round2 --use_pretrain --pretrain_dir checkpoint/cifar/cifar10/densenet_40.pt --criterion VBD_dis --strategy min_sum --compress_rate $cpr --epochs 300 --lr_decay_step '150,225' --weight_decay 0.003 --batch_size 256 --gpu 1
# done

# #round 3
# for cpr in [0.]+[0.2]*12+[0.]+[0.2]*12+[0.]+[0.2]*12 [0.]+[0.3]*12+[0.1]+[0.3]*12+[0.1]+[0.3]*12 [0.]+[0.4]*12+[0.]+[0.4]*12+[0.]+[0.4]*12
# do
#     python main/main.py --arch densenet_40 --job_dir round3 --use_pretrain --pretrain_dir checkpoint/cifar/cifar10/densenet_40.pt --criterion VBD_dis --strategy min_sum --compress_rate $cpr --epochs 300 --lr_decay_step '150,225' --weight_decay 0.002 --batch_size 128 --gpu 1
# done

# #round 5
# for cpr in [0.]+[0.17]*5+[0.16]*7+[0.18]*1+[0.17]*25 [0.]+[0.25]*5+[0.28]*8+[0.32]*15+[0.34]*10 [0.]+[0.08]*6+[0.09]*6+[0.08]*26
# do
#     python main/main.py --arch densenet_40 --job_dir round5 --use_pretrain --pretrain_dir checkpoint/cifar/cifar10/densenet_40.pt --criterion Euclide_dis --strategy min_sum --compress_rate $cpr --epochs 400 --lr_decay_step '150,250' --weight_decay 0.002 --batch_size 256
# done

#round 9
for cpr in [0.]+[0.17]*5+[0.16]*7+[0.18]*1+[0.17]*25 [0.]+[0.25]*5+[0.28]*8+[0.32]*15+[0.34]*10 [0.]+[0.08]*6+[0.09]*6+[0.08]*26 [0.]+[0.2]*12+[0.]+[0.2]*12+[0.]+[0.2]*12 [0.]+[0.4]*12+[0.]+[0.4]*12+[0.]+[0.4]*12 [0.]+[0.3]*12+[0.1]+[0.3]*12+[0.1]+[0.3]*12
do
    python main/main_lr.py --shot 1 --calib 0 --arch densenet_40 --job_dir round9 --pretrain_dir checkpoint/cifar/cifar10/densenet_40.pt --criterion VBD_dis --lr_type cos --strategy min_sum --compress_rate $cpr --epochs 400 --lr_decay_step '150,250' --weight_decay 0.002 --batch_size 256
done
