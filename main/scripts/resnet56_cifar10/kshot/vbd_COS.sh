# #round 4
# for cpr in [0.]+[0.12]*2+[0.4]*27 [0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9 [0.]+[0.18]*29
# do
#     python main/main_lr.py --shot 5 --arch resnet_56 --job_dir round4 --pretrain_dir checkpoint/cifar/cifar10/resnet_56.pt --criterion VBD_dis --strategy min_sum --compress_rate $cpr --epochs 400 --lr_decay_step '50,100' --lr_type cos --weight_decay 0.005 --batch_size 256
# done &
# for cpr in [0.]+[0.12]*2+[0.4]*27 [0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9 [0.]+[0.18]*29
# do
#     python main/main_lr.py --shot 10 --arch resnet_56 --job_dir round4 --pretrain_dir checkpoint/cifar/cifar10/resnet_56.pt --criterion VBD_dis --strategy min_sum --compress_rate $cpr --epochs 400 --lr_decay_step '50,100' --lr_type cos --weight_decay 0.005 --batch_size 256
# done &
# for cpr in [0.]+[0.12]*2+[0.4]*27 [0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9 [0.]+[0.18]*29
# do
#     python main/main_lr.py --shot 15 --arch resnet_56 --job_dir round4 --pretrain_dir checkpoint/cifar/cifar10/resnet_56.pt --criterion VBD_dis --strategy min_sum --compress_rate $cpr --epochs 400 --lr_decay_step '50,100' --lr_type cos --weight_decay 0.005 --batch_size 256
# done

#round 5
for cpr in [0.]+[0.12]*2+[0.4]*27 [0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9 [0.]+[0.18]*29
do
    python main/main_lr.py --shot 1 --calib 0 --arch resnet_56 --job_dir round5 --pretrain_dir checkpoint/cifar/cifar10/resnet_56.pt --criterion VBD_dis --strategy min_sum --compress_rate $cpr --epochs 400 --lr_decay_step '50,100' --lr_type cos --weight_decay 0.005 --batch_size 256
done &
for cpr in [0.]+[0.12]*2+[0.4]*27 [0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9 [0.]+[0.18]*29
do
    python main/main_lr.py --shot 5 --arch resnet_56 --job_dir round5 --pretrain_dir checkpoint/cifar/cifar10/resnet_56.pt --criterion VBD_dis --strategy min_sum --compress_rate $cpr --epochs 400 --lr_decay_step '50,100' --lr_type cos --weight_decay 0.005 --batch_size 256
done &
for cpr in [0.]+[0.12]*2+[0.4]*27 [0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9 [0.]+[0.18]*29
do
    python main/main_lr.py --shot 10 --arch resnet_56 --job_dir round5 --pretrain_dir checkpoint/cifar/cifar10/resnet_56.pt --criterion VBD_dis --strategy min_sum --compress_rate $cpr --epochs 400 --lr_decay_step '50,100' --lr_type cos --weight_decay 0.005 --batch_size 256
done &
for cpr in [0.]+[0.12]*2+[0.4]*27 [0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9 [0.]+[0.18]*29
do
    python main/main_lr.py --shot 15 --arch resnet_56 --job_dir round5 --pretrain_dir checkpoint/cifar/cifar10/resnet_56.pt --criterion VBD_dis --strategy min_sum --compress_rate $cpr --epochs 400 --lr_decay_step '50,100' --lr_type cos --weight_decay 0.005 --batch_size 256
done
