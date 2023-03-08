#round 5
for cpr in [0.05]+[0.1]*2+[0.32]*5+[0.35]*2 [0.3]+[0.6]*2+[0.7]*5+[0.8]*2 [0.4]+[0.85]*2+[0.9]*5+[0.9]*2
do
    python main/main_lr.py --shot 1 --calib 0 --arch googlenet --job_dir round5 --pretrain_dir checkpoint/cifar/cifar10/googlenet.pt --criterion cosine_sim --lr_type cos --strategy min_sum --compress_rate $cpr --epochs 400 --lr_decay_step '150,250' --weight_decay 0.005 --batch_size 256
done &

for cpr in [0.05]+[0.1]*2+[0.32]*5+[0.35]*2 [0.3]+[0.6]*2+[0.7]*5+[0.8]*2 [0.4]+[0.85]*2+[0.9]*5+[0.9]*2
do
    python main/main_lr.py --shot 1 --calib 0 --arch googlenet --job_dir round5 --pretrain_dir checkpoint/cifar/cifar10/googlenet.pt --criterion Euclide_dis --lr_type cos --strategy min_sum --compress_rate $cpr --epochs 400 --lr_decay_step '150,250' --weight_decay 0.005 --batch_size 256 --gpu 1
done &

for cpr in [0.05]+[0.1]*2+[0.32]*5+[0.35]*2 [0.3]+[0.6]*2+[0.7]*5+[0.8]*2 [0.4]+[0.85]*2+[0.9]*5+[0.9]*2
do
    python main/main_lr.py --shot 1 --calib 0 --arch googlenet --job_dir round5 --pretrain_dir checkpoint/cifar/cifar10/googlenet.pt --criterion VBD_dis --lr_type cos --strategy min_sum --compress_rate $cpr --epochs 400 --lr_decay_step '150,250' --weight_decay 0.005 --batch_size 256 --gpu 2
done

