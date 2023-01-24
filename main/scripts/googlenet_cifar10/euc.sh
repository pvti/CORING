#round 1
for cpr in [0.3]+[0.6]*2+[0.7]*5+[0.8]*2 [0.4]+[0.85]*2+[0.9]*5+[0.9]*2
do
    python main/main.py --arch googlenet --job_dir round1 --use_pretrain --pretrain_dir checkpoint/cifar/cifar10/googlenet.pt --criterion Euclide_dis --strategy min_sum --compress_rate $cpr --epochs 400 --lr_decay_step '150,225' --weight_decay 0.005 --batch_size 256
done
