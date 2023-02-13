#round 2
for cpr in [0.05]+[0.1]*2+[0.32]*5+[0.35]*2
do
    python main/main.py --arch googlenet --job_dir round2 --use_pretrain --pretrain_dir checkpoint/cifar/cifar10/googlenet.pt --criterion VBD_dis --strategy min_sum --compress_rate [0.05]+[0.1]*2+[0.32]*5+[0.35]*2 --epochs 0 --lr_decay_step '150,225' --weight_decay 0.005 --batch_size 256
done
