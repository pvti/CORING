#round 1
for cpr in [0.3]+[0.6]*2+[0.7]*5+[0.8]*2 [0.4]+[0.85]*2+[0.9]*5+[0.9]*2
do
    python main/main_kshot.py --arch googlenet --job_dir kshot/round1 --pretrain_dir checkpoint/cifar/cifar10/googlenet.pt --criterion cosine_sim --strategy min_sum --compress_rate $cpr --epochs 400 --lr_decay_step '150,225' --weight_decay 0.005 --batch_size 256
done
