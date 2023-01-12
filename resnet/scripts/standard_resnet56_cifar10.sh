# # example for single combination
# #hrank1 vbd
# python resnet/main.py --job_dir resnet/result/hrankplus/resnet_56/standard --data_dir ./data --use_pretrain --pretrain_dir ./resnet/checkpoint/cifar/cifar10/resnet_56.pt --arch resnet_56 --compress_rate [0.]+[0.18]*29 --lr 0.01 --epochs 300 --weight_decay 0.006 --lr_decay_step "150,225" --criterion VBD_dis

# #hrank1 cosine
# python resnet/main.py --job_dir resnet/result/hrankplus/resnet_56/standard --data_dir ./data --use_pretrain --pretrain_dir ./resnet/checkpoint/cifar/cifar10/resnet_56.pt --arch resnet_56 --compress_rate [0.]+[0.18]*29 --lr 0.01 --epochs 300 --weight_decay 0.006 --lr_decay_step "150,225" --criterion cosine_sim

# #hrank1 Euclide
# python resnet/main.py --job_dir resnet/result/hrankplus/resnet_56/standard --data_dir ./data --use_pretrain --pretrain_dir ./resnet/checkpoint/cifar/cifar10/resnet_56.pt --arch resnet_56 --compress_rate [0.]+[0.18]*29 --lr 0.01 --epochs 300 --weight_decay 0.006 --lr_decay_step "150,225" --criterion Euclide_dis

# #hrank2 vbd
# python resnet/main.py --job_dir resnet/result/hrankplus/resnet_56/standard --data_dir ./data --use_pretrain --pretrain_dir ./resnet/checkpoint/cifar/cifar10/resnet_56.pt --arch resnet_56 --compress_rate [0.]+[0.12]*2+[0.4]*27 --lr 0.01 --epochs 300 --weight_decay 0.006 --lr_decay_step "150,225" --criterion VBD_dis

# #hrank2 cosine
# python resnet/main.py --job_dir resnet/result/hrankplus/resnet_56/standard --data_dir ./data --use_pretrain --pretrain_dir ./resnet/checkpoint/cifar/cifar10/resnet_56.pt --arch resnet_56 --compress_rate [0.]+[0.12]*2+[0.4]*27 --lr 0.01 --epochs 300 --weight_decay 0.006 --lr_decay_step "150,225" --criterion cosine_sim

# #hrank2 Euclide
# python resnet/main.py --job_dir resnet/result/hrankplus/resnet_56/standard --data_dir ./data --use_pretrain --pretrain_dir ./resnet/checkpoint/cifar/cifar10/resnet_56.pt --arch resnet_56 --compress_rate [0.]+[0.12]*2+[0.4]*27 --lr 0.01 --epochs 300 --weight_decay 0.006 --lr_decay_step "150,225" --criterion Euclide_dis

# #hrank3 vbd
# python resnet/main.py --job_dir resnet/result/hrankplus/resnet_56/standard --data_dir ./data --use_pretrain --pretrain_dir ./resnet/checkpoint/cifar/cifar10/resnet_56.pt --arch resnet_56 --compress_rate [0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9 --lr 0.01 --epochs 300 --weight_decay 0.006 --lr_decay_step "150,225" --criterion VBD_dis

# #hrank3 cosine
# python resnet/main.py --job_dir resnet/result/hrankplus/resnet_56/standard --data_dir ./data --use_pretrain --pretrain_dir ./resnet/checkpoint/cifar/cifar10/resnet_56.pt --arch resnet_56 --compress_rate [0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9 --lr 0.01 --epochs 300 --weight_decay 0.006 --lr_decay_step "150,225" --criterion cosine_sim

# #hrank3 Euclide
# python resnet/main.py --job_dir resnet/result/hrankplus/resnet_56/standard --data_dir ./data --use_pretrain --pretrain_dir ./resnet/checkpoint/cifar/cifar10/resnet_56.pt --arch resnet_56 --compress_rate [0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9 --lr 0.01 --epochs 300 --weight_decay 0.006 --lr_decay_step "150,225" --criterion Euclide_dis

# loop over all combinations
for strategy in min_min min_sum sum
do
    for criterion in VBD_dis cosine_sim Euclide_dis SNR_dis Pearson_sim Manhattan_dis
        do
            for cpr in [0.]+[0.18]*29 [0.]+[0.12]*2+[0.4]*27 [0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9
            do
                python resnet/main.py --arch resnet_56 --job_dir resnet/result/hrankplus/resnet_56/standard --use_pretrain --pretrain_dir ./resnet/checkpoint/cifar/cifar10/resnet_56.pt --criterion $criterion --strategy $strategy --compress_rate $cpr
            done
        done
done
