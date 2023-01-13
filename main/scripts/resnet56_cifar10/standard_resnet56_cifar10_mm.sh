# loop over all combinations
for strategy in min_min
do
    for criterion in VBD_dis cosine_sim Euclide_dis SNR_dis Pearson_sim Manhattan_dis
        do
            for cpr in [0.]+[0.18]*29 [0.]+[0.12]*2+[0.4]*27 [0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9
            do
                python resnet/main.py --arch resnet_56 --job_dir resnet/result/hrankplus/resnet_56/standard --use_pretrain --pretrain_dir ./resnet/checkpoint/cifar/cifar10/resnet_56.pt --criterion $criterion --strategy $strategy --compress_rate $cpr
            done
        done
done
