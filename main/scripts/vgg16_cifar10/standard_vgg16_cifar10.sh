# loop over all combinations
for strategy in min_min min_sum sum
do
    for criterion in VBD_dis cosine_sim Euclide_dis SNR_dis Pearson_sim Manhattan_dis  
        do
            for cpr in [0.21]*7+[0.75]*5 [0.3]*7+[0.75]*5 [0.45]*7+[0.78]*5
            do
                python resnet/main.py --arch vgg_16_bn --job_dir resnet/result/hrankplus/vgg_16_bn/standard --use_pretrain --pretrain_dir ./resnet/checkpoint/cifar/cifar10/vgg_16_bn.pt --criterion $criterion --strategy $strategy --compress_rate $cpr
            done
        done
done
