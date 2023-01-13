# example for single combination
#python resnet/gen_rank.py --arch vgg_16_bn --job_dir resnet/result/hrankplus/vgg_16_bn/standard --pretrain_dir ./resnet/checkpoint/cifar/cifar10/vgg_16_bn.pt --criterion cosine_sim --strategy min_sum

#python resnet/gen_rank.py --arch resnet_56 --job_dir resnet/result/hrankplus/resnet_56/standard --pretrain_dir ./resnet/checkpoint/cifar/cifar10/resnet_56.pt --criterion cosine_sim --strategy min_sum

# generate rank for all combinations
for arch in vgg_16_bn resnet_56
do
    for strategy in sum min_sum min_min
    do
        for criterion in cosine_sim Pearson_sim Euclide_dis Manhattan_dis SNR_dis VBD_dis
            do
                python resnet/gen_rank.py --arch $arch --job_dir resnet/result/hrankplus/$arch/standard --pretrain_dir ./resnet/checkpoint/cifar/cifar10/$arch.pt --criterion $criterion --strategy $strategy
            done
    done
done
