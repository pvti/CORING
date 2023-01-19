# generate rank for all combinations
for arch in vgg_16_bn resnet_56 densenet_40
do
    for strategy in min_sum
    do
        for criterion in cosine_sim Euclide_dis VBD_dis
            do
                python main/gen_rank.py --arch $arch --pretrain_dir checkpoint/cifar/cifar10/$arch.pt --criterion $criterion --strategy $strategy
            done
    done
done
