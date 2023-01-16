for strategy in min_sum
    do
        for criterion in cosine_sim Euclide_dis VBD_dis
            do
                python main/gen_rank.py --arch resnet_50 --job_dir result/hrankplus/resnet_50/standard --pretrain_dir checkpoint/imagenet/resnet_50.pth --criterion $criterion --strategy $strategy
            done
    done
