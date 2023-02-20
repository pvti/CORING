# #round 1
# for shot in 5 10 15
# do
#     for cpr in [0.21]*7+[0.75]*5 [0.3]*7+[0.75]*5 [0.45]*7+[0.78]*5
#     do
#     python main/main_kshot.py --arch vgg_16_bn --job_dir kshot/round1 --pretrain_dir checkpoint/cifar/cifar10/vgg_16_bn.pt --criterion Euclide_dis --strategy min_sum --compress_rate $cpr --shot $shot --epochs 400 --weight_decay 0.005 --batch_size 256
#     done
# done

# #round 2
# for shot in 5 10 15
# do
#     for cpr in [0.21]*7+[0.75]*5 [0.3]*7+[0.75]*5 [0.45]*7+[0.78]*5
#     do
#         python main/main_kshot.py --arch vgg_16_bn --job_dir kshot/round2 --pretrain_dir checkpoint/cifar/cifar10/vgg_16_bn.pt --criterion Euclide_dis --strategy min_sum --compress_rate $cpr --shot $shot --epochs 400 --weight_decay 0.006 --batch_size 256
#     done
# done

# #round 3
# for shot in 5 10 15
# do
#     for cpr in [0.21]*7+[0.75]*5 [0.3]*7+[0.75]*5 [0.45]*7+[0.78]*5
#     do
#         python main/main_kshot.py --arch vgg_16_bn --job_dir kshot/round3 --pretrain_dir checkpoint/cifar/cifar10/vgg_16_bn.pt --criterion Euclide_dis --strategy min_sum --compress_rate $cpr --shot $shot --epochs 400 --weight_decay 0.005 --batch_size 128
#     done
# done

#round 4
for shot in 5 10 15
do
    for cpr in [0.05]*7+[0.2]*5 [0.1]*7+[0.38]*5 [0.5]*2+[0.65]*5+[0.8]*5
    do
    python main/main_kshot.py --arch vgg_16_bn --job_dir kshot/round4 --pretrain_dir checkpoint/cifar/cifar10/vgg_16_bn.pt --criterion cosine_sim --strategy min_sum --compress_rate $cpr --shot $shot --epochs 400 --weight_decay 0.005 --batch_size 256
    done
done

#round 5
for shot in 5 10 15
do
    for cpr in [0.05]*7+[0.2]*5 [0.1]*7+[0.38]*5 [0.5]*2+[0.65]*5+[0.8]*5
    do
        python main/main_kshot.py --arch vgg_16_bn --job_dir kshot/round5 --pretrain_dir checkpoint/cifar/cifar10/vgg_16_bn.pt --criterion cosine_sim --strategy min_sum --compress_rate $cpr --shot $shot --epochs 400 --weight_decay 0.006 --batch_size 256
    done
done

#round 6
for shot in 5 10 15
do
    for cpr in [0.05]*7+[0.2]*5 [0.1]*7+[0.38]*5 [0.5]*2+[0.65]*5+[0.8]*5
    do
        python main/main_kshot.py --arch vgg_16_bn --job_dir kshot/round6 --pretrain_dir checkpoint/cifar/cifar10/vgg_16_bn.pt --criterion cosine_sim --strategy min_sum --compress_rate $cpr --shot $shot --epochs 400 --weight_decay 0.005 --batch_size 128
    done
done
