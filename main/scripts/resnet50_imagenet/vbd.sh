for cpr in [0.]+[0.1]*3+[0.35]*16 [0.]+[0.12]*3+[0.38]*16 [0.]+[0.25]*3+[0.5]*16 [0.]+[0.5]*3+[0.6]*16
do
    python main/evaluate.py --data_dir data/imagenet/ --arch resnet_50 --job_dir result/hrankplus/resnet_50 --epochs 180 --lr_type cos --momentum 0.99 --pretrain_dir checkpoint/imagenet/resnet_50.pth --criterion VBD_dis --strategy min_sum --compress_rate $cpr --weight_decay 0.0001 --batch_size 512 --rank_conv_prefix rank --gpu 2
done
