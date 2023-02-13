for cpr in [0.]+[0.1]*3+[0.35]*16
do
    CUDA_VISIBLE_DEVICES=0,1 python main/evaluate.py --data_dir data/imagenet/ --arch resnet_50 --job_dir round2 --epochs 200 --lr_type cos --momentum 0.99 --use_pretrain --pretrain_dir checkpoint/imagenet/resnet_50.pth --criterion Euclide_dis --strategy min_sum --compress_rate $cpr --weight_decay 0.0001 --batch_size 256 --rank_conv_prefix rank --gpu '0 1'
done
