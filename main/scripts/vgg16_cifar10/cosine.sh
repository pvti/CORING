for cpr in [0.21]*7+[0.75]*5 [0.3]*7+[0.75]*5 [0.45]*7+[0.78]*5
do
    python main/main.py --arch vgg_16_bn --job_dir result/hrankplus/vgg_16_bn/standard --use_pretrain --pretrain_dir checkpoint/cifar/cifar10/vgg_16_bn.pt --criterion cosine_sim --strategy min_sum --compress_rate $cpr --epochs 150 --weight_decay 0.006 --batch_size 512
done
