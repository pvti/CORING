#round 1
for cpr in [0.]+[0.05]+[0.1]*2+[0.15]*2+[0.2]*2 [0.]+[0.1]+[0.25]*2+[0.25]*2+[0.3]*2
do
   python main/main.py --arch mobilenet_v2 --job_dir round1 --use_pretrain --pretrain_dir checkpoint/cifar/cifar10/mobilenet_v2.pt --criterion Euclide_dis --strategy min_sum --compress_rate $cpr --epochs 400 --weight_decay 0.005 --batch_size 256
done

#round 2
for cpr in [0.]+[0.05]+[0.1]*2+[0.15]*2+[0.2]*2 [0.]+[0.1]+[0.25]*2+[0.25]*2+[0.3]*2
do
   python main/main.py --arch mobilenet_v2 --job_dir round2 --use_pretrain --pretrain_dir checkpoint/cifar/cifar10/mobilenet_v2.pt --criterion Euclide_dis --strategy min_sum --compress_rate $cpr --epochs 400 --weight_decay 0.006 --batch_size 256
done

#round 3
for cpr in [0.]+[0.05]+[0.1]*2+[0.15]*2+[0.2]*2 [0.]+[0.1]+[0.25]*2+[0.25]*2+[0.3]*2
do
   python main/main.py --arch mobilenet_v2 --job_dir round3 --use_pretrain --pretrain_dir checkpoint/cifar/cifar10/mobilenet_v2.pt --criterion Euclide_dis --strategy min_sum --compress_rate $cpr --epochs 400 --weight_decay 0.004 --batch_size 256
done
