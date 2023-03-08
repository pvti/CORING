#round 1
for cpr in [0.]+[0.1]+[0.25]*2+[0.25]*2+[0.33]*2 [0.]+[0.05]+[0.15]*2+[0.2]*2+[0.25]*2 [0.]+[0.12]+[0.25]*2+[0.36]*2+[0.4]*2
do
   python main/evaluate_kshot.py --arch mobilenet_v2 --job_dir kshot/round1 --use_pretrain --pretrain_dir checkpoint/imagenet/mobilenet_v2.pt --criterion VBD_dis --strategy min_sum --compress_rate $cpr --epochs 200 --weight_decay 0.00004 --batch_size 1024 --shot 5 --gpu 0,1,2,3
done
