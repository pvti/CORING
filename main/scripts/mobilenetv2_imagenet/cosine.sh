#round 3
for cpr in [0.]+[0.1]+[0.25]*2+[0.25]*2+[0.33]*2 [0.]+[0.05]+[0.15]*2+[0.2]*2+[0.25]*2 [0.]+[0.12]+[0.25]*2+[0.36]*2+[0.4]*2
do
   python main/evaluate_disable_thop.py --arch mobilenet_v2 --job_dir round3 --use_pretrain --pretrain_dir checkpoint/imagenet/mobilenet_v2.pt --criterion cosine_sim --strategy min_sum --compress_rate $cpr --epochs 200 --momentum 0.9 --weight_decay 0.0005 --batch_size 2048 --learning_rate 0.05 --gpu 0,1,2,3,4
done
