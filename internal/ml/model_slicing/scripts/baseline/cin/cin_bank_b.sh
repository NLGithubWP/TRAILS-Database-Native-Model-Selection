~/anaconda3/bin/python3 main.py \
    --device cuda:2  --net cin \
    --data_dir "/hdd1/sams/data/" \
    --exp /hdd1/sams/tensor_log_baseline\
    --dataset bank --nfeat 80 --nfield 16\
    --nhid 16 --data_nemb 10 \
    --epoch 50 --batch_size 1024 --lr 0.001 \
    --seed 1998 \
    --train_dir "cin_hid16"