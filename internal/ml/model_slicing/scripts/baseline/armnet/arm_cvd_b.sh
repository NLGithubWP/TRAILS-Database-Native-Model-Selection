
~/anaconda3/bin/python3 main.py \
    --device cuda:2  --net armnet \
    --data_dir "/hdd1/sams/data/" \
    --exp /hdd1/sams/tensor_log_baseline\
    --dataset cvd --nfeat 110 --nfield 11  \
    --moe_hid_layer_len 32 --data_nemb 10 \
    --nhead 4 --nhid 16 --dropout 0.4 \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --seed 2020\
    --train_dir "armnet_nhead4_nhid16_hidden32" 