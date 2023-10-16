
# [160 -> 32 -> 32 -> 32 -> 1]
~/anaconda3/bin/python3 main.py \
    --device cuda:2  --net armnet \
    --data_dir "/hdd1/sams/data/" \
    --exp /hdd1/sams/tensor_log_baseline\
    --dataset bank --nfeat 80 --nfield 16\
    --moe_hid_layer_len 32 --data_nemb 10 \
    --nhead 4 --nhid 16 --dropout 0.2 \
    --epoch 50 --batch_size 1024 --lr 0.001 \
    --seed 1998\
    --train_dir "armnet_nhead4_nhid16_hidden32_d2e-2" 