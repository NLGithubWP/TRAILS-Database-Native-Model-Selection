
~/anaconda3/bin/python3 main.py \
    --device cuda:2  --net afn \
    --data_dir "/hdd1/sams/data/" \
    --exp /hdd1/sams/tensor_log_baseline\
    --dataset adult --nfeat 140 --nfield 13  \
    --moe_hid_layer_len 32 --data_nemb 10 \
    --nhid 13 --dropout 0.3 \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir "afn_nhid13" 