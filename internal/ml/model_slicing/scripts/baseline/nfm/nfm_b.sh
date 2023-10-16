

# adult
~/anaconda3/bin/python3 main.py \
    --device cuda:2 --net nfm\
    --data_dir "/hdd1/sams/data/" --exp /hdd1/sams/tensor_log_baseline\
    --dataset adult --nfeat 140 --nfield 13  \
    --moe_hid_layer_len 32 --data_nemb 10 \
    --dropout 0.3 \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir "nfm" 

# disease
~/anaconda3/bin/python3 main.py \
    --device cuda:2 --net nfm\
    --data_dir "/hdd1/sams/data/" --exp /hdd1/sams/tensor_log_baseline\
    --dataset cvd --nfeat 110 --nfield 11  \
    --moe_hid_layer_len 32 --data_nemb 10 \
    --dropout 0.3 \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --seed 1999 \
    --train_dir "nfm" 

# bank
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net nfm\
    --data_dir "/hdd1/sams/data/" --exp /hdd1/sams/tensor_log_baseline\
    --dataset bank --nfeat 80 --nfield 16\
    --moe_hid_layer_len 32 --data_nemb 10 \
    --dropout 0.5 \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --seed 2000 \
    --train_dir "nfm" 


# App Rec

~/anaconda3/bin/python3 main.py \
    --device cuda:1 --net nfm\
    --data_dir "/hdd1/sams/data/" --exp /hdd1/sams/tensor_log_baseline\
    --dataset frappe --nfeat 5500 --nfield 10 \
    --moe_hid_layer_len 32  --data_nemb 10 \
    --dropout 0.3 --epoch 100 --batch_size 1024 --lr 0.002 \
    --train_dir "nfm" 
