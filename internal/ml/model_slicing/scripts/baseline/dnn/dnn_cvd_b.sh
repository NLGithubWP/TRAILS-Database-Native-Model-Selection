

~/anaconda3/bin/python3 main.py \
    --device cuda:2 --net dnn\
    --data_dir "/hdd1/sams/data/" --exp /hdd1/sams/tensor_log_baseline\
    --dataset cvd --nfeat 110 --nfield 11  \
    --moe_hid_layer_len 32 --data_nemb 10 \
    --dropout 0.3 \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --seed 2018 \
    --train_dir "dnn_hidden32" 