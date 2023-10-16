
# [100, 32, 32, 1]
~/anaconda3/bin/python3 main.py \
    --device cuda:1 --log_folder sams_logs --net dnn\
    --data_dir "/hdd1/sams/data/" --exp /hdd1/sams/tensor_log_baseline\
    --dataset frappe --nfeat 5500 --nfield 10 \
    --moe_hid_layer_len 32  --data_nemb 10 \
    --dropout 0.5 --epoch 100 --batch_size 1024 --lr 0.002 \
    --train_dir "dnn_hidden32" 


