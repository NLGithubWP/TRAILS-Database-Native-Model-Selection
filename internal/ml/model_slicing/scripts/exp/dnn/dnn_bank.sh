

~/anaconda3/bin/python3 main.py \
    --device cuda:1 --net sparsemax_vertical_sams --expert dnn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset bank --nfeat 80 --nfield 16 \
    --moe_hid_layer_len 32 --data_nemb 10  \
    --hid_layer_len 32 --sql_nemb 5\
    --dropout 0.1   \
    --K 16 --alpha 2.3 --max_filter_col 13\
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --seed 3047 --beta=0.005 \
    --train_dir dnn_K16_alpha2-3_beta1e-3 --report_freq 5