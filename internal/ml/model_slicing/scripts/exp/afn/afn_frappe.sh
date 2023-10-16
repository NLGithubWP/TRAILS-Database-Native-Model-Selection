
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net sparsemax_vertical_sams --expert afn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log \
    --dataset frappe --nfeat 5500 --nfield 10 \
    --moe_hid_layer_len 32 --data_nemb 10\
    --nhid 10\
    --hid_layer_len 32 --sql_nemb 5\
    --dropout 0.1   \
    --K 16 --alpha 4 --max_filter_col 9\
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --seed 2023\
    --train_dir afn_K16_alpha4  --report_freq 5