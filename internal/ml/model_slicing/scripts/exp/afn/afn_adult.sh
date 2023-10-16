

# specific model afn add nhid
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net sparsemax_vertical_sams --expert afn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log \
    --dataset adult --nfeat 140 --nfield 13  \
    --moe_hid_layer_len 32 --data_nemb 10\
    --nhid 13\
    --hid_layer_len 32 --sql_nemb 5\
    --dropout 0.2   \
    --K 16 --alpha 2.5 --max_filter_col 8\
    --epoch 50 --batch_size 1024 --lr 0.003 \
    --seed 2025\
    --train_dir Afn_K16_alpha2-5  --report_freq 5