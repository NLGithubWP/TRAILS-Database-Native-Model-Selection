

# specific model armnet add nhid, nhead
~/anaconda3/bin/python3 main.py \
    --device cuda:1 --net sparsemax_vertical_sams --expert armnet\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log \
    --dataset frappe --nfeat 5500 --nfield 10 \
    --moe_hid_layer_len 32 --data_nemb 10\
    --nhid 8 --nhead 4\
    --hid_layer_len 32 --sql_nemb 5\
    --dropout 0.1   \
    --K 16 --alpha 4 --max_filter_col 8\
    --epoch 100 --batch_size 1024 --lr 0.005 \
    --seed 2022 --beta 0.01\
    --train_dir armnet_K16_alpha3-5_beta1e-2  --report_freq 10