

# adult
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net sparsemax_vertical_sams --expert nfm\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset adult --nfeat 140 --nfield 13  \
    --moe_hid_layer_len 32 --data_nemb 10  \
    --hid_layer_len 32 --sql_nemb 5\
    --dropout 0.1   \
    --K 16 --alpha 2.3 --max_filter_col 10\
    --epoch 50 --batch_size 1024 --lr 0.001 \
    --seed 3047 --beta 0.005\
    --train_dir nfm_K16_alpha2-3  --report_freq 5


# bank
~/anaconda3/bin/python3 main.py \
    --device cuda:1 --net sparsemax_vertical_sams --expert nfm\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset bank --nfeat 80 --nfield 16 \
    --moe_hid_layer_len 32 --data_nemb 10  \
    --hid_layer_len 32 --sql_nemb 5\
    --dropout 0.1   \
    --K 16 --alpha 2.3 --max_filter_col 12\
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --seed 3407 --beta=0.002 \
    --train_dir nfm_K16_alpha2-3_beta1e-3 --report_freq 5


# disease
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net sparsemax_vertical_sams --expert nfm\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset cvd --nfeat 110 --nfield 11  \
    --moe_hid_layer_len 32 --data_nemb 10  \
    --hid_layer_len 32 --sql_nemb 5\
    --dropout 0.2   \
    --K 16 --alpha 2.3 --max_filter_col 8\
    --epoch 50 --batch_size 1024 --lr 0.001 \
    --seed 3047 --beta=0.001 \
    --train_dir nfm_K16_alpha2-3  --report_freq 5

# appRec
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net sparsemax_vertical_sams --expert nfm\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset frappe --nfeat 5500 --nfield 10 \
    --moe_hid_layer_len 32 --data_nemb 10  \
    --hid_layer_len 32 --sql_nemb 5\
    --dropout 0.1   \
    --K 16 --alpha 5 --max_filter_col 10\
    --epoch 100 --batch_size 1024 --lr 0.005 \
    --seed 2023 --beta=0.002 \
    --train_dir nfm_K16_alpha5  --report_freq 5

~/anaconda3/bin/python3 main.py     --device cuda:0 --net sparsemax_vertical_sams --expert nfm    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log    --dataset frappe --nfeat 5500 --nfield 10     --moe_hid_layer_len 32 --data_nemb 10      --hid_layer_len 32 --sql_nemb 5    --dropout 0.1       --K 16 --alpha 4 --max_filter_col 8    --epoch 100 --batch_size 1024 --lr 0.002     --seed 2023 --beta=0.005     --train_dir nfm_K16_alpha4  --report_freq 5