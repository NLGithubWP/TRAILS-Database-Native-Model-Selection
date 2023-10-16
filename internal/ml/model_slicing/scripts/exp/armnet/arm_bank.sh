

# specific model armnet add nhid, nhead
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net sparsemax_vertical_sams --expert armnet\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log \
    --dataset bank --nfeat 80 --nfield 16 \
    --moe_hid_layer_len 32 --data_nemb 10\
    --nhead 4 --nhid 16\
    --hid_layer_len 32 --sql_nemb 5\
    --dropout 0.1   \
    --K 8 --alpha 2.5 --max_filter_col 13\
    --epoch 50 --batch_size 1024 --lr 0.005 \
    --seed 3407 --beta 0.005\
    --train_dir armnet_K8_alpha2-5_beta5e-3  --report_freq 5


# *
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net sparsemax_vertical_sams --expert armnet\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log \
    --dataset bank --nfeat 80 --nfield 16 \
    --moe_hid_layer_len 32 --data_nemb 10\
    --nhead 4 --nhid 16\
    --hid_layer_len 32 --sql_nemb 5\
    --dropout 0.1   \
    --K 8 --alpha 2 --max_filter_col 13\
    --epoch 50 --batch_size 1024 --lr 0.005 \
    --seed 3407 --beta 0.005\
    --train_dir armnet_K8_alpha2_beta5e-3  --report_freq 5


# ok
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net sparsemax_vertical_sams --expert armnet\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log \
    --dataset bank --nfeat 80 --nfield 16 \
    --moe_hid_layer_len 32 --data_nemb 10\
    --nhead 4 --nhid 16\
    --hid_layer_len 32 --sql_nemb 5\
    --dropout 0.1   \
    --K 16 --alpha 2.5 --max_filter_col 13\
    --epoch 50 --batch_size 1024 --lr 0.005 \
    --seed 3407 --beta 0.005\
    --train_dir armnet_K16_alpha2-5_beta5e-3  --report_freq 5


