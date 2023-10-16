
# K = 2
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net sparsemax_vertical_sams --expert dnn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log_analysis\
    --dataset adult --nfeat 140 --nfield 13  \
    --moe_hid_layer_len 32 --data_nemb 10  \
    --hid_layer_len 32 --sql_nemb 5\
    --dropout 0.1   \
    --K 2 --alpha 2.2 --max_filter_col 8\
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --seed 3047 --beta 0.005\
    --train_dir SAMS_K2  --report_freq 5


# K = 4
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net sparsemax_vertical_sams --expert dnn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log_analysis\
    --dataset adult --nfeat 140 --nfield 13  \
    --moe_hid_layer_len 32 --data_nemb 10  \
    --hid_layer_len 32 --sql_nemb 5\
    --dropout 0.1   \
    --K 4 --alpha 2.3 --max_filter_col 8\
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --seed 3047 --beta 0.005\
    --train_dir SAMS_K4  --report_freq 5


# K = 8
~/anaconda3/bin/python3 main.py \
    --device cuda:1 --net sparsemax_vertical_sams --expert dnn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log_analysis\
    --dataset adult --nfeat 140 --nfield 13  \
    --moe_hid_layer_len 32 --data_nemb 10  \
    --hid_layer_len 32 --sql_nemb 5\
    --dropout 0.1   \
    --K 8 --alpha 2.3 --max_filter_col 8\
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --seed 3047 --beta 0.005\
    --train_dir SAMS_K8  --report_freq 5


# K = 32
~/anaconda3/bin/python3 main.py \
    --device cuda:2 --net sparsemax_vertical_sams --expert dnn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log_analysis\
    --dataset adult --nfeat 140 --nfield 13  \
    --moe_hid_layer_len 32 --data_nemb 10  \
    --hid_layer_len 32 --sql_nemb 5\
    --dropout 0.1   \
    --K 32 --alpha 2.3 --max_filter_col 8\
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --seed 3047 --beta 0.005\
    --train_dir SAMS_K32  --report_freq 5


# K = 64
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net sparsemax_vertical_sams --expert dnn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log_analysis\
    --dataset adult --nfeat 140 --nfield 13  \
    --moe_hid_layer_len 32 --data_nemb 10  \
    --hid_layer_len 32 --sql_nemb 5\
    --dropout 0.1   \
    --K 64 --alpha 2.3 --max_filter_col 8\
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --seed 3047 --beta 0.005\
    --train_dir SAMS_K64  --report_freq 5


# K = 128
~/anaconda3/bin/python3 main.py \
    --device cuda:1 --net sparsemax_vertical_sams --expert dnn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log_analysis\
    --dataset adult --nfeat 140 --nfield 13  \
    --moe_hid_layer_len 32 --data_nemb 10  \
    --hid_layer_len 32 --sql_nemb 5\
    --dropout 0.1   \
    --K 128 --alpha 2.3 --max_filter_col 8\
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --seed 3047 --beta 0.005\
    --train_dir SAMS_K128  --report_freq 5


# K = 256
~/anaconda3/bin/python3 main.py \
    --device cuda:2 --net sparsemax_vertical_sams --expert dnn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log_analysis\
    --dataset adult --nfeat 140 --nfield 13  \
    --moe_hid_layer_len 32 --data_nemb 10  \
    --hid_layer_len 32 --sql_nemb 5\
    --dropout 0.1   \
    --K 256 --alpha 2.3 --max_filter_col 8\
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --seed 3047 --beta 0.005\
    --train_dir SAMS_K256  --report_freq 5