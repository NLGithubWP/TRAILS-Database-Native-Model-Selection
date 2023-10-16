
# K 2
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net sparsemax_vertical_sams --expert afn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log \
    --dataset frappe --nfeat 5500 --nfield 10 \
    --moe_hid_layer_len 32 --data_nemb 10\
    --nhid 10\
    --hid_layer_len 32 --sql_nemb 5\
    --dropout 0.1   \
    --K 2 --alpha 3.6 --max_filter_col 9\
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --seed 2023\
    --train_dir SAMS_AFN_K2  --report_freq 5


# K 4
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net sparsemax_vertical_sams --expert afn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log \
    --dataset frappe --nfeat 5500 --nfield 10 \
    --moe_hid_layer_len 32 --data_nemb 10\
    --nhid 10\
    --hid_layer_len 32 --sql_nemb 5\
    --dropout 0.1   \
    --K 4 --alpha 3.6 --max_filter_col 9\
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --seed 2023\
    --train_dir SAMS_AFN_K4  --report_freq 5



# K 8
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net sparsemax_vertical_sams --expert afn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log \
    --dataset frappe --nfeat 5500 --nfield 10 \
    --moe_hid_layer_len 32 --data_nemb 10\
    --nhid 10\
    --hid_layer_len 32 --sql_nemb 5\
    --dropout 0.1   \
    --K 8 --alpha 3.6 --max_filter_col 9\
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --seed 2023\
    --train_dir SAMS_AFN_K8  --report_freq 5



# K 32
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net sparsemax_vertical_sams --expert afn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log \
    --dataset frappe --nfeat 5500 --nfield 10 \
    --moe_hid_layer_len 32 --data_nemb 10\
    --nhid 10\
    --hid_layer_len 32 --sql_nemb 5\
    --dropout 0.1   \
    --K 32 --alpha 3.6 --max_filter_col 9\
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --seed 2023\
    --train_dir SAMS_AFN_K32  --report_freq 5


# K 64
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net sparsemax_vertical_sams --expert afn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log \
    --dataset frappe --nfeat 5500 --nfield 10 \
    --moe_hid_layer_len 32 --data_nemb 10\
    --nhid 10\
    --hid_layer_len 32 --sql_nemb 5\
    --dropout 0.1   \
    --K 64 --alpha 3.6 --max_filter_col 9\
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --seed 2023\
    --train_dir SAMS_AFN_K64  --report_freq 5



# K 128
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net sparsemax_vertical_sams --expert afn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log \
    --dataset frappe --nfeat 5500 --nfield 10 \
    --moe_hid_layer_len 32 --data_nemb 10\
    --nhid 10\
    --hid_layer_len 32 --sql_nemb 5\
    --dropout 0.1   \
    --K 128 --alpha 3.6 --max_filter_col 9\
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --seed 2023\
    --train_dir SAMS_AFN_K128  --report_freq 5


# K 256
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net sparsemax_vertical_sams --expert afn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log \
    --dataset frappe --nfeat 5500 --nfield 10 \
    --moe_hid_layer_len 32 --data_nemb 10\
    --nhid 10\
    --hid_layer_len 32 --sql_nemb 5\
    --dropout 0.1   \
    --K 256 --alpha 3.6 --max_filter_col 9\
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --seed 2023\
    --train_dir SAMS_AFN_K256  --report_freq 5