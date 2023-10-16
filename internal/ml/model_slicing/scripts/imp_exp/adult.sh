
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net sparsemax_vertical_sams --expert dnn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log_imp\
    --dataset adult --nfeat 140 --nfield 13  \
    --moe_hid_layer_len 32 --data_nemb 10  \
    --hid_layer_len 32 --sql_nemb 5\
    --dropout 0.1   \
    --K 16 --alpha 2 --max_filter_col 8\
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --seed 3407 --beta 0.01 --gamma 0.02\
    --train_dir DNN_Test  --report_freq 5


~/anaconda3/bin/python3 main.py\
    --device cuda:0 \
    --net sparsemax_vertical_sams \
    --expert dnn    \
    --data_dir /hdd1/sams/data/ \
    --exp /hdd1/sams/tensor_log_imp    \
    --dataset adult --nfeat 140 --nfield 13\
    --moe_hid_layer_len 32 --data_nemb 10  \
    --hid_layer_len 32 --sql_nemb 5 \
    --dropout 0.1      \
    --K 16 --alpha 2 \
    --max_filter_col 8    \
    --epoch 50 \
    --batch_size 1024 \
    --lr 0.002     \
    --seed 3407 \
    --beta 0 \
    --gamma 0.02    \
    --train_dir DNN_Test_wo_beta  



~/anaconda3/bin/python3 main.py\
    --device cuda:0 \
    --net sparsemax_vertical_sams \
    --expert dnn    \
    --data_dir /hdd1/sams/data/ \
    --exp /hdd1/sams/tensor_log_imp    \
    --dataset adult --nfeat 140 --nfield 13\
    --moe_hid_layer_len 32 --data_nemb 10  \
    --hid_layer_len 32 --sql_nemb 5 \
    --dropout 0.3      \
    --K 16 --alpha 2 \
    --max_filter_col 8    \
    --epoch 50 \
    --batch_size 1024 \
    --lr 0.002     \
    --seed 3407 \
    --beta 0.01 \
    --gamma 0    \
    --train_dir DNN_Test_wo_gamma  \
    --report_freq 5



~/anaconda3/bin/python3 main.py\
    --device cuda:0 \
    --net sparsemax_vertical_sams \
    --expert dnn    \
    --data_dir /hdd1/sams/data/ \
    --exp /hdd1/sams/tensor_log_imp    \
    --dataset adult --nfeat 140 --nfield 13\
    --moe_hid_layer_len 32 --data_nemb 10  \
    --hid_layer_len 32 --sql_nemb 5 \
    --dropout 0.2      \
    --K 16 --alpha 2 \
    --max_filter_col 8    \
    --epoch 50 \
    --batch_size 1024 \
    --lr 0.002     \
    --seed 3407 \
    --beta 0 \
    --gamma 0    \
    --train_dir DNN_Test_wo_gamma_wo_beta 