
# DNN
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net meanMoE --expert dnn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log_baseline\
    --dataset cvd --nfeat 110 --nfield 11   \
    --moe_hid_layer_len 32 --data_nemb 10  \
    --dropout 0.2   \
    --K 16 --max_filter_col 8\
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --seed 2023\
    --train_dir MoEMean_DNN  --report_freq 5


# CIN
~/anaconda3/bin/python3 main.py \
    --device cuda:1 --net meanMoE  --expert cin\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log_baseline \
    --dataset cvd --nfeat 110 --nfield 11   \
    --moe_hid_layer_len 32 --data_nemb 10\
    --nhid 16\
    --dropout 0.3   \
    --K 16 --max_filter_col 8\
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --seed 2022\
    --train_dir MoEMean_CIN  --report_freq 5

#Afn
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net meanMoE  --expert afn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log_baseline \
    --dataset cvd --nfeat 110 --nfield 11   \
    --moe_hid_layer_len 32 --data_nemb 10\
    --nhid 13\
    --dropout 0.3   \
    --K 16 --max_filter_col 8\
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --seed 2025\
    --train_dir MoEMean_AFN  --report_freq 5

# Armnet
~/anaconda3/bin/python3 main.py \
    --device cuda:2 --net meanMoE --expert armnet\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log \
    --dataset cvd --nfeat 110 --nfield 11   \
    --moe_hid_layer_len 32 --data_nemb 10\
    --nhead 4 --nhid 16\
    --dropout 0.3   \
    --K 16 --max_filter_col 8\
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --seed 2022\
    --train_dir MoEMean_ARMNet  --report_freq 5


#nfm
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net meanMoE --expert nfm\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset cvd --nfeat 110 --nfield 11   \
    --moe_hid_layer_len 32 --data_nemb 10  \
    --dropout 0.1   \
    --K 16 \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --seed 2022 \
    --train_dir MoEMean_NFM  --report_freq 5