# DNN
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net sparseMoE --expert dnn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log_baseline\
    --dataset bank --nfeat 80 --nfield 16 \
    --moe_hid_layer_len 32 --data_nemb 10  \
    --dropout 0.1  \
    --K 16 --C 1 \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --seed 2023\
    --train_dir MoESparse_DNN  --report_freq 5


# CIN
~/anaconda3/bin/python3 main.py \
    --device cuda:1 --net sparseMoE  --expert cin\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log_baseline \
    --dataset bank --nfeat 80 --nfield 16 \
    --moe_hid_layer_len 32 --data_nemb 10\
    --nhid 16\
    --dropout 0.1  \
    --K 16 --C 1\
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --seed 2023\
    --train_dir MoESparse_CIN  --report_freq 5

#Afn
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net sparseMoE  --expert afn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log_baseline \
    --dataset bank --nfeat 80 --nfield 16 \
    --moe_hid_layer_len 32 --data_nemb 10\
    --nhid 16\
    --dropout 0.1  \
    --K 16 --C 1 \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --seed 2023\
    --train_dir MoESparse_AFN  --report_freq 5

# Armnet
~/anaconda3/bin/python3 main.py \
    --device cuda:1 --net sparseMoE --expert armnet\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log \
    --dataset bank --nfeat 80 --nfield 16 \
    --moe_hid_layer_len 32 --data_nemb 10\
    --nhead 4 --nhid 16\
    --dropout 0.1   \
    --K 16 --C 1\
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --seed 2022\
    --train_dir MoESparse_ARMNet  --report_freq 5

# nfm
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net sparseMoE --expert nfm\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset bank --nfeat 80 --nfield 16 \
    --moe_hid_layer_len 32 --data_nemb 10  \
    --dropout 0.1   \
    --K 16 --C 1\
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --seed 2022 \
    --train_dir MoESparse_NFM  --report_freq 5