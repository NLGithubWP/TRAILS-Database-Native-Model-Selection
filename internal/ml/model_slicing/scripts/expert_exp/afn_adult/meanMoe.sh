
# AFN_K2
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net meanMoE  --expert afn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log_analysis\
    --dataset adult --nfeat 140 --nfield 13  \
    --moe_hid_layer_len 32 --data_nemb 10  \
    --dropout 0.4   \
    --nhid 13\
    --K 2 \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --seed 2023\
    --train_dir MoEMean_AFN_K2  --report_freq 5


# AFN_K4
~/anaconda3/bin/python3 main.py \
    --device cuda:1 --net meanMoE --expert afn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log_analysis\
    --dataset adult --nfeat 140 --nfield 13  \
    --moe_hid_layer_len 32 --data_nemb 10  \
    --dropout 0.4   \
    --nhid 13\
    --K 4 \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --seed 2023\
    --train_dir MoEMean_AFN_K4  --report_freq 5


# AFN_K8
~/anaconda3/bin/python3 main.py \
    --device cuda:2 --net meanMoE --expert afn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log_analysis\
    --dataset adult --nfeat 140 --nfield 13  \
    --moe_hid_layer_len 32 --data_nemb 10  \
    --dropout 0.4   \
    --nhid 13\
    --K 8 \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --seed 2023\
    --train_dir MoEMean_AFN_K8  --report_freq 5



# AFN_K32
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net meanMoE --expert afn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log_analysis\
    --dataset adult --nfeat 140 --nfield 13  \
    --moe_hid_layer_len 32 --data_nemb 10  \
    --dropout 0.2   \
    --nhid 13\
    --K 32 \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --seed 2023\
    --train_dir MoEMean_AFN_K32  --report_freq 5


# AFN_K64
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net meanMoE --expert afn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log_analysis\
    --dataset adult --nfeat 140 --nfield 13  \
    --moe_hid_layer_len 32 --data_nemb 10  \
    --dropout 0.2   \
    --nhid 13\
    --K 64 \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --seed 2023\
    --train_dir MoEMean_AFN_K64  --report_freq 5

# AFN_K128
~/anaconda3/bin/python3 main.py \
    --device cuda:1 --net meanMoE --expert afn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log_analysis\
    --dataset adult --nfeat 140 --nfield 13  \
    --moe_hid_layer_len 32 --data_nemb 10  \
    --dropout 0.2   \
    --nhid 13\
    --K 128 \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --seed 2023\
    --train_dir MoEMean_AFN_K128  --report_freq 5

# AFN_K256
~/anaconda3/bin/python3 main.py \
    --device cuda:2 --net meanMoE --expert afn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log_analysis\
    --dataset adult --nfeat 140 --nfield 13  \
    --moe_hid_layer_len 32 --data_nemb 10  \
    --dropout 0.2   \
    --nhid 13\
    --K 256 \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --seed 2023\
    --train_dir MoEMean_AFN_K256  --report_freq 5