# K2
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net meanMoE --expert dnn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log_analysis\
    --dataset frappe --nfeat 5500 --nfield 10 \
    --moe_hid_layer_len 32 --data_nemb 10  \
    --dropout 0.3   \
    --K 2 \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --seed 2023\
    --train_dir MoEMean_DNN_K2  --report_freq 5


# K4
~/anaconda3/bin/python3 main.py \
    --device cuda:1 --net meanMoE --expert dnn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log_analysis\
    --dataset frappe --nfeat 5500 --nfield 10 \
    --moe_hid_layer_len 32 --data_nemb 10  \
    --dropout 0.3   \
    --K 4 \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --seed 2023\
    --train_dir MoEMean_DNN_K4  --report_freq 5


# K8
~/anaconda3/bin/python3 main.py \
    --device cuda:2 --net meanMoE --expert dnn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log_analysis\
    --dataset frappe --nfeat 5500 --nfield 10 \
    --moe_hid_layer_len 32 --data_nemb 10  \
    --dropout 0.3   \
    --K 8 \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --seed 2023\
    --train_dir MoEMean_DNN_K8  --report_freq 5



# K32
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net meanMoE --expert dnn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log_analysis\
    --dataset frappe --nfeat 5500 --nfield 10 \
    --moe_hid_layer_len 32 --data_nemb 10  \
    --dropout 0.2   \
    --K 32 \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --seed 2023\
    --train_dir MoEMean_DNN_K32  --report_freq 5


# K64
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net meanMoE --expert dnn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log_analysis\
    --dataset frappe --nfeat 5500 --nfield 10 \
    --moe_hid_layer_len 32 --data_nemb 10  \
    --dropout 0.2   \
    --K 64 \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --seed 2023\
    --train_dir MoEMean_DNN_K64  --report_freq 5

# K128
~/anaconda3/bin/python3 main.py \
    --device cuda:1 --net meanMoE --expert dnn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log_analysis\
    --dataset frappe --nfeat 5500 --nfield 10 \
    --moe_hid_layer_len 32 --data_nemb 10  \
    --dropout 0.2   \
    --K 128 \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --seed 2023\
    --train_dir MoEMean_DNN_K128  --report_freq 5


# K256
~/anaconda3/bin/python3 main.py \
    --device cuda:2 --net meanMoE --expert dnn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log_analysis\
    --dataset frappe --nfeat 5500 --nfield 10 \
    --moe_hid_layer_len 32 --data_nemb 10  \
    --dropout 0.2   \
    --K 256 \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --seed 2023\
    --train_dir MoEMean_DNN_K256  --report_freq 5