

# DNN K = 2
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net sparseMoE --expert dnn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log_analysis\
    --dataset frappe --nfeat 5500 --nfield 10 \
    --moe_hid_layer_len 32 --data_nemb 10  \
    --dropout 0.4  \
    --K 2 --C 1 \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --seed 2023\
    --train_dir MoESparse_DNN_K2  --report_freq 5



# DNN K = 4
~/anaconda3/bin/python3 main.py \
    --device cuda:1 --net sparseMoE --expert dnn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log_analysis\
    --dataset frappe --nfeat 5500 --nfield 10 \
    --moe_hid_layer_len 32 --data_nemb 10  \
    --dropout 0.4  \
    --K 4 --C 1 \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --seed 2023\
    --train_dir MoESparse_DNN_K4  --report_freq 5




# DNN K = 8
~/anaconda3/bin/python3 main.py \
    --device cuda:2 --net sparseMoE --expert dnn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log_analysis\
    --dataset frappe --nfeat 5500 --nfield 10 \
    --moe_hid_layer_len 32 --data_nemb 10  \
    --dropout 0.4  \
    --K 8 --C 1 \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --seed 2023\
    --train_dir MoESparse_DNN_K8  --report_freq 5



# DNN K = 32
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net sparseMoE --expert dnn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log_analysis\
    --dataset frappe --nfeat 5500 --nfield 10 \
    --moe_hid_layer_len 32 --data_nemb 10  \
    --dropout 0.4  \
    --K 32 --C 1 \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --seed 2023\
    --train_dir MoESparse_DNN_K32  --report_freq 5


# DNN K = 64
~/anaconda3/bin/python3 main.py \
    --device cuda:1 --net sparseMoE --expert dnn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log_analysis\
    --dataset frappe --nfeat 5500 --nfield 10 \
    --moe_hid_layer_len 32 --data_nemb 10  \
    --dropout 0.4  \
    --K 64 --C 1 \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --seed 2023\
    --train_dir MoESparse_DNN_K64  --report_freq 5


# DNN K = 128
~/anaconda3/bin/python3 main.py \
    --device cuda:2 --net sparseMoE --expert dnn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log_analysis\
    --dataset frappe --nfeat 5500 --nfield 10 \
    --moe_hid_layer_len 32 --data_nemb 10  \
    --dropout 0.4  \
    --K 128 --C 1 \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --seed 2023\
    --train_dir MoESparse_DNN_K128  --report_freq 5


# DNN K = 256
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net sparseMoE --expert dnn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log_analysis\
    --dataset frappe --nfeat 5500 --nfield 10 \
    --moe_hid_layer_len 32 --data_nemb 10  \
    --dropout 0.4  \
    --K 256 --C 1 \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --seed 2023\
    --train_dir MoESparse_DNN_K256  --report_freq 5