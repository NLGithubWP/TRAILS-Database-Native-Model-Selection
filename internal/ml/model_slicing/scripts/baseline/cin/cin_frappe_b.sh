

 ~/anaconda3/bin/python3 main.py \
    --device cuda:0  --net cin \
    --data_dir "/hdd1/sams/data/" \
    --exp /hdd1/sams/tensor_log_baseline\
    --dataset frappe --nfeat 5500 --nfield 10 \
    --nhid 16 --data_nemb 10 \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --seed 1 \
    --train_dir "cin_hid16"