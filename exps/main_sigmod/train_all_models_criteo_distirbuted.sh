

# frappe
python exps/main_sigmod/ground_truth/2.seq_train_dist_online.py  \
    --log_name=baseline_train_based \
    --search_space=mlp_sp \
    --base_dir=/Users/kevin/project_python/firmest_data/ \
    --num_labels=1 \
    --device=cpu \
    --batch_size=512 \
    --lr=0.001 \
    --epoch=1 \
    --iter_per_epoch=2 \
    --dataset=frappe \
    --nfeat=5500 \
    --nfield=10 \
    --nemb=10 \
    --total_models_per_worker=2 \
    --worker_each_gpu=1 \
    --gpu_num=2

# criteo
python exps/main_sigmod/ground_truth/2.seq_train_dist_online.py  \
    --log_name=baseline_train_based \
    --search_space=mlp_sp \
    --base_dir=/home/naili/firmest_data/ \
    --num_labels=1 \
    --device=gpu \
    --batch_size=512 \
    --lr=0.001 \
    --epoch=100 \
    --iter_per_epoch=200 \
    --dataset=criteo \
    --nfeat=2100000 \
    --nfield=39 \
    --nemb=10 \
    --total_models_per_worker=2 \
    --worker_each_gpu=5 \
    --gpu_num=8 \
    --log_folder=LogCriteo