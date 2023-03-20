

#!/bin/bash

worker_id=0
GPU_NUM=8
worker_each_gpu=9
total_workers=$((worker_each_gpu*GPU_NUM))

for((gpu_id=0; gpu_id < GPU_NUM; ++gpu_id)); do
#  echo "GPU id is $gpu_id"
  for((i=0; i < worker_each_gpu; ++i)); do
#    echo "worker id is $worker_id"
    python exps/main_sigmod/ground_truth/2.seq_train_online.py  \
    --log_name=baseline_train_based \
    --search_space=mlp_sp \
    --num_layers=4 \
    --hidden_choice_len=10 \
    --base_dir=/home/shaofeng/naili/firmest_data/ \
    --num_labels=1 \
    --device=cuda:$gpu_id \
    --batch_size=1024 \
    --lr=0.001 \
    --epoch=10 \
    --iter_per_epoch=2000 \
    --dataset=criteo \
    --nfeat=2100000 \
    --nfield=39 \
    --nemb=10 \
    --worker_id=$worker_id \
    --total_workers=$total_workers \
    --workers=0 \
    --log_folder=LogCriteo &
#    --total_models_per_worker=2 &

    sleep 1
    worker_id=$((worker_id+1))
  done
done


# pkill -9 -f exps/main_sigmod/ground_truth/2.seq_train_online.py
# run with bash exps/main_sigmod/train_all_models_criteo.sh >criteobash &
