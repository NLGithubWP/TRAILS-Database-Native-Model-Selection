

export PYTHONPATH=$PYTHONPATH:./internal/ml/model_selection


# ntk_cond_num
# run the 2phase-MS
python internal/ml/model_selection/exps/macro/anytime_simulate.py \
      --tfmem=ntk_cond_num \
      --search_space mlp_sp \
      --num_layers 4 \
      --hidden_choice_len 20 \
      --batch_size 128 \
      --nfeat 5500 \
      --nfield 10 \
      --base_dir=/hdd1/xingnaili/exp_data/ \
      --dataset frappe \
      --num_labels 2 \
      --only_phase1 False \
      --is_simulate True \
      --device cpu \
      --log_folder any_time_frappe \
      --result_dir ./internal/ml/model_selection/exp_result/ \
      --num_points 10

# run the training-free MS
python internal/ml/model_selection/exps/macro/anytime_simulate.py \
      --tfmem=ntk_cond_num \
      --search_space mlp_sp \
      --num_layers 4 \
      --hidden_choice_len 20 \
      --batch_size 128 \
      --nfeat 5500 \
      --nfield 10 \
      --base_dir=/hdd1/xingnaili/exp_data/ \
      --dataset frappe \
      --num_labels 2 \
      --only_phase1 True \
      --is_simulate True \
      --device cpu \
      --log_folder any_time_frappe \
      --result_dir ./internal/ml/model_selection/exp_result/ \
      --num_points 10




# grasp
# run the 2phase-MS
python internal/ml/model_selection/exps/macro/anytime_simulate.py \
      --tfmem=grasp \
      --search_space mlp_sp \
      --num_layers 4 \
      --hidden_choice_len 20 \
      --batch_size 128 \
      --nfeat 5500 \
      --nfield 10 \
      --base_dir=/hdd1/xingnaili/exp_data/ \
      --dataset frappe \
      --num_labels 2 \
      --only_phase1 False \
      --is_simulate True \
      --device cpu \
      --log_folder any_time_frappe \
      --result_dir ./internal/ml/model_selection/exp_result/ \
      --num_points 10

# run the training-free MS
python internal/ml/model_selection/exps/macro/anytime_simulate.py \
      --tfmem=grasp \
      --search_space mlp_sp \
      --num_layers 4 \
      --hidden_choice_len 20 \
      --batch_size 128 \
      --nfeat 5500 \
      --nfield 10 \
      --base_dir=/hdd1/xingnaili/exp_data/ \
      --dataset frappe \
      --num_labels 2 \
      --only_phase1 True \
      --is_simulate True \
      --device cpu \
      --log_folder any_time_frappe \
      --result_dir ./internal/ml/model_selection/exp_result/ \
      --num_points 10

