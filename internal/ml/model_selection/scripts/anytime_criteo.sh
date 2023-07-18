


# run the 2phase-MS
python internal/ml/model_selection/exps/system/anytime_simulate.py \
      --num_layers 4 \
      --hidden_choice_len 10 \
      --batch_size 128 \
      --nfeat 2100000 \
      --nfield 39 \
      --base_dir ../exp_data/ \
      --dataset criteo \
      --num_labels 2 \
      --only_phase1 False \
      --is_simulate True \
      --device cpu \
      --log_folder any_time_criteo \
      --result_dir ./internal/ml/model_selection/exps/result/


# run the training-free MS
python internal/ml/model_selection/exps/system/anytime_simulate.py \
      --num_layers 4 \
      --hidden_choice_len 10 \
      --batch_size 128 \
      --nfeat 2100000 \
      --nfield 39 \
      --base_dir ../exp_data/ \
      --dataset criteo \
      --num_labels 2 \
      --only_phase1 True \
      --is_simulate True \
      --device cpu \
      --log_folder any_time_criteo \
      --result_dir ./internal/ml/model_selection/exps/result/


