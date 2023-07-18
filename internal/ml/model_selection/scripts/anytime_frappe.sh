
# run the 2phase-MS
python internal/ml/model_selection/exps/system/anytime_simulate.py \
      --num_layers 4 \
      --hidden_choice_len 20 \
      --batch_size 128 \
      --nfeat 5500 \
      --nfield 10 \
      --base_dir ../exp_data/ \
      --dataset frappe \
      --num_labels 2 \
      --only_phase1 False \
      --is_simulate True \
      --device cpu \
      --log_folder any_time_frappe \
      --result_dir ./internal/ml/model_selection/exps/result/


# run the training-free MS
python internal/ml/model_selection/exps/system/anytime_simulate.py \
      --num_layers 4 \
      --hidden_choice_len 20 \
      --batch_size 128 \
      --nfeat 5500 \
      --nfield 10 \
      --base_dir ../exp_data/ \
      --dataset frappe \
      --num_labels 2 \
      --only_phase1 True \
      --is_simulate True \
      --device cpu \
      --log_folder any_time_frappe \
      --result_dir ./internal/ml/model_selection/exps/result/


