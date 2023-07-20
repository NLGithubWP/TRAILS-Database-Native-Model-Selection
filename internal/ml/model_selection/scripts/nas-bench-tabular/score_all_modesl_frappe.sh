
export PYTHONPATH=$PYTHONPATH:./internal/ml/model_selection

nohup  python ./internal/ml/model_selection/exps/nas_bench_tabular/4.seq_score_online.py \
  --log_name=score_based_all_metrics \
  --search_space=mlp_sp \
  --num_layers=4 \
  --hidden_choice_len=20 \
  --base_dir=../exp_data/ \
  --num_labels=2 \
  --device=cpu \
  --batch_size=32 \
  --dataset=frappe \
  --nfeat=5500 \
  --nfield=10 \
  --nemb=10 \
  --workers=0 \
  --result_dir=./internal/ml/model_selection/exp_result/ \
  --log_folder=log_score_time_frappe  > output.log&



