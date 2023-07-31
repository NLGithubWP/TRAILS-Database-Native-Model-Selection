

export PYTHONPATH=$PYTHONPATH:./internal/ml/model_selection

# frappe
nohup  python3 ./internal/ml/model_selection/exps/nas_bench_tabular/benchmark_filtering_latency.py \
  --tfmem=express_flow \
  --models_explore=10 \
  --log_name=score_based \
  --search_space=mlp_sp \
  --num_layers=4 \
  --hidden_choice_len=20 \
  --base_dir=/hdd1/xingnaili/exp_data/ \
  --num_labels=2 \
  --device=cuda:0 \
  --batch_size=32 \
  --dataset=frappe \
  --nfeat=5500 \
  --nfield=10 \
  --nemb=10 \
  --workers=0 \
  --result_dir=./internal/ml/model_selection/exp_result/ \
  --log_folder=log_score_time_frappe  > output.log&

#critep
nohup  python3 ./internal/ml/model_selection/exps/nas_bench_tabular/benchmark_filtering_latency.py \
  --tfmem=express_flow \
  --models_explore=10 \
  --log_name=score_based \
  --search_space=mlp_sp \
  --num_layers=4 \
  --hidden_choice_len=10 \
  --base_dir=/hdd1/xingnaili/exp_data/ \
  --num_labels=2 \
  --device=cuda:0 \
  --batch_size=32 \
  --dataset=criteo \
  --nfeat=2100000 \
  --nfield=39 \
  --nemb=10 \
  --workers=0 \
  --result_dir=./internal/ml/model_selection/exp_result/ \
  --log_folder=log_score_time_frappe  > output.log&

# uci
nohup  python3 ./internal/ml/model_selection/exps/nas_bench_tabular/benchmark_filtering_latency.py \
  --tfmem=express_flow \
  --models_explore=10 \
  --log_name=score_based \
  --search_space=mlp_sp \
  --num_layers=4 \
  --hidden_choice_len=20 \
  --base_dir=/hdd1/xingnaili/exp_data/ \
  --num_labels=2 \
  --device=cuda:0 \
  --batch_size=32 \
  --dataset=uci_diabetes \
  --nfeat=369 \
  --nfield=43 \
  --nemb=10 \
  --workers=0 \
  --result_dir=./internal/ml/model_selection/exp_result/ \
  --log_folder=log_score_time_frappe  > output.log&

