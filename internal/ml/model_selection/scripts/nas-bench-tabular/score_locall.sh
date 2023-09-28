

export PYTHONPATH=$PYTHONPATH:./internal/ml/model_selection
python ./internal/ml/model_selection/exps/nas_bench_tabular/4.seq_score_online.py \
  --embedding_cache_filtering=True \
  --models_explore=9999 \
  --tfmem=express_flow \
  --log_name=score_based \
  --search_space=mlp_sp \
  --num_layers=4 \
  --hidden_choice_len=10 \
  --base_dir=../exp_data/ \
  --num_labels=2 \
  --device=cpu \
  --batch_size=32 \
  --dataset=criteo \
  --nfeat=2100000 \
  --nfield=39 \
  --nemb=10 \
  --workers=0 \
  --result_dir=./internal/ml/model_selection/exp_result/


python ./internal/ml/model_selection/exps/nas_bench_tabular/4.seq_score_online.py \
  --embedding_cache_filtering=True \
  --models_explore=159999 \
  --tfmem=express_flow \
  --log_name=score_based \
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
  --result_dir=./internal/ml/model_selection/exp_result/


python ./internal/ml/model_selection/exps/nas_bench_tabular/4.seq_score_online.py \
  --embedding_cache_filtering=True \
  --models_explore=159999 \
  --tfmem=express_flow \
  --log_name=score_based \
  --search_space=mlp_sp \
  --num_layers=4 \
  --hidden_choice_len=20 \
  --base_dir=../exp_data/ \
  --num_labels=2 \
  --device=cpu \
  --batch_size=32 \
  --dataset=uci_diabetes \
  --nfeat=369 \
  --nfield=43 \
  --nemb=10 \
  --workers=0 \
  --result_dir=./internal/ml/model_selection/exp_result/

