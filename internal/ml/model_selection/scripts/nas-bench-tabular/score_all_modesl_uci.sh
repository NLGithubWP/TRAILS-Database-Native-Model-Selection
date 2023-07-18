

export PYTHONPATH=$PYTHONPATH:./internal/ml/model_selection

nohup  python ./internal/ml/model_selection/exps/nas_bench_tabular/4.seq_score_online.py \
  --log_name=score_based_all_metrics \
  --search_space=mlp_sp \
  --num_layers=4 \
  --hidden_choice_len=20 \
  --base_dir=../firmest_data/ \
  --num_labels=2 \
  --device=cuda:2 \
  --batch_size=32 \
  --dataset=uci_diabetes \
  --nfeat=369 \
  --nfield=43 \
  --nemb=10 \
  --workers=0 \
  --log_folder=LogMeasureStoreTime  > outputUciScoreALl.log&



