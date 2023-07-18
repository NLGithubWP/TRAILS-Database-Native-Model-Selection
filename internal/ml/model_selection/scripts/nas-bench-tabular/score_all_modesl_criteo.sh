

export PYTHONPATH=$PYTHONPATH:./internal/ml/model_selection

nohup  python exps/main_sigmod/ground_truth/4.seq_score_online.py \
  --log_name=score_based \
  --search_space=mlp_sp \
  --num_layers=4 \
  --hidden_choice_len=10 \
  --base_dir=../firmest_data/ \
  --num_labels=2 \
  --device=cuda:6 \
  --batch_size=32 \
  --dataset=criteo \
  --nfeat=2100000 \
  --nfield=39 \
  --nemb=10 \
  --workers=0 \
  --log_folder=LogMeasureStoreTime  > outputCriScorAll.log&



