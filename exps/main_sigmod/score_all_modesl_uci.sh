


nohup  python exps/main_sigmod/ground_truth/4.seq_score_online.py \
  --log_name=score_based \
  --search_space=mlp_sp \
  --num_layers=4 \
  --hidden_choice_len=20 \
  --base_dir=../firmest_data/ \
  --num_labels=1 \
  --device=cuda:1 \
  --batch_size=32 \
  --dataset=uci_diabetes \
  --nfeat=369 \
  --nfield=43 \
  --nemb=10 \
  --workers=0 \
  --log_folder=LogUCIScore  > output.log&



