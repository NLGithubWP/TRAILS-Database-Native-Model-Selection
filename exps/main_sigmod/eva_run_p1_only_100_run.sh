

nohup  python exps/main_sigmod/system/simulate.py \
  --log_name=score_based \
  --search_space=mlp_sp \
  --num_layers=4 \
  --hidden_choice_len=20 \
  --base_dir=../firmest_data/ \
  --num_labels=1 \
  --device=cuda:5 \
  --batch_size=32 \
  --dataset=frappe \
  --nfeat=5500 \
  --nfield=10 \
  --nemb=10 \
  --workers=0 \
  --log_folder=LogP1Frappe  > p1output.log&