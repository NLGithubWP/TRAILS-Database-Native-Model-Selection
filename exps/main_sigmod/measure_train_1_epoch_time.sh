

nohup  python exps/main_sigmod/ground_truth/0.train_one_model.py  \
  --log_name=baseline_train_based \
  --search_space=mlp_sp \
  --num_layers=4 \
  --hidden_choice_len=20 \
  --base_dir=../firmest_data/ \
  --num_labels=1 \
  --device=cuda:0 \
  --batch_size=512 \
  --lr=0.001 \
  --epoch=5 \
  --iter_per_epoch=200 \
  --dataset=frappe \
  --nfeat=5500 \
  --nfield=10 \
  --nemb=10 \
  --workers=0 \
  --log_folder=LogFrappetrainTime >frappeTrainTIme.log &



nohup  python exps/main_sigmod/ground_truth/0.train_one_model.py  \
  --log_name=baseline_train_based \
  --search_space=mlp_sp \
  --num_layers=4 \
  --hidden_choice_len=20 \
  --base_dir=../firmest_data/ \
  --num_labels=1 \
  --device=cuda:0 \
  --batch_size=1024 \
  --lr=0.001 \
  --epoch=3 \
  --iter_per_epoch=200 \
  --dataset=uci_diabetes \
  --nfeat=369 \
  --nfield=43 \
  --nemb=10 \
  --workers=0 \
  --log_folder=LogUCITrainTime >uciTtrainTime.log &




nohup  python exps/main_sigmod/ground_truth/0.train_one_model.py  \
  --log_name=baseline_train_based \
  --search_space=mlp_sp \
  --num_layers=4 \
  --hidden_choice_len=20 \
  --base_dir=../firmest_data/ \
  --num_labels=1 \
  --device=cuda:0 \
  --batch_size=1024 \
  --lr=0.001 \
  --epoch=5 \
  --iter_per_epoch=2000 \
  --dataset=criteo \
  --nfeat=2100000 \
  --nfield=39 \
  --nemb=10 \
  --workers=0 \
  --log_folder=LogCriteoTrainTime >uci_5.log &