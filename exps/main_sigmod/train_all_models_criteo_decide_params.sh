


# default setting.
python exps/main_sigmod/ground_truth/0.train_one_model.py  \
  --log_name=baseline_train_based \
  --search_space=mlp_sp \
  --base_dir=/home/naili/firmest_data/ \
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
  --log_folder=LogCriteoTrainTune >criteo_5.log &


python exps/main_sigmod/ground_truth/0.train_one_model.py  \
  --log_name=baseline_train_based \
  --search_space=mlp_sp \
  --base_dir=/home/naili/firmest_data/ \
  --num_labels=1 \
  --device=cuda:0 \
  --batch_size=1024 \
  --lr=0.001 \
  --epoch=10 \
  --iter_per_epoch=2000 \
  --dataset=criteo \
  --nfeat=2100000 \
  --nfield=39 \
  --nemb=10 \
  --workers=0 \
  --log_folder=LogCriteoTrainTune >criteo_10.log &



python exps/main_sigmod/ground_truth/0.train_one_model.py  \
  --log_name=baseline_train_based \
  --search_space=mlp_sp \
  --base_dir=/home/naili/firmest_data/ \
  --num_labels=1 \
  --device=cuda:1 \
  --batch_size=1024 \
  --lr=0.001 \
  --epoch=20 \
  --iter_per_epoch=2000 \
  --dataset=criteo \
  --nfeat=2100000 \
  --nfield=39 \
  --nemb=10 \
  --workers=0 \
  --log_folder=LogCriteoTrainTune >criteo_20.log &




python exps/main_sigmod/ground_truth/0.train_one_model.py  \
  --log_name=baseline_train_based \
  --search_space=mlp_sp \
  --base_dir=/home/naili/firmest_data/ \
  --num_labels=1 \
  --device=cuda:2 \
  --batch_size=1024 \
  --lr=0.001 \
  --epoch=40 \
  --iter_per_epoch=2000 \
  --dataset=criteo \
  --nfeat=2100000 \
  --nfield=39 \
  --nemb=10 \
  --workers=0 \
  --log_folder=LogCriteoTrainTune >criteo_40.log &



python exps/main_sigmod/ground_truth/0.train_one_model.py  \
  --log_name=baseline_train_based \
  --search_space=mlp_sp \
  --base_dir=/home/naili/firmest_data/ \
  --num_labels=1 \
  --device=cuda:3 \
  --batch_size=1024 \
  --lr=0.001 \
  --epoch=60 \
  --iter_per_epoch=2000 \
  --dataset=criteo \
  --nfeat=2100000 \
  --nfield=39 \
  --nemb=10 \
  --workers=0 \
  --log_folder=LogCriteoTrainTune >criteo_60.log &



python exps/main_sigmod/ground_truth/0.train_one_model.py  \
  --log_name=baseline_train_based \
  --search_space=mlp_sp \
  --base_dir=/home/naili/firmest_data/ \
  --num_labels=1 \
  --device=cuda:4 \
  --batch_size=1024 \
  --lr=0.001 \
  --epoch=80 \
  --iter_per_epoch=2000 \
  --dataset=criteo \
  --nfeat=2100000 \
  --nfield=39 \
  --nemb=10 \
  --workers=0 \
  --log_folder=LogCriteoTrainTune >criteo_80.log &



python exps/main_sigmod/ground_truth/0.train_one_model.py  \
  --log_name=baseline_train_based \
  --search_space=mlp_sp \
  --base_dir=/home/naili/firmest_data/ \
  --num_labels=1 \
  --device=cuda:5 \
  --batch_size=1024 \
  --lr=0.001 \
  --epoch=100 \
  --iter_per_epoch=2000 \
  --dataset=criteo \
  --nfeat=2100000 \
  --nfield=39 \
  --nemb=10 \
  --workers=0 \
  --log_folder=LogCriteoTrainTune >criteo_100.log &

