



nohup python main/4_system/model_evaluation.py --metrics_result score_c10_201.json --num_run 500 --num_arch 300 --base_dir "/home/author/Fast-AutoNAS/data" --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --device cuda:0 &

nohup python main/4_system/model_evaluation.py --metrics_result score_c10_101.json --num_run 500 --num_arch 300 --base_dir "/home/author/Fast-AutoNAS/data" --search_space nasbench101 --api_loc nasbench_only108.pkl --device cuda:1 &


# 101
nohup python src/apiserver/serach_strategy.py --controller_port 8006 --log_name SS_1wk_500run_NB101_c10_upto93.log --run 500 --save_file_latency 1wk_500run_NB101_c10_upto93latency --save_file_all 1wk_500run_NB101_c10_upto93all --search_space nasbench101 --dataset cifar10 --num_labels 10 --api_loc nasbench_only108.pkl > ss_101_newupto93 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --worker_save_file wk_time_usage_upto93101 --controller_url http://0.0.0.0:8006 --dataset cifar10 --num_labels 10 --search_space nasbench101 --device cuda:6 > w1101_cmdline_newupto93 &
#nohup python main/4_system/0_model_selection/model_evaluation_online.py --controller_url http://0.0.0.0:8004 --dataset cifar10 --num_labels 10 --search_space nasbench101 --device cpu > w1101_cmdline &


# 201
nohup python src/apiserver/serach_strategy.py --controller_port 8007 --log_name SS_1wk_500run_NB201_c10_upto93.log --run 500 --save_file_latency 1wk_500run_NB201_c10_upto93latency --save_file_all 1wk_500run_NB201_c10_upto93all --search_space nasbench201 --dataset cifar10 --num_labels 10 --api_loc NAS-Bench-201-v1_1-096897.pth > ss_201_newupto93 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --worker_save_file wk_time_usage_upto93201 --controller_url http://0.0.0.0:8007 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:7 > w1201_cmdline_newupto93 &

# 201 distributed setting

# 1 wkr
nohup python src/apiserver/serach_strategy.py --controller_port 8011 --log_name SS_1wk_500run_NB201_c10.log --run 500 --save_file_latency 1wk_500run_NB201_c10_latency --save_file_all 1wk_500run_NB201_c10_all --search_space nasbench201 --dataset cifar10 --num_labels 10 --api_loc NAS-Bench-201-v1_1-096897.pth &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 1wk_log_id1.log --worker_save_file 1wk_time_usage --controller_url http://0.0.0.0:8011 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:1 &

# 2 wkr
nohup python src/apiserver/serach_strategy.py --controller_port 8009 --log_name SS_2wk_500run_NB201_c10.log --run 500 --save_file_latency 2wk_500run_NB201_c10_latency --save_file_all 2wk_500run_NB201_c10_all --search_space nasbench201 --dataset cifar10 --num_labels 10 --api_loc NAS-Bench-201-v1_1-096897.pth > ss_201_2wks.conslue &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 2wk_log_id1.log --worker_save_file 2wk_time_usage_id1.res --controller_url http://0.0.0.0:8009 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:6 > 2wk201_id1.consule &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 2wk_log_id2.log --worker_save_file 2wk_time_usage_id2.res --controller_url http://0.0.0.0:8009 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:7 > 2wk201_id2.consule &

# 4 wkr
nohup python src/apiserver/serach_strategy.py --controller_port 8010 --log_name SS_4wk_500run_NB201_c10.log --run 500 --save_file_latency 4wk_500run_NB201_c10_latency --save_file_all 4wk_500run_NB201_c10_all --search_space nasbench201 --dataset cifar10 --num_labels 10 --api_loc NAS-Bench-201-v1_1-096897.pth > ss_201_4wks.conslue &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 4wk_log_id1.log --worker_save_file 4wk_time_usage_id1.res --controller_url http://0.0.0.0:8010 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:1 > 4wk201_idX.consule &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 4wk_log_id2.log --worker_save_file 4wk_time_usage_id2.res --controller_url http://0.0.0.0:8010 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:2 > 4wk201_idX.consule &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 4wk_log_id3.log --worker_save_file 4wk_time_usage_id3.res --controller_url http://0.0.0.0:8010 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:3 > 4wk201_idX.consule &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 4wk_log_id4.log --worker_save_file 4wk_time_usage_id4.res --controller_url http://0.0.0.0:8010 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:4 > 4wk201_idX.consule &

# 8 wkr
nohup python src/apiserver/serach_strategy.py --controller_port 8012 --log_name SS_8wk_500run_NB201_c10.log --run 500 --save_file_latency 8wk_500run_NB201_c10_latency --save_file_all 8wk_500run_NB201_c10_all --search_space nasbench201 --dataset cifar10 --num_labels 10 --api_loc NAS-Bench-201-v1_1-096897.pth &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 8wk_log_id1.log --worker_save_file 8wk_time_usage_id1.res --controller_url http://0.0.0.0:8012 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:0 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 8wk_log_id2.log --worker_save_file 8wk_time_usage_id2.res --controller_url http://0.0.0.0:8012 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:1 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 8wk_log_id3.log --worker_save_file 8wk_time_usage_id3.res --controller_url http://0.0.0.0:8012 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:2 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 8wk_log_id4.log --worker_save_file 8wk_time_usage_id4.res --controller_url http://0.0.0.0:8012 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:3 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 8wk_log_id5.log --worker_save_file 8wk_time_usage_id5.res --controller_url http://0.0.0.0:8012 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:4 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 8wk_log_id6.log --worker_save_file 8wk_time_usage_id6.res --controller_url http://0.0.0.0:8012 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:5 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 8wk_log_id7.log --worker_save_file 8wk_time_usage_id7.res --controller_url http://0.0.0.0:8012 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:6 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 8wk_log_id8.log --worker_save_file 8wk_time_usage_id8.res --controller_url http://0.0.0.0:8012 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:7 &


# those are for throughput
# 12 wkr
nohup python src/apiserver/serach_strategy.py --throughput_run_models 15000 --controller_port 8013 --log_name SS_12wk_TPS_1run_NB201_c10.log --run 1 --save_file_latency 12wk_TPS_1run_NB201_c10_latency --save_file_all 12wk_TPS_1run_NB201_c10_all --search_space nasbench201 --dataset cifar10 --num_labels 10 --api_loc NAS-Bench-201-v1_1-096897.pth &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 12wk_TPS_log_id1.log --worker_save_file 12wk_TPS_time_usage_id1.res --controller_url http://172.28.176.55:8013 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:0 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 12wk_TPS_log_id2.log --worker_save_file 12wk_TPS_time_usage_id2.res --controller_url http://172.28.176.55:8013 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:1 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 12wk_TPS_log_id3.log --worker_save_file 12wk_TPS_time_usage_id3.res --controller_url http://172.28.176.55:8013 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:2 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 12wk_TPS_log_id4.log --worker_save_file 12wk_TPS_time_usage_id4.res --controller_url http://172.28.176.55:8013 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:3 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 12wk_TPS_log_id5.log --worker_save_file 12wk_TPS_time_usage_id5.res --controller_url http://172.28.176.55:8013 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:4 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 12wk_TPS_log_id6.log --worker_save_file 12wk_TPS_time_usage_id6.res --controller_url http://172.28.176.55:8013 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:5 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 12wk_TPS_log_id7.log --worker_save_file 12wk_TPS_time_usage_id7.res --controller_url http://172.28.176.55:8013 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:6 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 12wk_TPS_log_id8.log --worker_save_file 12wk_TPS_time_usage_id8.res --controller_url http://172.28.176.55:8013 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:7 &

nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 12wk_TPS_log_id9.log --worker_save_file 12wk_TPS_time_usage_id9.res --controller_url http://172.28.176.55:8013 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:7 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 12wk_TPS_log_id10.log --worker_save_file 12wk_TPS_time_usage_id10.res --controller_url http://172.28.176.55:8013 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:6 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 12wk_TPS_log_id11.log --worker_save_file 12wk_TPS_time_usage_id11.res --controller_url http://172.28.176.55:8013 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:5 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 12wk_TPS_log_id12.log --worker_save_file 12wk_TPS_time_usage_id12.res --controller_url http://172.28.176.55:8013 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:4 &

# 14 wkr
nohup python src/apiserver/serach_strategy.py --throughput_run_models 15000 --controller_port 8014 --log_name SS_14wk_TPS_1run_NB201_c10.log --run 1 --save_file_latency 14wk_TPS_1run_NB201_c10_latency --save_file_all 14wk_TPS_1run_NB201_c10_all --search_space nasbench201 --dataset cifar10 --num_labels 10 --api_loc NAS-Bench-201-v1_1-096897.pth &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 14wk_TPS_log_id1.log --worker_save_file 14wk_TPS_time_usage_id1.res --controller_url http://172.28.176.55:8014 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:0 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 14wk_TPS_log_id2.log --worker_save_file 14wk_TPS_time_usage_id2.res --controller_url http://172.28.176.55:8014 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:1 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 14wk_TPS_log_id3.log --worker_save_file 14wk_TPS_time_usage_id3.res --controller_url http://172.28.176.55:8014 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:2 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 14wk_TPS_log_id4.log --worker_save_file 14wk_TPS_time_usage_id4.res --controller_url http://172.28.176.55:8014 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:3 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 14wk_TPS_log_id5.log --worker_save_file 14wk_TPS_time_usage_id5.res --controller_url http://172.28.176.55:8014 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:4 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 14wk_TPS_log_id6.log --worker_save_file 14wk_TPS_time_usage_id6.res --controller_url http://172.28.176.55:8014 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:5 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 14wk_TPS_log_id7.log --worker_save_file 14wk_TPS_time_usage_id7.res --controller_url http://172.28.176.55:8014 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:6 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 14wk_TPS_log_id8.log --worker_save_file 14wk_TPS_time_usage_id8.res --controller_url http://172.28.176.55:8014 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:7 &

nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 14wk_TPS_log_id9.log --worker_save_file 14wk_TPS_time_usage_id9.res --controller_url http://172.28.176.55:8014 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:0 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 14wk_TPS_log_id10.log --worker_save_file 14wk_TPS_time_usage_id10.res --controller_url http://172.28.176.55:8014 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:1 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 14wk_TPS_log_id11.log --worker_save_file 14wk_TPS_time_usage_id11.res --controller_url http://172.28.176.55:8014 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:2 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 14wk_TPS_log_id12.log --worker_save_file 14wk_TPS_time_usage_id12.res --controller_url http://172.28.176.55:8014 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:3 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 14wk_TPS_log_id13.log --worker_save_file 14wk_TPS_time_usage_id13.res --controller_url http://172.28.176.55:8014 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:4 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 14wk_TPS_log_id14.log --worker_save_file 14wk_TPS_time_usage_id14.res --controller_url http://172.28.176.55:8014 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:5 &


# 16 wkr
nohup python src/apiserver/serach_strategy.py --throughput_run_models 15000 --controller_port 8015 --log_name SS_16wk_TPS_1run_NB201_c10.log --run 1 --save_file_latency 16wk_TPS_1run_NB201_c10_latency --save_file_all 16wk_TPS_1run_NB201_c10_all --search_space nasbench201 --dataset cifar10 --num_labels 10 --api_loc NAS-Bench-201-v1_1-096897.pth &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 16wk_TPS_log_id1.log --worker_save_file 16wk_TPS_time_usage_id1.res --controller_url http://172.28.176.55:8015 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:0 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 16wk_TPS_log_id2.log --worker_save_file 16wk_TPS_time_usage_id2.res --controller_url http://172.28.176.55:8015 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:1 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 16wk_TPS_log_id3.log --worker_save_file 16wk_TPS_time_usage_id3.res --controller_url http://172.28.176.55:8015 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:2 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 16wk_TPS_log_id4.log --worker_save_file 16wk_TPS_time_usage_id4.res --controller_url http://172.28.176.55:8015 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:3 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 16wk_TPS_log_id5.log --worker_save_file 16wk_TPS_time_usage_id5.res --controller_url http://172.28.176.55:8015 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:4 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 16wk_TPS_log_id6.log --worker_save_file 16wk_TPS_time_usage_id6.res --controller_url http://172.28.176.55:8015 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:5 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 16wk_TPS_log_id7.log --worker_save_file 16wk_TPS_time_usage_id7.res --controller_url http://172.28.176.55:8015 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:6 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 16wk_TPS_log_id8.log --worker_save_file 16wk_TPS_time_usage_id8.res --controller_url http://172.28.176.55:8015 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:7 &

nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 16wk_TPS_log_id9.log --worker_save_file 16wk_TPS_time_usage_id9.res --controller_url http://172.28.176.55:8015 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:0 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 16wk_TPS_log_id10.log --worker_save_file 16wk_TPS_time_usage_id10.res --controller_url http://172.28.176.55:8015 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:1 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 16wk_TPS_log_id11.log --worker_save_file 16wk_TPS_time_usage_id11.res --controller_url http://172.28.176.55:8015 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:2 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 16wk_TPS_log_id12.log --worker_save_file 16wk_TPS_time_usage_id12.res --controller_url http://172.28.176.55:8015 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:3 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 16wk_TPS_log_id13.log --worker_save_file 16wk_TPS_time_usage_id13.res --controller_url http://172.28.176.55:8015 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:4 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 16wk_TPS_log_id14.log --worker_save_file 16wk_TPS_time_usage_id14.res --controller_url http://172.28.176.55:8015 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:5 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 16wk_TPS_log_id15.log --worker_save_file 16wk_TPS_time_usage_id15.res --controller_url http://172.28.176.55:8015 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:6 &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 16wk_TPS_log_id16.log --worker_save_file 16wk_TPS_time_usage_id16.res --controller_url http://172.28.176.55:8015 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cuda:7 &



# local debug
python src/apiserver/serach_strategy.py --controller_port 8006 --log_name SS_1wk_1run_NB101_c10debug.log --run 1 --save_file_latency 1wk_500run_NB101_c10_latencydebug --save_file_all 1wk_500run_NB101_c10_alldebug --search_space nasbench101 --dataset cifar10 --num_labels 10 --api_loc nasbench_only108.pkl
python main/4_system/0_model_selection/model_evaluation_online.py --worker_save_file wk_time_usage_101debug --controller_url http://0.0.0.0:8006 --dataset cifar10 --num_labels 10 --search_space nasbench101 --device cpu


# local dist
nohup python src/apiserver/serach_strategy.py --controller_port 8009 --log_name SS_2wk_500run_NB201_c10.log --run 500 --save_file_latency 2wk_500run_NB201_c10_latency --save_file_all 2wk_500run_NB201_c10_all --search_space nasbench201 --dataset cifar10 --num_labels 10 --api_loc NAS-Bench-201-v1_1-096897.pth > ss_201_2wks.conslue &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 2wk_log_id1.log --worker_save_file 2wk_time_usage_id1.res --controller_url http://0.0.0.0:8009 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cpu > 2wk201_id1.consule &
nohup python main/4_system/0_model_selection/model_evaluation_online.py --log_name 2wk_log_id2.log --worker_save_file 2wk_time_usage_id2.res --controller_url http://0.0.0.0:8009 --dataset cifar10 --num_labels 10 --search_space nasbench201 --device cpu > 2wk201_id2.consule &

