



# cifar10
## 101, bn
#sleep 2
# nohup python main/1_benchmark_metrics/benchmark_metrics.py --pre_trained_result   --metrics_result "101_15k_c10_bs32_ic16_BN.json" --num_arch 15625 --base_dir "./data" --batch_size 32 --init_channels 16 --dataset cifar10 --search_space nasbench101 --api_loc nasbench_only108.pkl --evaluation all_matrix --device cuda:0  &
## 101, no-bn
#sleep 2
# nohup python main/1_benchmark_metrics/benchmark_metrics.py --pre_trained_result   --metrics_result "101_15k_c10_bs32_ic16_noBN.json" --num_arch 15625 --base_dir "./data" --batch_size 32 --init_channels 16 --dataset cifar10 --search_space nasbench101 --api_loc nasbench_only108.pkl --evaluation synflow --device cuda:1  --bn 0  &

# 201, bn
sleep 2
nohup python main/1_benchmark_metrics/benchmark_metrics.py --pre_trained_result "201_15k_c10_bs32_ic16_unionBest.json" --metrics_result "201_15625_c10_bs32_ic16_BN.json" --num_arch 15625 --base_dir "./data" --batch_size 32 --init_channels 16 --dataset cifar10 --search_space nasbench201 --api_loc  NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:0  &
# 201, no-bn
sleep 2
nohup python main/1_benchmark_metrics/benchmark_metrics.py --pre_trained_result "201_15k_c10_bs32_ic16_unionBest.json" --metrics_result "201_15625_c10_bs32_ic16_noBN.json" --num_arch 15625 --base_dir "./data" --batch_size 32 --init_channels 16 --dataset cifar10 --search_space nasbench201 --api_loc  NAS-Bench-201-v1_0-e61699.pth --evaluation synflow --device cuda:1 --bn 0  &


# cifar100
# 101, bn
#sleep 2
# nohup python main/1_benchmark_metrics/benchmark_metrics.py --pre_trained_result   --metrics_result "101_15k_c100_bs32_ic16_BN.json" --num_arch 15625 --base_dir "./data" --batch_size 32 --init_channels 16 --dataset cifar100 --search_space nasbench101 --api_loc nasbench_only108.pkl --evaluation all_matrix --device cuda:4  &
# 101, no-bn
#sleep 2
# nohup python main/1_benchmark_metrics/benchmark_metrics.py --pre_trained_result   --metrics_result "101_15k_c100_bs32_ic16_noBN.json" --num_arch 15625 --base_dir "./data" --batch_size 32 --init_channels 16 --dataset cifar100 --search_space nasbench101 --api_loc nasbench_only108.pkl --evaluation synflow --device cuda:5  --bn 0  &

# 201, bn
sleep 2
nohup python main/1_benchmark_metrics/benchmark_metrics.py --pre_trained_result "201_15k_c100_bs32_ic16_unionBest.json" --metrics_result "201_15625_c100_bs32_ic16_BN.json" --num_arch 15625 --base_dir "./data" --batch_size 32 --init_channels 16 --dataset cifar100 --search_space nasbench201 --api_loc  NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:2  &
# 201, no-bn
sleep 2
nohup python main/1_benchmark_metrics/benchmark_metrics.py --pre_trained_result "201_15k_c100_bs32_ic16_unionBest.json" --metrics_result "201_15625_c100_bs32_ic16_noBN.json" --num_arch 15625 --base_dir "./data" --batch_size 32 --init_channels 16 --dataset cifar100 --search_space nasbench201 --api_loc  NAS-Bench-201-v1_0-e61699.pth --evaluation synflow --device cuda:3 --bn 0  &


#ImageNET
## 101, bn
#sleep 2
# nohup python main/1_benchmark_metrics/benchmark_metrics.py --pre_trained_result   --metrics_result "101_15k_imgNet_bs32_ic16_BN.json" --num_arch 15625 --base_dir "./data" --batch_size 32 --init_channels 16 --dataset ImageNet16-120 --search_space nasbench101 --api_loc nasbench_only108.pkl --evaluation all_matrix --device cuda:1  &
## 101, no-bn
#sleep 2
# nohup python main/1_benchmark_metrics/benchmark_metrics.py --pre_trained_result   --metrics_result "101_15k_imgNet_bs32_ic16_noBN.json" --num_arch 15625 --base_dir "./data" --batch_size 32 --init_channels 16 --dataset ImageNet16-120 --search_space nasbench101 --api_loc nasbench_only108.pkl --evaluation synflow --device cuda:3  --bn 0  &

# 201, bn
sleep 2
nohup python main/1_benchmark_metrics/benchmark_metrics.py --pre_trained_result "201_15k_imgNet_bs32_ic16_unionBest.json" --metrics_result "201_15625_imgNet_bs32_ic16_BN.json" --num_arch 15625 --base_dir "./data" --batch_size 32 --init_channels 16 --dataset ImageNet16-120 --search_space nasbench201 --api_loc  NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:4  &
# 201, no-bn
sleep 2
nohup python main/1_benchmark_metrics/benchmark_metrics.py --pre_trained_result "201_15k_imgNet_bs32_ic16_unionBest.json" --metrics_result "201_15625_imgNet_bs32_ic16_noBN.json" --num_arch 15625 --base_dir "./data" --batch_size 32 --init_channels 16 --dataset ImageNet16-120 --search_space nasbench201 --api_loc  NAS-Bench-201-v1_0-e61699.pth --evaluation synflow --device cuda:5 --bn 0 &
