



# cifar10

# weight initialization

# kaiming
# 201, bn
nohup python main/1_benchmark_metrics/benchmark_metrics.py --metrics_result "201_15k_c10_bs32_ic16_BN_kaiming.json" --num_arch 15625 --base_dir "./data" --batch_size 32 --init_channels 16 --init_w_type kaiming --dataset cifar10 --search_space nasbench201 --api_loc  NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:3  &
# 201, no-bn
nohup python main/1_benchmark_metrics/benchmark_metrics.py --metrics_result "201_15k_c10_bs32_ic16_noBN_kaiming.json" --num_arch 15625 --base_dir "./data" --batch_size 32 --init_channels 16 --init_w_type kaiming --dataset cifar10 --search_space nasbench201 --api_loc  NAS-Bench-201-v1_0-e61699.pth --evaluation synflow --device cuda:4 --bn 0  &

# zero
# 201, bn
nohup python main/1_benchmark_metrics/benchmark_metrics.py --metrics_result "201_15k_c10_bs32_ic16_BN_xavier.json" --num_arch 15625 --base_dir "./data" --batch_size 32 --init_channels 16 --init_w_type xavier --dataset cifar10 --search_space nasbench201 --api_loc  NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:5  &
# 201, no-bn
nohup python main/1_benchmark_metrics/benchmark_metrics.py --metrics_result "201_15k_c10_bs32_ic16_noBN_xavier.json" --num_arch 15625 --base_dir "./data" --batch_size 32 --init_channels 16 --init_w_type xavier --dataset cifar10 --search_space nasbench201 --api_loc  NAS-Bench-201-v1_0-e61699.pth --evaluation synflow --device cuda:6 --bn 0  &

# xavier
# 201, bn
nohup python main/1_benchmark_metrics/benchmark_metrics.py --metrics_result "201_15k_c10_bs32_ic16_BN_zero.json" --num_arch 15625 --base_dir "./data" --batch_size 32 --init_channels 16 --init_w_type zero --dataset cifar10 --search_space nasbench201 --api_loc  NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:7  &
# 201, no-bn
nohup python main/1_benchmark_metrics/benchmark_metrics.py --metrics_result "201_15k_c10_bs32_ic16_noBN_zero.json" --num_arch 15625 --base_dir "./data" --batch_size 32 --init_channels 16 --init_w_type zero --dataset cifar10 --search_space nasbench201 --api_loc  NAS-Bench-201-v1_0-e61699.pth --evaluation synflow --device cuda:1 --bn 0  &


