



# 201, bs128, channel size
sleep 2
#nohup python main/1_benchmark_metrics/benchmark_metrics.py --metrics_result "201_15625_c10_bs128_ic8_BN.json" --num_arch 15625 --base_dir "./data" --batch_size 128 --init_channels 8 --dataset cifar10 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:1 &
sleep 2
#nohup python main/1_benchmark_metrics/benchmark_metrics.py --metrics_result "201_15625_c10_bs128_ic16_BN.json" --num_arch 15625 --base_dir "./data" --batch_size 128 --init_channels 16 --dataset cifar10 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:2 &
sleep 2
#nohup python main/1_benchmark_metrics/benchmark_metrics.py --metrics_result "201_15625_c10_bs128_ic32_BN.json" --num_arch 15625 --base_dir "./data" --batch_size 128 --init_channels 32 --dataset cifar10 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:3 &
sleep 2
#nohup python main/1_benchmark_metrics/benchmark_metrics.py --metrics_result "201_15625_c10_bs128_ic64_BN.json" --num_arch 15625 --base_dir "./data" --batch_size 128 --init_channels 64 --dataset cifar10 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:4 &
sleep 2
nohup python main/1_benchmark_metrics/benchmark_metrics.py --metrics_result "201_15625_c10_bs128_ic128_BN.json" --num_arch 15625 --base_dir "./data" --batch_size 128 --init_channels 128 --dataset cifar10 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:1 &

# 201, bs64, channel size
sleep 2
#nohup python main/1_benchmark_metrics/benchmark_metrics.py --metrics_result "201_15625_c10_bs64_ic8_BN.json" --num_arch 15625 --base_dir "./data" --batch_size 64 --init_channels 8 --dataset cifar10 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:2 &
sleep 2
#nohup python main/1_benchmark_metrics/benchmark_metrics.py --metrics_result "201_15625_c10_bs64_ic16_BN.json" --num_arch 15625 --base_dir "./data" --batch_size 64 --init_channels 16 --dataset cifar10 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:5 &
sleep 2
#nohup python main/1_benchmark_metrics/benchmark_metrics.py --metrics_result "201_15625_c10_bs64_ic32_BN.json" --num_arch 15625 --base_dir "./data" --batch_size 64 --init_channels 32 --dataset cifar10 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:7 &
sleep 2
#nohup python main/1_benchmark_metrics/benchmark_metrics.py --metrics_result "201_15625_c10_bs64_ic64_BN.json" --num_arch 15625 --base_dir "./data" --batch_size 64 --init_channels 64 --dataset cifar10 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:0 &
sleep 2
nohup python main/1_benchmark_metrics/benchmark_metrics.py --metrics_result "201_15625_c10_bs64_ic128_BN.json" --num_arch 15625 --base_dir "./data" --batch_size 64 --init_channels 128 --dataset cifar10 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:1 &

# 201, bs32, channel size
sleep 2
#nohup python main/1_benchmark_metrics/benchmark_metrics.py --metrics_result "201_15625_c10_bs32_ic8_BN.json" --num_arch 15625 --base_dir "./data" --batch_size 32 --init_channels 8  --dataset cifar10 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:2 &
sleep 2
#nohup python main/1_benchmark_metrics/benchmark_metrics.py --metrics_result "201_15625_c10_bs32_ic16_BN.json" --num_arch 15625 --base_dir "./data" --batch_size 32 --init_channels 16  --dataset cifar10 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:3 &
sleep 2
#nohup python main/1_benchmark_metrics/benchmark_metrics.py --metrics_result "201_15625_c10_bs32_ic32_BN.json" --num_arch 15625 --base_dir "./data" --batch_size 32 --init_channels 32  --dataset cifar10 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:4 &
sleep 2
#nohup python main/1_benchmark_metrics/benchmark_metrics.py --metrics_result "201_15625_c10_bs32_ic64_BN.json" --num_arch 15625 --base_dir "./data" --batch_size 32 --init_channels 64  --dataset cifar10 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:5 &
sleep 2
nohup python main/1_benchmark_metrics/benchmark_metrics.py --metrics_result "201_15625_c10_bs32_ic128_BN.json" --num_arch 15625 --base_dir "./data" --batch_size 32 --init_channels 128  --dataset cifar10 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:6 &

# 201, bs16, channel size
sleep 2
#nohup python main/1_benchmark_metrics/benchmark_metrics.py --metrics_result "201_15625_c10_bs16_ic8_BN.json" --num_arch 15625 --base_dir "./data" --batch_size 16 --init_channels 8 --dataset cifar10 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:7 &
sleep 2
#nohup python main/1_benchmark_metrics/benchmark_metrics.py --metrics_result "201_15625_c10_bs16_ic16_BN.json" --num_arch 15625 --base_dir "./data" --batch_size 16 --init_channels 16 --dataset cifar10 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:5 &
sleep 2
#nohup python main/1_benchmark_metrics/benchmark_metrics.py --metrics_result "201_15625_c10_bs16_ic32_BN.json" --num_arch 15625 --base_dir "./data" --batch_size 16 --init_channels 32 --dataset cifar10 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:6 &
sleep 2
#nohup python main/1_benchmark_metrics/benchmark_metrics.py --metrics_result "201_15625_c10_bs16_ic64_BN.json" --num_arch 15625 --base_dir "./data" --batch_size 16 --init_channels 64 --dataset cifar10 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:7 &
sleep 2
nohup python main/1_benchmark_metrics/benchmark_metrics.py --metrics_result "201_15625_c10_bs16_ic128_BN.json" --num_arch 15625 --base_dir "./data" --batch_size 16 --init_channels 128 --dataset cifar10 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:0 &

# 201, bs8, channel size
sleep 2
#nohup python main/1_benchmark_metrics/benchmark_metrics.py --metrics_result "201_15625_c10_bs8_ic8_BN.json" --num_arch 15625 --base_dir "./data" --batch_size 8 --init_channels 8 --dataset cifar10 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:1 &
sleep 2
#nohup python main/1_benchmark_metrics/benchmark_metrics.py --metrics_result "201_15625_c10_bs8_ic16_BN.json" --num_arch 15625 --base_dir "./data" --batch_size 8 --init_channels 16 --dataset cifar10 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:2 &
sleep 2
#nohup python main/1_benchmark_metrics/benchmark_metrics.py --metrics_result "201_15625_c10_bs8_ic32_BN.json" --num_arch 15625 --base_dir "./data" --batch_size 8 --init_channels 32 --dataset cifar10 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:3 &
sleep 2
#nohup python main/1_benchmark_metrics/benchmark_metrics.py --metrics_result "201_15625_c10_bs8_ic64_BN.json" --num_arch 15625 --base_dir "./data" --batch_size 8 --init_channels 64 --dataset cifar10 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:4 &
sleep 2
nohup python main/1_benchmark_metrics/benchmark_metrics.py --metrics_result "201_15625_c10_bs8_ic128_BN.json" --num_arch 15625 --base_dir "./data" --batch_size 8 --init_channels 128 --dataset cifar10 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:0 &
