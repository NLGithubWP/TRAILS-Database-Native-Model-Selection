


# nas bench 101, 5k
#nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "101_15k_c10_32.json" --num_arch 15000 --base_dir "./data" --batch_size 32 --dataset cifar10 --search_space nasbench101 --api_loc nasbench_only108.pkl --evaluation all_matrix --device cuda:4 --arch_size 5 &
nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "101_15k_c10_64.json" --num_arch 15000 --base_dir "./data" --batch_size 64 --dataset cifar10 --search_space nasbench101 --api_loc nasbench_only108.pkl --evaluation all_matrix --device cuda:4  --arch_size 5 &
nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "101_15k_c10_128.json" --num_arch 15000 --base_dir "./data" --batch_size 128 --dataset cifar10 --search_space nasbench101 --api_loc nasbench_only108.pkl --evaluation all_matrix --device cuda:5  --arch_size 5 &
nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "101_15k_c10_256.json" --num_arch 15000 --base_dir "./data" --batch_size 256 --dataset cifar10 --search_space nasbench101 --api_loc nasbench_only108.pkl --evaluation all_matrix --device cuda:6  --arch_size 5 &


# nas bench 201, 15k
nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "201_15k_c10_32.json" --num_arch 15000 --base_dir "./data" --batch_size 32 --dataset cifar10 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:1 --arch_size 1 &
nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "201_15k_c10_64.json" --num_arch 15000 --base_dir "./data" --batch_size 64 --dataset cifar10 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:2 --arch_size 1 &
nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "201_15k_c10_128.json" --num_arch 15000 --base_dir "./data" --batch_size 128 --dataset cifar10 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:3 --arch_size 1 &


# with/without  batch norm
nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "201_15k_c10_128_noBN.json" --num_arch 15000 --base_dir "./data" --batch_size 128 --dataset cifar10 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation synflow --device cuda:0 --arch_size 1 --bn 0 --arch_sampler test &
nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "201_15k_c10_128_noBN.json" --num_arch 15000 --base_dir "./data" --batch_size 128 --dataset cifar10 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation synflow --device cuda:0 --arch_size 1 --bn 0 &
nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "101_15k_c10_128_noBN.json" --num_arch 15000 --base_dir "./data" --batch_size 128 --dataset cifar10 --search_space nasbench101 --api_loc nasbench_only108.pkl --evaluation synflow --device cuda:1  --arch_size 5  --bn 0  &



# only for NTK
nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "101_15k_c10_64_ntk.json" --num_arch 15000 --base_dir "./data" --batch_size 64 --dataset cifar10 --search_space nasbench101 --api_loc nasbench_only108.pkl --evaluation all_matrix --device cuda:4  --arch_size 5 --bn 1 &
nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "101_15k_c10_128_ntk.json" --num_arch 15000 --base_dir "./data" --batch_size 128 --dataset cifar10 --search_space nasbench101 --api_loc nasbench_only108.pkl --evaluation all_matrix --device cuda:5  --arch_size 5 --bn 1 &
nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "101_15k_c10_256_ntk.json" --num_arch 15000 --base_dir "./data" --batch_size 256 --dataset cifar10 --search_space nasbench101 --api_loc nasbench_only108.pkl --evaluation all_matrix --device cuda:6  --arch_size 5 --bn 1 &
nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "101_15k_c10_128_ntk_noBN.json" --num_arch 15000 --base_dir "./data" --batch_size 128 --dataset cifar10 --search_space nasbench101 --api_loc nasbench_only108.pkl --evaluation all_matrix --device cuda:7  --arch_size 5 --bn 0 &

nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "201_15k_c10_64_ntk.json" --num_arch 15000 --base_dir "./data" --batch_size 64 --dataset cifar10 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:1 --arch_size 1 --bn 1 &
nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "201_15k_c10_128_ntk.json" --num_arch 15000 --base_dir "./data" --batch_size 128 --dataset cifar10 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:2 --arch_size 1 --bn 1 &
nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "201_15k_c10_256_ntk.json" --num_arch 15000 --base_dir "./data" --batch_size 256 --dataset cifar10 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:3 --arch_size 1 --bn 1 &
nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "201_15k_c10_128_ntk_noBN.json" --num_arch 15000 --base_dir "./data" --batch_size 128 --dataset cifar10 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:0 --arch_size 1 --bn 0 &



# debug
nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "opt_ntk_1.json" --num_arch 3 --base_dir "./data" --batch_size 256 --dataset cifar10 --search_space nasbench101 --api_loc nasbench_only108.pkl --evaluation ntk_cond_num --device cuda:0  --arch_size 5 --arch_sampler test &


# second time evaluate all metrics

# nas bench 101, 5k
nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "101_15k_c10_128.json" --num_arch 15000 --base_dir "./data" --batch_size 128 --dataset cifar10 --search_space nasbench101 --api_loc nasbench_only108.pkl --evaluation all_matrix --device cuda:0 --arch_sampler random &
# nas bench 201, 15k
nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "201_15k_c10_128.json" --num_arch 15000 --base_dir "./data" --batch_size 128 --dataset cifar10 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:1 --arch_sampler random &


nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "101_30k_c10_128_BN.json" --num_arch 30000 --base_dir "./data" --batch_size 128 --dataset cifar10 --search_space nasbench101 --api_loc nasbench_only108.pkl --evaluation all_matrix --device cuda:0 --arch_sampler random --bn 1 &
nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "101_30k_c10_128_noBN.json" --num_arch 30000 --base_dir "./data" --batch_size 128 --dataset cifar10 --search_space nasbench101 --api_loc nasbench_only108.pkl --evaluation all_matrix --device cuda:1 --arch_sampler random --bn 0 &
nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "101_15k_c10_128_noBN.json" --num_arch 15000 --base_dir "./data" --batch_size 128 --dataset cifar10 --search_space nasbench101 --api_loc nasbench_only108.pkl --evaluation all_matrix --device cuda:2 --arch_sampler random --bn 0 &



# cifar100
nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "101_15k_c100_128_BN.json" --num_arch 15000 --base_dir "./data" --batch_size 128 --dataset cifar100 --search_space nasbench101 --api_loc nasbench_only108.pkl --evaluation all_matrix --device cuda:3 --arch_sampler random --bn 1 &
nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "101_15k_c100_128_noBN.json" --num_arch 15000 --base_dir "./data" --batch_size 128 --dataset cifar100 --search_space nasbench101 --api_loc nasbench_only108.pkl --evaluation all_matrix --device cuda:4 --arch_sampler random --bn 0 &
nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "201_15k_c100_128_BN.json" --num_arch 15000 --base_dir "./data" --batch_size 128 --dataset cifar100 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:5 --arch_sampler random --bn 1 &
nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "201_15k_c100_128_noBN.json" --num_arch 15000 --base_dir "./data" --batch_size 128 --dataset cifar100 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:6 --arch_sampler random --bn 0 &


nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "101_15625_c10_128_BN.json" --num_arch 15625 --base_dir "./data" --batch_size 128 --dataset cifar10 --search_space nasbench101 --api_loc nasbench_only108.pkl --evaluation all_matrix --device cuda:0 --arch_sampler random --bn 1 &
nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "101_15625_c10_128_noBN.json" --num_arch 15625 --base_dir "./data" --batch_size 128 --dataset cifar10 --search_space nasbench101 --api_loc nasbench_only108.pkl --evaluation all_matrix --device cuda:1 --arch_sampler random --bn 0 &
nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "201_15625_c10_128_BN.json" --num_arch 15625 --base_dir "./data" --batch_size 128 --dataset cifar10 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:2 --arch_sampler random --bn 1 &
nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "201_15625_c10_128_noBN.json" --num_arch 15625 --base_dir "./data" --batch_size 128 --dataset cifar10 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:3 --arch_sampler random --bn 0 &

nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "101_15625_c100_128_BN.json" --num_arch 15625 --base_dir "./data" --batch_size 128 --dataset cifar100 --search_space nasbench101 --api_loc nasbench_only108.pkl --evaluation all_matrix --device cuda:4 --arch_sampler random --bn 1 &
nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "101_15625_c100_128_noBN.json" --num_arch 15625 --base_dir "./data" --batch_size 128 --dataset cifar100 --search_space nasbench101 --api_loc nasbench_only108.pkl --evaluation all_matrix --device cuda:5 --arch_sampler random --bn 0 &
nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "201_15625_c100_128_BN.json" --num_arch 15625 --base_dir "./data" --batch_size 128 --dataset cifar100 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:6 --arch_sampler random --bn 1 &
nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "201_15625_c100_128_noBN.json" --num_arch 15625 --base_dir "./data" --batch_size 128 --dataset cifar100 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:7 --arch_sampler random --bn 0 &


