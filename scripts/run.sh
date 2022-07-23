


# nas bench 101, 5k
#nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "101_15k_c10_32.json" --num_arch 15000 --base_dir "./data" --batch_size 32 --dataset cifar10 --search_space nasbench101 --api_loc nasbench_only108.pkl --evaluation all_matrix --device cuda:4 --arch_size 5 &

nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "101_15k_c10_64.json" --num_arch 15000 --base_dir "./data" --batch_size 64 --dataset cifar10 --search_space nasbench101 --api_loc nasbench_only108.pkl --evaluation all_matrix --device cuda:4  --arch_size 5 &
nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "101_15k_c10_128.json" --num_arch 15000 --base_dir "./data" --batch_size 128 --dataset cifar10 --search_space nasbench101 --api_loc nasbench_only108.pkl --evaluation all_matrix --device cuda:5  --arch_size 5 &
nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "101_15k_c10_256.json" --num_arch 15000 --base_dir "./data" --batch_size 256 --dataset cifar10 --search_space nasbench101 --api_loc nasbench_only108.pkl --evaluation all_matrix --device cuda:6  --arch_size 5 &


# nas bench 201, 15k
nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "201_15k_c10_32.json" --num_arch 15000 --base_dir "./data" --batch_size 32 --dataset cifar10 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:1 --arch_size 1 &
nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "201_15k_c10_64.json" --num_arch 15000 --base_dir "./data" --batch_size 64 --dataset cifar10 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:2 --arch_size 1 &
nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "201_15k_c10_128.json" --num_arch 15000 --base_dir "./data" --batch_size 128 --dataset cifar10 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation all_matrix --device cuda:3 --arch_size 1 &



nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "201_15k_c10_128_noBN.json" --num_arch 15000 --base_dir "./data" --batch_size 128 --dataset cifar10 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation synflow --device cuda:0 --arch_size 1 --bn 0 --arch_sampler test &

nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "201_15k_c10_128_noBN.json" --num_arch 15000 --base_dir "./data" --batch_size 128 --dataset cifar10 --search_space nasbench201 --api_loc NAS-Bench-201-v1_0-e61699.pth --evaluation synflow --device cuda:0 --arch_size 1 --bn 0 &

nohup /home/naili/miniconda3/envs/fastautonas/bin/python3.8 main/benchmark.py --log_name "101_15k_c10_128_noBN.json" --num_arch 15000 --base_dir "./data" --batch_size 128 --dataset cifar10 --search_space nasbench101 --api_loc nasbench_only108.pkl --evaluation synflow --device cuda:1  --arch_size 5  --bn 0  &





