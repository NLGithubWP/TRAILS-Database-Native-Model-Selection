
# cifar10
## 101, bn

nohup python main/3_benchmark_sampler/benchmark_sampling_online.py --arch_sampler random --is_vote 0 --pre_scored_data  "./result_append/CIFAR10_15625/union/101_15k_c10_bs32_ic16_unionBest.json" --search_space nasbench101 --device cuda:0 --total_run 300 --num_arch_each_run 400 &



sleep 10
while true; do
  res=$(ps aux | grep main/3_benchmark_sampler/benchmark_sampling_online.py | wc -l)
  if [[ $res -eq 2 ]]
  then
    echo "sleep 2"
    sleep 2
  else
    echo "break loop"
    break
  fi
done


nohup python main/3_benchmark_sampler/benchmark_sampling_online.py --arch_sampler random --is_vote 1 --pre_scored_data  "./result_append/CIFAR10_15625/union/101_15k_c10_bs32_ic16_unionBest.json" --search_space nasbench101 --device cuda:0 --total_run 300 --num_arch_each_run 400 &

sleep 10
while true; do
  res=$(ps aux | grep main/3_benchmark_sampler/benchmark_sampling_online.py | wc -l)
  if [[ $res -eq 2  ]]
  then
    echo "sleep 2"
    sleep 2
  else
    echo "break loop"
    break
  fi
done

nohup python main/3_benchmark_sampler/benchmark_sampling_online.py --arch_sampler ea --is_vote 0 --pre_scored_data  "./result_append/CIFAR10_15625/union/101_15k_c10_bs32_ic16_unionBest.json" --search_space nasbench101 --device cuda:0 --total_run 300 --num_arch_each_run 400 &

sleep 10
while true; do
  res=$(ps aux | grep main/3_benchmark_sampler/benchmark_sampling_online.py | wc -l)
  if [[ $res -eq 2  ]]
  then
    echo "sleep 2"
    sleep 2
  else
    echo "break loop"
    break
  fi
done

nohup python main/3_benchmark_sampler/benchmark_sampling_online.py --arch_sampler ea --is_vote 1 --pre_scored_data  "./result_append/CIFAR10_15625/union/101_15k_c10_bs32_ic16_unionBest.json" --search_space nasbench101 --device cuda:0 --total_run 300 --num_arch_each_run 400 &

sleep 10
while true; do
  res=$(ps aux | grep main/3_benchmark_sampler/benchmark_sampling_online.py | wc -l)
  if [[ $res -eq 2  ]]
  then
    echo "sleep 2"
    sleep 2
  else
    echo "break loop"
    break
  fi
done

nohup python main/3_benchmark_sampler/benchmark_sampling_online.py --arch_sampler rl --is_vote 0 --pre_scored_data  "./result_append/CIFAR10_15625/union/101_15k_c10_bs32_ic16_unionBest.json" --search_space nasbench101 --device cuda:0 --total_run 300 --num_arch_each_run 400 &

sleep 10
while true; do
  res=$(ps aux | grep main/3_benchmark_sampler/benchmark_sampling_online.py | wc -l)
  if [[ $res -eq 2  ]]
  then
    echo "sleep 2"
    sleep 2
  else
    echo "break loop"
    break
  fi
done

nohup python main/3_benchmark_sampler/benchmark_sampling_online.py --arch_sampler rl --is_vote 1 --pre_scored_data  "./result_append/CIFAR10_15625/union/101_15k_c10_bs32_ic16_unionBest.json" --search_space nasbench101 --device cuda:0 --total_run 300 --num_arch_each_run 400 &

sleep 10
while true; do
  res=$(ps aux | grep main/3_benchmark_sampler/benchmark_sampling_online.py | wc -l)
  if [[ $res -eq 2  ]]
  then
    echo "sleep 2"
    sleep 2
  else
    echo "break loop"
    break
  fi
done

nohup python main/3_benchmark_sampler/bohb.py --is_vote 0 --out_folder "./" --pre_scored_data "./result_append/CIFAR10_15625/union/101_15k_c10_bs32_ic16_unionBest.json" --search_space nasbench101 --device cuda:0 --total_run 300 --num_arch_each_run 1000 &
sleep 2
nohup python main/3_benchmark_sampler/bohb.py --is_vote 1 --out_folder "./" --pre_scored_data "./result_append/CIFAR10_15625/union/101_15k_c10_bs32_ic16_unionBest.json" --search_space nasbench101 --device cuda:1 --total_run 300 --num_arch_each_run 1000 &

#nohup python main/2_verify_sampler/check_sampler.py --total_run 500 --num_arch_each_run 1000 &
