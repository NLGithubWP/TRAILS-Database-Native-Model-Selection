import json
import matplotlib.pyplot as plt
import numpy as np

# Assume these are the names and corresponding JSON files of your datasets
datasets_wo_cache = {
    'frappe': {'gpu': './internal/ml/model_selection/exp_result'
                      '/resource_score_mlp_sp_frappe_batch_size_32_cuda:0_express_flow.json'},
    # 'diabetes': {'cpu': './internal/ml/model_selection/exp_result_sever/exp_result'
    #                     '/time_score_mlp_sp_uci_diabetes_batch_size_32_cpu.json',
    #              'gpu': './internal/ml/model_selection/exp_result'
    #                     '/time_score_mlp_sp_uci_diabetes_batch_size_32_cuda:0_express_flow.json'},
    #
    # 'criteo': {'cpu': './internal/ml/model_selection/exp_result_sever/exp_result'
    #                   '/time_score_mlp_sp_criteo_batch_size_32_cpu.json',
    #            'gpu': './internal/ml/model_selection/exp_result'
    #                   '/time_score_mlp_sp_criteo_batch_size_32_cuda:0_express_flow.json'},
    #
    # 'c10': {'cpu': './internal/ml/model_selection/exp_result_sever/exp_result'
    #                '/time_score_nasbench201_cifar10_batch_size_32_cpu.json',
    #         'gpu': './internal/ml/model_selection/exp_result_sever/exp_result'
    #                '/time_score_nasbench201_cifar10_batch_size_32_cuda:0.json'},
    # 'c100': {'cpu': './internal/ml/model_selection/exp_result_sever/exp_result'
    #                 '/time_score_nasbench201_cifar100_batch_size_32_cpu.json',
    #          'gpu': './internal/ml/model_selection/exp_result_sever/exp_result'
    #                 '/time_score_nasbench201_cifar100_batch_size_32_cuda:0.json'},
    #
    # 'IN-16': {'cpu': './internal/ml/model_selection/exp_result_sever/exp_result'
    #                  '/time_score_nasbench201_ImageNet16-120_batch_size_32_cpu.json',
    #           'gpu': './internal/ml/model_selection/exp_result_sever/exp_result'
    #                  '/time_score_nasbench201_ImageNet16-120_batch_size_32_cuda:0.json'},
}

from src.tools.io_tools import read_json

import matplotlib.pyplot as plt
import json


def plot_memory_usage(metrics_file, interval=0.5):
    metrics = read_json(datasets_wo_cache['frappe']["gpu"])

    # Extract GPU memory usage for device 0
    gpu_mem_device_0 = [mem[2] for mem in metrics['gpu_usage'] if mem[0] == 0]

    # Create a time list
    times = [interval * i for i in range(len(gpu_mem_device_0))]

    # Plotting
    plt.figure(figsize=(10, 6))

    plt.plot(times, metrics['memory_usage'], label='CPU Memory Usage (MB)', color='blue')
    plt.plot(times, gpu_mem_device_0, label='GPU Memory Usage (MB)', color='red')

    plt.xlabel('Time (seconds)')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage over Time')
    plt.legend()

    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Call the function
plot_memory_usage("path_to_folder")
