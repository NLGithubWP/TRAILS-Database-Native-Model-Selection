import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# Helper function to load data
def load_data(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)


# Set your plot parameters
bar_width = 0.25
opacity = 0.8
set_font_size = 15  # Set the font size
set_lgend_size = 10
set_tick_size = 12
cpu_colors = ['#FFB579']
gpu_colors = ['#3498DB']
gpu_memory_colors = ['#98DF8A']

hatches = ['/', '\\', 'x', 'o', 'O', '.', '*', '//', '\\\\', 'xx', 'oo', 'OO', '..', '**']
# hatches = ['', '', '', '', '']


datasets_wo_cache = {
    'Frappe': {
        'gpu': './internal/ml/model_selection/exp_result_sever_wo_cache'
               '/resource_score_mlp_sp_frappe_batch_size_32_cuda:0_express_flow.json',
        'cpu': './internal/ml/model_selection/exp_result_sever_wo_cache'
               '/resource_score_mlp_sp_frappe_batch_size_32_cpu_express_flow.json'
    },

    'Diabetes': {
        'gpu': './internal/ml/model_selection/exp_result_sever_wo_cache'
               '/resource_score_mlp_sp_uci_diabetes_batch_size_32_cuda:0_express_flow.json',
        'cpu': './internal/ml/model_selection/exp_result_sever_wo_cache'
               '/resource_score_mlp_sp_uci_diabetes_batch_size_32_cpu_express_flow.json'
    },

    'Criteo': {
        'gpu': './internal/ml/model_selection/exp_result_sever_wo_cache'
               '/resource_score_mlp_sp_criteo_batch_size_32_cuda:0_express_flow.json',
        'cpu': './internal/ml/model_selection/exp_result_sever_wo_cache'
               '/resource_score_mlp_sp_criteo_batch_size_32_cpu_express_flow.json'
    },

    'C10': {
        'gpu': './internal/ml/model_selection/exp_result_sever_wo_cache'
               '/resource_score_nasbench201_cifar10_batch_size_32_cuda:0_express_flow.json',
        'cpu': './internal/ml/model_selection/exp_result_sever_wo_cache'
               '/resource_score_nasbench201_cifar10_batch_size_32_cpu_express_flow.json'
    },

    'C100': {
        'gpu': './internal/ml/model_selection/exp_result_sever_wo_cache'
               '/resource_score_nasbench201_cifar100_batch_size_32_cuda:0_express_flow.json',
        'cpu': './internal/ml/model_selection/exp_result_sever_wo_cache'
               '/resource_score_nasbench201_cifar100_batch_size_32_cpu_express_flow.json'
    },

    'IN-16': {
        'gpu': './internal/ml/model_selection/exp_result_sever_wo_cache'
               '/resource_score_nasbench201_ImageNet16-120_batch_size_32_cuda:0_express_flow.json',
        'cpu': './internal/ml/model_selection/exp_result_sever_wo_cache'
               '/resource_score_nasbench201_ImageNet16-120_batch_size_32_cpu_express_flow.json'
    },
}


# Collecting data for plotting
datasets = list(datasets_wo_cache.keys())
gpu_totals, cpu_totals, gpu_mem_device, gpu_mem_host = [], [], [], []

for dataset in datasets:
    # For GPU
    gpu_data = load_data(datasets_wo_cache[dataset]['gpu'])

    gpu_mem_device_0 = [mem[2] for mem in gpu_data['gpu_usage'] if mem[0] == 0]
    break_point = next((idx for idx, val in enumerate(gpu_mem_device_0[:-1]) if gpu_mem_device_0[idx + 1] > 200), 0)

    gpu_mem_device_0 = gpu_mem_device_0[break_point:]
    mem_host_gpu = gpu_data['memory_usage'][break_point:]
    total_memory_usage_gpu = [a + b for a, b in zip(gpu_mem_device_0, mem_host_gpu)]

    # For CPU
    cpu_data = load_data(datasets_wo_cache[dataset]['cpu'])
    mem_host_cpu = cpu_data['memory_usage']

    # Appending pick memory usage
    gpu_totals.append(max(total_memory_usage_gpu))
    gpu_mem_device.append(max(gpu_mem_device_0))
    gpu_mem_host.append(max(mem_host_gpu))
    cpu_totals.append(max(mem_host_cpu))

# Plotting
fig, ax = plt.subplots(figsize=(6.4, 4.5))

index = np.arange(len(datasets))

# Left bar - GPU total memory usage split into GPU memory and host memory
ax.bar(index - bar_width / 2, gpu_mem_host, bar_width, color=gpu_colors, hatch=hatches[0],
       label='(GPU) Host Memory', edgecolor='black')
ax.bar(index - bar_width / 2, gpu_mem_device, bar_width, color=gpu_memory_colors, hatch=hatches[1],
       bottom=gpu_mem_host,
       label='(GPU) GPU Memory', edgecolor='black')

# Right bar - CPU host memory
ax.bar(index + bar_width / 2, cpu_totals, bar_width, color=cpu_colors, hatch=hatches[2],
       label='(CPU) Host Memory', edgecolor='black')

ax.set_ylabel('Memory (MB)', fontsize=set_font_size)
ax.set_xticks(index)
# ax.set_yscale('symlog')  # Set y-axis to logarithmic scale
ax.set_xticklabels(datasets, rotation=0, fontsize=set_font_size)
ax.legend(fontsize=set_lgend_size)

ax.tick_params(axis='y', which='major', labelsize=set_tick_size + 5)

plt.tight_layout()
fig.tight_layout()
# plt.show()
print(f"saving to ./internal/ml/model_selection/exp_result/filter_latency_memory_bar.pdf")
fig.savefig(f"./internal/ml/model_selection/exp_result/filter_latency_memory_bar.pdf",
            bbox_inches='tight')