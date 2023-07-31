import json
import matplotlib.pyplot as plt
import numpy as np

# Assume these are the names and corresponding JSON files of your datasets
datasets = {
    'frappe': {'cpu': './internal/ml/model_selection/exp_result_sever/exp_result'
                      '/time_score_mlp_sp_frappe_batch_size_32_cpu.json',
               'gpu': './internal/ml/model_selection/exp_result_sever/exp_result'
                      '/time_score_mlp_sp_frappe_batch_size_32_cuda:0.json'},

    'diabetes': {'cpu': './internal/ml/model_selection/exp_result_sever/exp_result'
                        '/time_score_mlp_sp_uci_diabetes_batch_size_32_cpu.json',
                 'gpu': './internal/ml/model_selection/exp_result_sever/exp_result'
                        '/time_score_mlp_sp_uci_diabetes_batch_size_32_cuda:0.json'},

    'criteo': {'cpu': './internal/ml/model_selection/exp_result_sever/exp_result'
                      '/time_score_mlp_sp_criteo_batch_size_32_cpu.json',
               'gpu': './internal/ml/model_selection/exp_result_sever/exp_result'
                      '/time_score_mlp_sp_criteo_batch_size_32_cuda:0.json'},

    'c10': {'cpu': './internal/ml/model_selection/exp_result_sever/exp_result'
                   '/time_score_nasbench201_cifar10_batch_size_32_cpu.json',
            'gpu': './internal/ml/model_selection/exp_result_sever/exp_result'
                   '/time_score_nasbench201_cifar10_batch_size_32_cuda:0.json'},

    'c100': {'cpu': './internal/ml/model_selection/exp_result_sever/exp_result'
                    '/time_score_nasbench201_cifar100_batch_size_32_cpu.json',
             'gpu': './internal/ml/model_selection/exp_result_sever/exp_result'
                    '/time_score_nasbench201_cifar100_batch_size_32_cuda:0.json'},

    'IN-16': {'cpu': './internal/ml/model_selection/exp_result_sever/exp_result'
                     '/time_score_nasbench201_ImageNet16-120_batch_size_32_cpu.json',
              'gpu': './internal/ml/model_selection/exp_result_sever/exp_result'
                     '/time_score_nasbench201_ImageNet16-120_batch_size_32_cuda:0.json'},
}

# Set your plot parameters
bar_width = 0.25
opacity = 0.8
fontsize = 14  # Set the font size
cpu_colors = ['#729ECE', '#FFB579']  # Colors for CPU bars
gpu_colors = ['#98DF8A', '#D62728']  # Colors for GPU bars
hatches = ['/', '\\', 'x', '.', '*']
# hatches = ['', '', '', '', '']

# Create the figure and axis
fig, ax = plt.subplots(figsize=(6.4, 4.5))
ax.grid()

for i, (dataset_name, json_files) in enumerate(datasets.items()):
    # Load the JSON data for cpu
    with open(json_files['cpu']) as f:
        data_cpu = json.load(f)

    # Load the JSON data for gpu
    with open(json_files['gpu']) as f:
        data_gpu = json.load(f)

    # Plot bars for cpu
    ax.bar(i - bar_width / 2, data_cpu['io_latency'], bar_width,
           alpha=opacity, color=cpu_colors[0], hatch=hatches[0], edgecolor='black',
           label='CPU IO' if i == 0 else "")

    ax.bar(i - bar_width / 2, data_cpu['compute_latency'], bar_width,
           alpha=opacity, color=cpu_colors[1], hatch=hatches[1], edgecolor='black',
           label='CPU Compute' if i == 0 else "",
           bottom=data_cpu['io_latency'])

    # Plot bars for gpu
    ax.bar(i + bar_width / 2, data_gpu['io_latency'], bar_width,
           alpha=opacity, color=gpu_colors[0], hatch=hatches[2], edgecolor='black',
           label='GPU IO' if i == 0 else "")

    ax.bar(i + bar_width / 2, data_gpu['compute_latency'], bar_width,
           alpha=opacity, color=gpu_colors[1], hatch=hatches[3], edgecolor='black',
           label='GPU Compute' if i == 0 else "",
           bottom=data_gpu['io_latency'])

# Set x-ticks and x-tick labels
ax.set_xticks(range(len(datasets)))
ax.set_xticklabels(datasets.keys(), fontsize=fontsize)

# Set axis labels
ax.set_ylabel('Latency (s)', fontsize=fontsize)

# Set y-scale to logarithmic
ax.set_yscale('log')
# adjust the upper limit as per your need
plt.ylim(0, 20 ** 3)

# Set other parameters and show plot
ax.legend(fontsize=fontsize, loc='upper right', ncol=2)
fig.tight_layout()
plt.show()

# Save the plot
fig.savefig("./internal/ml/model_selection/exp_result_sever/exp_result/filter_latency.pdf",
            bbox_inches='tight')
