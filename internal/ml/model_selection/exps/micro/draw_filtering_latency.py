import json
import matplotlib.pyplot as plt
import numpy as np

# Assume these are the names and corresponding JSON files of your datasets
datasets = {
    'dataset1': {'cpu': './internal/ml/model_selection/exp_result_sever/exp_result'
          '/time_score_mlp_sp_frappe_batch_size_32_cuda:0.json',
                 'gpu': './internal/ml/model_selection/exp_result_sever/exp_result'
          '/time_score_mlp_sp_frappe_batch_size_32_cuda:0.json'},

    'dataset2': {'cpu': './internal/ml/model_selection/exp_result_sever/exp_result'
          '/time_score_mlp_sp_frappe_batch_size_32_cuda:0.json',
                 'gpu': './internal/ml/model_selection/exp_result_sever/exp_result'
          '/time_score_mlp_sp_frappe_batch_size_32_cuda:0.json'},

    'dataset3': {'cpu': './internal/ml/model_selection/exp_result_sever/exp_result'
          '/time_score_mlp_sp_frappe_batch_size_32_cuda:0.json',
                 'gpu': './internal/ml/model_selection/exp_result_sever/exp_result'
          '/time_score_mlp_sp_frappe_batch_size_32_cuda:0.json'},
}

# Set your plot parameters
bar_width = 0.15
opacity = 0.8
fontsize = 14  # Set the font size
colors = ['#729ECE', '#FFB579', '#98DF8A']  # Softer colors #FF7F7F
hatches = ['/', '\\',  'x', '.', '*']

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
    ax.bar(i - bar_width/2, data_cpu['io_latency'], bar_width,
            alpha=opacity, color=colors[0], hatch=hatches[0], edgecolor='black', label='CPU IO Latency' if i==0 else "")

    ax.bar(i - bar_width/2, data_cpu['compute_latency'], bar_width,
            alpha=opacity, color=colors[1], hatch=hatches[1], edgecolor='black', label='CPU Compute Latency' if i==0 else "",
            bottom=data_cpu['io_latency'])

    # Plot bars for gpu
    ax.bar(i + bar_width/2, data_gpu['io_latency'], bar_width,
            alpha=opacity, color=colors[0], hatch=hatches[2], edgecolor='black', label='GPU IO Latency' if i==0 else "")

    ax.bar(i + bar_width/2, data_gpu['compute_latency'], bar_width,
            alpha=opacity, color=colors[1], hatch=hatches[3], edgecolor='black', label='GPU Compute Latency' if i==0 else "",
            bottom=data_gpu['io_latency'])

# Set x-ticks and x-tick labels
ax.set_xticks(range(len(datasets)))
ax.set_xticklabels(datasets.keys(), fontsize=fontsize)

# Set axis labels
ax.set_ylabel('Time Consumption (s)', fontsize=fontsize)

# Set other parameters and show plot
ax.legend(fontsize=fontsize, loc='upper right')
ax.tick_params(axis='both', labelsize=fontsize)
fig.tight_layout()
plt.show()

# Save the plot
fig.savefig("./internal/ml/model_selection/exp_result_sever/exp_result/image.pdf", bbox_inches='tight')