import json
import matplotlib.pyplot as plt
import numpy as np

# Assume these are the names and corresponding JSON files of your datasets
datasets_cpu = {
    'frappe': {'cache': './internal/ml/model_selection/exp_result_sever_filtering_cache/exp_filter_cache'
                        '/time_score_mlp_sp_frappe_batch_size_32_cpu.json',
               'no_cache': './internal/ml/model_selection/exp_result_sever/exp_result'
                           '/time_score_mlp_sp_frappe_batch_size_32_cpu.json'},

    'diabetes': {'cache': './internal/ml/model_selection/exp_result_sever_filtering_cache/exp_filter_cache'
                          '/time_score_mlp_sp_uci_diabetes_batch_size_32_cpu.json',
                 'no_cache': './internal/ml/model_selection/exp_result_sever/exp_result'
                             '/time_score_mlp_sp_uci_diabetes_batch_size_32_cpu.json'},

    'criteo': {'cache': './internal/ml/model_selection/exp_result_sever_filtering_cache/exp_filter_cache'
                        '/time_score_mlp_sp_criteo_batch_size_32_cpu.json',

               'no_cache': './internal/ml/model_selection/exp_result_sever/exp_result'
                           '/time_score_mlp_sp_criteo_batch_size_32_cpu.json'
               },

}

# Set your plot parameters
bar_width = 0.25
opacity = 0.8
set_font_size = 20  # Set the font size
set_lgend_size = 15
set_tick_size = 20
cpu_colors = ['#729ECE', '#FFB579']  # Colors for CPU bars
gpu_colors = ['#98DF8A', '#D62728']  # Colors for GPU bars
hatches = ['/', '\\', 'x', '.', '*']

# Load your datasets
datasets = dict(list(datasets_cpu.items()))

# Create a figure
fig, ax = plt.subplots(figsize=(6.4, 4.5))

for i, (dataset_name, json_files) in enumerate(datasets.items()):
    # Load the JSON data for cpu
    with open(json_files['cache']) as f:
        data_cache = json.load(f)

    # Load the JSON data for gpu
    with open(json_files['no_cache']) as f:
        data_non_cache = json.load(f)

    # Plot bars for cpu
    ax.bar(i - bar_width / 2, data_cache['io_latency'], bar_width,
           alpha=opacity, color=cpu_colors[0], hatch=hatches[0], edgecolor='black',
           label='(Embedding Cache) Model I/O' if i == 0 else "")

    ax.bar(i - bar_width / 2, data_cache['compute_latency'], bar_width,
           alpha=opacity, color=cpu_colors[1], hatch=hatches[1], edgecolor='black',
           label='(Embedding Cache) TFMEM' if i == 0 else "",
           bottom=data_cache['io_latency'])

    # Plot bars for gpu
    ax.bar(i + bar_width / 2, data_non_cache['io_latency'], bar_width,
           alpha=opacity, color=gpu_colors[0], hatch=hatches[2], edgecolor='black',
           label='Model I/O' if i == 0 else "")

    ax.bar(i + bar_width / 2, data_non_cache['compute_latency'], bar_width,
           alpha=opacity, color=gpu_colors[1], hatch=hatches[3], edgecolor='black',
           label='TFMEM' if i == 0 else "",
           bottom=data_non_cache['io_latency'])

ax.set_xticks(range(len(datasets)))
ax.set_xticklabels(datasets.keys(), fontsize=set_font_size)
ax.set_yscale('symlog')
ax.set_ylim(0, 3000)

# Set axis labels and legend
ax.set_ylabel('Latency (s)', fontsize=set_font_size)
ax.legend(fontsize=set_lgend_size, loc='upper left', ncol=1)
ax.tick_params(axis='both', which='major', labelsize=set_tick_size)
plt.tight_layout()
plt.show()

# Save the plot
fig.savefig(f"./internal/ml/model_selection/exp_result_sever/exp_result/embedding_cache.pdf",
            bbox_inches='tight')
