import json
import matplotlib.pyplot as plt
import numpy as np

# Assume these are the names and corresponding JSON files of your datasets
datasets_wo_cache = {
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

datasets_embedding_cache = {
    'frappe': {'cpu': './internal/ml/model_selection/exp_result_sever_filtering_cache/exp_filter_cache'
                      '/time_score_mlp_sp_frappe_batch_size_32_cpu.json',
               'gpu': './internal/ml/model_selection/exp_result_sever/exp_result'
                      '/time_score_mlp_sp_frappe_batch_size_32_cuda:0.json'},

    'diabetes': {'cpu': './internal/ml/model_selection/exp_result_sever_filtering_cache/exp_filter_cache'
                        '/time_score_mlp_sp_uci_diabetes_batch_size_32_cpu.json',
                 'gpu': './internal/ml/model_selection/exp_result_sever/exp_result'
                        '/time_score_mlp_sp_uci_diabetes_batch_size_32_cuda:0.json'},

    'criteo': {'cpu': './internal/ml/model_selection/exp_result_sever_filtering_cache/exp_filter_cache'
                      '/time_score_mlp_sp_criteo_batch_size_32_cpu.json',
               'gpu': './internal/ml/model_selection/exp_result_sever/exp_result'
                      '/time_score_mlp_sp_criteo_batch_size_32_cuda:0.json'},

    'c10': {'cpu': './internal/ml/model_selection/exp_result_sever_filtering_cache/exp_filter_cache'
                   '/time_score_nasbench201_cifar10_batch_size_32_cpu.json',
            'gpu': './internal/ml/model_selection/exp_result_sever/exp_result'
                   '/time_score_nasbench201_cifar10_batch_size_32_cuda:0.json'},

    'c100': {'cpu': './internal/ml/model_selection/exp_result_sever_filtering_cache/exp_filter_cache'
                    '/time_score_nasbench201_cifar100_batch_size_32_cpu.json',
             'gpu': './internal/ml/model_selection/exp_result_sever/exp_result'
                    '/time_score_nasbench201_cifar100_batch_size_32_cuda:0.json'},

    'IN-16': {'cpu': './internal/ml/model_selection/exp_result_sever_filtering_cache/exp_filter_cache'
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

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

for img_id, datasets in enumerate([datasets_wo_cache, datasets_embedding_cache]):
    # Define the grid for subplots with widths ratio as 1:2
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])

    fig = plt.figure(figsize=(6.4, 4.5))
    fig.subplots_adjust(wspace=0.01)  # adjust the space between

    # Split your datasets into two groups
    datasets_left = dict(list(datasets.items())[:2])
    datasets_right = dict(list(datasets.items())[2:])

    for idx, datasets in enumerate([datasets_left, datasets_right]):
        # Create a subplot with custom width
        ax = plt.subplot(gs[idx])

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
                   label='(CPU) Model I/O' if i == 0 else "")

            ax.bar(i - bar_width / 2, data_cpu['compute_latency'], bar_width,
                   alpha=opacity, color=cpu_colors[1], hatch=hatches[1], edgecolor='black',
                   label='(CPU) TFMEM' if i == 0 else "",
                   bottom=data_cpu['io_latency'])

            # Plot bars for gpu
            ax.bar(i + bar_width / 2, data_gpu['io_latency'], bar_width,
                   alpha=opacity, color=gpu_colors[0], hatch=hatches[2], edgecolor='black',
                   label='(GPU) Model I/O' if i == 0 else "")

            ax.bar(i + bar_width / 2, data_gpu['compute_latency'], bar_width,
                   alpha=opacity, color=gpu_colors[1], hatch=hatches[3], edgecolor='black',
                   label='(GPU) TFMEM' if i == 0 else "",
                   bottom=data_gpu['io_latency'])

            ax.set_xticks(range(len(datasets)))
            ax.set_xticklabels(datasets.keys(), fontsize=fontsize)

        # Set axis labels and legend for first subplot only
        if idx == 0:
            ax.set_ylabel('Latency (s)', fontsize=fontsize)
            ax.legend().remove()  # remove the legend
        else:
            # ax.set_ylim(0, 2000)
            ax.legend(fontsize=fontsize, loc='upper right', ncol=1)

    fig.tight_layout()
    plt.show()

    # Save the plot
    fig.savefig(f"./internal/ml/model_selection/exp_result_sever/exp_result/filter_latency_{img_id}.pdf",
                bbox_inches='tight')
