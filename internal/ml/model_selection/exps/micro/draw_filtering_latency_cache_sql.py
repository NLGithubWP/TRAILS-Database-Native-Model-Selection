import json
import matplotlib.pyplot as plt
import numpy as np

# Assume these are the names and corresponding JSON files of your datasets
datasets_cpu = {
    'Frappe': {'in_db': './internal/ml/model_selection/exp_result_sever_cache_sql_indb/'
                        '/time_score_mlp_sp_frappe_batch_size_32_cpu_express_flow.json',
               'out_db': './internal/ml/model_selection/exp_result_sever_cache_sql/'
                         '/time_score_mlp_sp_frappe_batch_size_32_cpu_express_flow.json'},

    'Diabetes': {'in_db': './internal/ml/model_selection/exp_result_sever_cache_sql_indb/'
                          '/time_score_mlp_sp_uci_diabetes_batch_size_32_cpu_express_flow.json',
                 'out_db': './internal/ml/model_selection/exp_result_sever_cache_sql/'
                           '/time_score_mlp_sp_uci_diabetes_batch_size_32_cpu_express_flow.json'},

    'Criteo': {'in_db': './internal/ml/model_selection/exp_result_sever_cache_sql_indb/'
                        '/time_score_mlp_sp_criteo_batch_size_32_cpu_express_flow.json',
               'out_db': './internal/ml/model_selection/exp_result_sever_cache_sql/'
                         '/time_score_mlp_sp_criteo_batch_size_32_cpu_express_flow.json'},
}

# Set your plot parameters
bar_width = 0.25
opacity = 0.8
set_font_size = 16  # Set the font size
set_lgend_size = 12
set_tick_size = 16
cpu_colors = ['#729ECE', '#FFB579', '#E74C3C', '#2ECC71', '#3498DB', '#F39C12', '#8E44AD', '#C0392B']
gpu_colors = ['#98DF8A', '#D62728', '#1ABC9C', '#9B59B6', '#34495E', '#16A085', '#27AE60', '#2980B9']
hatches = ['/', '\\', 'x', 'o', 'O', '.', '*', '//', '\\\\', 'xx', 'oo', 'OO', '..', '**']

# Load your datasets
datasets = dict(list(datasets_cpu.items()))

# Create a figure
fig, ax = plt.subplots(figsize=(6.4, 4.5))

for i, (dataset_name, json_files) in enumerate(datasets.items()):
    # Load the JSON data for cpu
    with open(json_files['in_db']) as f:
        data_in_db = json.load(f)

    # Load the JSON data for gpu
    with open(json_files['out_db']) as f:
        data_out_db = json.load(f)

    ## debug
    print("==="*20)
    print(dataset_name, f"in-db: avg_compute_latency={data_in_db['avg_compute_latency']}, "
                        f"compute_latency={data_in_db['compute_latency']}, "
                        f"track_io_model_init={sum(data_in_db['track_io_model_init'][2:])}, "
                        f"track_io_data_retrievel={sum(data_in_db['track_io_data_retrievel'][2:])}, "
                        f"track_io_data_preprocess={sum(data_in_db['track_io_data_preprocess'][2:])}, ")

    print(dataset_name, f"out-db: avg_compute_latency={str(data_out_db['avg_compute_latency'])}, "
                        f"compute_latency={data_out_db['compute_latency']}, ",
                        f"track_io_model_init={sum(data_out_db['track_io_model_init'][2:])}, ",
                        f"track_io_data_retrievel={sum(data_out_db['track_io_data_retrievel'][2:])}, ",
                        f"track_io_data_preprocess={sum(data_out_db['track_io_data_preprocess'][2:])}",
          )

    # Plot bars for cpu
    data_retrievel_latency = sum(data_in_db['track_io_data_retrievel'][2:])
    data_pre_processing_latency = sum(data_in_db['track_io_data_preprocess'][2:])
    sql_overall_latency = data_in_db['latency'] - \
                          sum(data_in_db['track_io_data_retrievel'][2:])
                          # - sum(data_in_db['track_io_data_preprocess'][2:])

    ax.bar(i - bar_width / 2, data_retrievel_latency, bar_width,
           alpha=opacity, color=cpu_colors[0], hatch=hatches[0], edgecolor='black',
           label='(In-DB) Data Retrievel' if i == 0 else "")

    # ax.bar(i - bar_width / 2,
    #        data_pre_processing_latency,
    #        bar_width,
    #        alpha=opacity, color=cpu_colors[1], hatch=hatches[1], edgecolor='black',
    #        label='(In-DB) Data Processing' if i == 0 else "",
    #        bottom=data_retrievel_latency)

    ax.bar(i - bar_width / 2,
           sql_overall_latency,
           bar_width,
           alpha=opacity, color=cpu_colors[2], hatch=hatches[1], edgecolor='black',
           label='(In-DB) Data Proc & Model Init & TFMEM' if i == 0 else "",
           bottom=data_retrievel_latency)

    # Plot bars for gpu
    data_retrievel_latency = sum(data_out_db['track_io_data_retrievel'][2:])
    data_pre_processing_latency = sum(data_out_db['track_io_data_preprocess'][2:])
    sql_overall_latency = data_out_db['latency'] - \
                          sum(data_out_db['track_io_data_retrievel'][2:])
                          # - sum(data_out_db['track_io_data_preprocess'][2:])

    ax.bar(i + bar_width / 2, data_retrievel_latency, bar_width,
           alpha=opacity, color=gpu_colors[0], hatch=hatches[2], edgecolor='black',
           label='Data Retrievel' if i == 0 else "")

    # ax.bar(i + bar_width / 2,
    #        data_pre_processing_latency,
    #        bar_width,
    #        alpha=opacity, color=gpu_colors[3], hatch=hatches[1], edgecolor='black',
    #        label='Data Processing' if i == 0 else "",
    #        bottom=data_retrievel_latency)

    ax.bar(i + bar_width / 2,
           sql_overall_latency,
           bar_width,
           alpha=opacity, color=gpu_colors[2], hatch=hatches[7], edgecolor='black',
           label='Data Proc & Model Init & TFMEM' if i == 0 else "",
           bottom=data_retrievel_latency)

ax.set_xticks(range(len(datasets)))
ax.set_xticklabels(datasets.keys(), fontsize=set_font_size)
# ax.set_yscale('symlog')
ax.set_ylim(0, 180)

# Set axis labels and legend
ax.set_ylabel('Latency (s)', fontsize=set_font_size)
ax.legend(fontsize=set_lgend_size, loc='upper right', ncol=1)
ax.tick_params(axis='both', which='major', labelsize=set_tick_size)
plt.tight_layout()

# Save the plot
fig.savefig(f"./internal/ml/model_selection/exp_result/filtering_sql_2.pdf",
            bbox_inches='tight')
