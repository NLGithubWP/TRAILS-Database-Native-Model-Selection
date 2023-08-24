from src.tools.io_tools import read_json
import matplotlib.pyplot as plt

# Set your plot parameters
bar_width = 0.25
opacity = 0.8
set_font_size = 20  # Set the font size
set_lgend_size = 15
set_tick_size = 15
cpu_colors = ['#729ECE', '#FFB579', '#E74C3C', '#2ECC71', '#3498DB', '#F39C12', '#8E44AD', '#C0392B']
gpu_colors = ['#98DF8A', '#D62728', '#1ABC9C', '#9B59B6', '#34495E', '#16A085', '#27AE60', '#2980B9']
hatches = ['/', '\\', 'x', 'o', 'O', '.', '*', '//', '\\\\', 'xx', 'oo', 'OO', '..', '**']
# hatches = ['', '', '', '', '']

# Assume these are the names and corresponding JSON files of your datasets
datasets_wo_cache = {
    'frappe': {
        'with_cache': './internal/ml/model_selection/exp_result_sever_filtering_cache'
                      '/resource_score_mlp_sp_frappe_batch_size_32_cpu_express_flow.json',
        'wo_cache': './internal/ml/model_selection/exp_result_sever_wo_cache'
                    '/resource_score_mlp_sp_frappe_batch_size_32_cpu_express_flow.json'},

    'diabetes': {
        'cache': './internal/ml/model_selection/exp_result_sever_filtering_cache'
                 '/resource_score_mlp_sp_uci_diabetes_batch_size_32_cpu_express_flow.json',
        'wo_cache': './internal/ml/model_selection/exp_result_sever_wo_cache'
                    '/resource_score_mlp_sp_uci_diabetes_batch_size_32_cpu_express_flow.json'},

    'criteo': {
        'cache': './internal/ml/model_selection/exp_result_sever_filtering_cache'
                 '/resource_score_mlp_sp_criteo_batch_size_32_cpu_express_flow.json',
        'wo_cache': './internal/ml/model_selection/exp_result_sever_wo_cache'
                    '/resource_score_mlp_sp_criteo_batch_size_32_cpu_express_flow.json'}
}


def plot_memory_usage(params, interval=0.5):
    fig = plt.figure(figsize=(6.4, 4.5))
    for dataset_name, value in params.items():
        for is_cache, file_path in value.items():
            metrics = read_json(file_path)
            # Extract GPU memory usage for device 0
            memory_usage_lst = metrics['memory_usage']
            # Create a time list
            times = [interval * i for i in range(len(memory_usage_lst))]
            # plt.plot(times, metrics['memory_usage'], label='CPU Memory Usage (MB))
            plt.plot(times, memory_usage_lst, label=dataset_name + "_" + is_cache)

    plt.xlabel('Time (Seconds)', fontsize=set_font_size)
    plt.ylabel('Memory Usage (MB)', fontsize=set_font_size)
    plt.legend(fontsize=set_lgend_size)

    # Set tick size
    plt.tick_params(axis='both', which='major', labelsize=set_tick_size)
    plt.xscale("symlog")
    plt.tight_layout()
    plt.grid(True)
    # plt.show()
    print(f"saving to ./internal/ml/model_selection/exp_result/filter_latency_memory_cache.pdf")
    fig.savefig(f"./internal/ml/model_selection/exp_result/filter_latency_memory_cache.pdf",
                bbox_inches='tight')


# Call the function
plot_memory_usage(datasets_wo_cache)
