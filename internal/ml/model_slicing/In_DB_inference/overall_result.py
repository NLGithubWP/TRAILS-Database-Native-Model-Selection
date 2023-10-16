import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.ticker import FuncFormatter
from brokenaxes import brokenaxes


def thousands_formatter(x, pos):
    if x >= 1e3:
        return '{:.1f}k'.format(x * 1e-3)
    else:
        return '{:.1f}'.format(x)


thousands_format = FuncFormatter(thousands_formatter)


# Helper function to load data
def load_data(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)


# Set your plot parameters
bar_width = 0.25
opacity = 0.8
set_font_size = 15  # Set the font size
set_lgend_size = 12
set_tick_size = 12
colors = ['#729ECE', '#2ECC71', '#8E44AD', '#3498DB', '#F39C12']
hatches = ['/', '\\', 'x', '.', '*', '//', '\\\\', 'xx', '..', '**']


def scale_to_ms(latencies):
    result = {}
    for key, value in latencies.items():
        value = value * 1000
        result[key] = value
    return result


# here run 10k rows for inference.,
# each sub-list is "compute time" and "data fetch time"
datasets_result = {
    'Frappe': {
        'In-Db':
            {'model_init_time': 0.007789646, 'data_query_time_spi': 0.115799835, 'data_query_time': 5.050895535,
             'overall_query_latency': 11.061697293, 'python_compute_time': 6.002932697, 'diff': -7.941500000008261e-05,
             'py_conver_to_tensor': 1.5937657356262207, 'py_compute': 1.6089363956451416,
             'py_overall_duration': 3.496934652328491, 'py_diff': 0.6342325210571289},

        'In-Db-opt':
            {'python_compute_time': 4.490436555, 'overall_query_latency': 6.706514911,
             'data_query_time_spi': 0.107068849, 'data_query_time': 2.207659889, 'diff': -0.0003570320000010341,
             'model_init_time': 0.008061435, 'mem_allocate_time': 0.000264492, 'py_conver_to_tensor': 1.836038589477539,
             'py_compute': 1.5662827491760254, 'py_overall_duration': 4.0564281940460205, 'py_diff': 0.654106855392456},

        'out-DB-cpu':
            {'data_query_time': 0.6817991733551025, 'py_conver_to_tensor': 2.835449457168579,
             'tensor_to_gpu': 0.00025272369384765625, 'py_compute': 1.6024252700805664,
             'overall_query_latency': 5.489258050918579},

        'out-DB-gpu':
            {'data_query_time': 0.6990134716033936, 'py_conver_to_tensor': 2.716031074523926,
             'tensor_to_gpu': 19.514512062072754, 'py_compute': 0.1083493709564209,
             'overall_query_latency': 23.310205221176147},
    },

    # 'Adult': {
    #     'In-Db': {},
    #     'out-DB-gpu': {},
    #     'out-DB-cpu': {},
    # },
    #
    # 'Cvd': {
    #     'In-Db': {},
    #     'out-DB-gpu': {},
    #     'out-DB-cpu': {},
    # },
    #
    # 'Bank': {
    #     'In-Db': {},
    #     'out-DB-gpu': {},
    #     'out-DB-cpu': {},
    # },
    #
}

# Collecting data for plotting
datasets = list(datasets_result.keys())

# Plotting
fig = plt.figure(figsize=(6.4, 4.5))

# Create a broken y-axis within the fig
ax = brokenaxes(ylims=((0, 6000), (22000, 24000)), hspace=.25, fig=fig, d=0)

index = np.arange(len(datasets))
# Initial flags to determine whether the labels have been set before
set_label_outdb_gpu_data = True
set_label_outdb_gpu_inference = True
set_label_outdb_cpu_data = True
set_label_outdb_cpu_inference = True
set_label_indb_data = True
set_label_indb_inference = True

for dataset, valuedic in datasets_result.items():
    indb_med = scale_to_ms(valuedic["In-Db"])
    indb_med_opt = scale_to_ms(valuedic["In-Db-opt"])
    outgpudb_med = scale_to_ms(valuedic["out-DB-gpu"])
    outcpudb_med = scale_to_ms(valuedic["out-DB-cpu"])

    # in-db w/o optimize
    # this is query_from_db + copy_to_python +  luanch_python_module
    in_db_data_query = indb_med["data_query_time_spi"]
    in_db_data_copy_start_py = 0
    in_db_data_preprocess = indb_med["py_conver_to_tensor"] + \
                            indb_med["python_compute_time"] - \
                            indb_med["py_overall_duration"]
    in_db_data_compute = indb_med["py_compute"]
    # here - indb_med["data_query_time"] remove the type cpnvert time
    in_db_data_others = indb_med["overall_query_latency"] - \
                        indb_med["data_query_time"] - \
                        in_db_data_copy_start_py - \
                        in_db_data_preprocess - \
                        in_db_data_compute - \
                        indb_med["py_diff"] - indb_med["model_init_time"]

    label_in_db_data_query = 'Data Retrieval'
    label_in_db_data_copy_start_py = 'Data Copy'
    label_in_db_data_preprocess = 'Data Preprocess'
    label_in_db_data_compute = 'Model Inference'
    label_in_db_data_others = 'Others'

    ax.bar(index + 0.5 * bar_width, in_db_data_query, bar_width, color=colors[0], hatch=hatches[0],
           label=label_in_db_data_query, edgecolor='black')
    # ax.bar(index + bar_width, in_db_data_copy_start_py, bar_width, color=colors[1], hatch=hatches[1],
    #        bottom=in_db_data_query,
    #        label=label_in_db_data_copy_start_py, edgecolor='black')
    ax.bar(index + 0.5 * bar_width, in_db_data_preprocess, bar_width, color=colors[2], hatch=hatches[2],
           bottom=in_db_data_query + in_db_data_copy_start_py,
           label=label_in_db_data_preprocess, edgecolor='black')
    ax.bar(index + 0.5 * bar_width, in_db_data_compute, bar_width, color=colors[3], hatch=hatches[3],
           bottom=in_db_data_query + in_db_data_copy_start_py + in_db_data_preprocess,
           label=label_in_db_data_compute, edgecolor='black')
    # ax.bar(index + 0.5 * bar_width, in_db_data_others, bar_width, color=colors[4], hatch=hatches[4],
    #        bottom=in_db_data_query + in_db_data_copy_start_py + in_db_data_preprocess + in_db_data_compute,
    #        label=label_in_db_data_others, edgecolor='black')

    # in-db with optimizization
    in_db_data_copy_start_py = 0
    in_db_data_query = indb_med_opt["data_query_time_spi"]
    in_db_data_preprocess = indb_med_opt["py_conver_to_tensor"] \
                            + indb_med_opt["python_compute_time"] \
                            - indb_med_opt["py_overall_duration"]
    in_db_data_compute = indb_med_opt["py_compute"]
    in_db_data_others = indb_med_opt["overall_query_latency"] - \
                        indb_med_opt["data_query_time"] - \
                        in_db_data_copy_start_py - \
                        in_db_data_preprocess - \
                        in_db_data_compute

    ax.bar(index + 1.5 * bar_width, in_db_data_query, bar_width, color=colors[0], hatch=hatches[0],
           edgecolor='black')
    # ax.bar(index, in_db_data_copy_start_py, bar_width, color=colors[1], hatch=hatches[1],
    #        bottom=in_db_data_query,
    #        edgecolor='black')
    ax.bar(index + 1.5 * bar_width, in_db_data_preprocess, bar_width, color=colors[2], hatch=hatches[2],
           bottom=in_db_data_query + in_db_data_copy_start_py,
           edgecolor='black')
    ax.bar(index + 1.5 * bar_width, in_db_data_compute, bar_width, color=colors[3], hatch=hatches[3],
           bottom=in_db_data_query + in_db_data_copy_start_py + in_db_data_preprocess,
           edgecolor='black')
    # ax.bar(index + 1.5 * bar_width, in_db_data_others, bar_width, color=colors[4], hatch=hatches[4],
    #        bottom=in_db_data_query + in_db_data_copy_start_py + in_db_data_preprocess + in_db_data_compute,
    #        edgecolor='black')

    # out-db GPU
    in_db_data_query = outgpudb_med["data_query_time"]
    in_db_data_copy_gpu = outgpudb_med["tensor_to_gpu"]
    in_db_data_preprocess = outgpudb_med["py_conver_to_tensor"]
    in_db_data_compute = outgpudb_med["py_compute"]
    in_db_data_others = outgpudb_med["overall_query_latency"] - \
                        in_db_data_query - \
                        in_db_data_copy_gpu - \
                        in_db_data_preprocess - \
                        in_db_data_compute

    ax.bar(index - 1.5 * bar_width, in_db_data_query, bar_width, color=colors[0], hatch=hatches[0],
           edgecolor='black')
    ax.bar(index - 1.5 * bar_width, in_db_data_preprocess, bar_width, color=colors[2], hatch=hatches[2],
           bottom=in_db_data_query,
           edgecolor='black')
    ax.bar(index - 1.5 * bar_width, in_db_data_copy_gpu, bar_width, color=colors[1], hatch=hatches[1],
           bottom=in_db_data_query + in_db_data_preprocess,
           label=label_in_db_data_copy_start_py,
           edgecolor='black')
    ax.bar(index - 1.5 * bar_width, in_db_data_compute, bar_width, color=colors[3], hatch=hatches[3],
           bottom=in_db_data_query + in_db_data_copy_gpu + in_db_data_preprocess,
           edgecolor='black')
    # ax.bar(index - 1.5 * bar_width, in_db_data_others, bar_width, color=colors[4], hatch=hatches[4],
    #        bottom=in_db_data_query + in_db_data_copy_gpu + in_db_data_preprocess + in_db_data_compute,
    #        edgecolor='black')

    # # out-db CPU
    in_db_data_query = outcpudb_med["data_query_time"]
    in_db_data_copy_gpu = outcpudb_med["tensor_to_gpu"]
    in_db_data_preprocess = outcpudb_med["py_conver_to_tensor"]
    in_db_data_compute = outcpudb_med["py_compute"]
    in_db_data_others = outcpudb_med["overall_query_latency"] - \
                        in_db_data_query - \
                        in_db_data_copy_gpu - \
                        in_db_data_preprocess - \
                        in_db_data_compute

    ax.bar(index - 0.5 * bar_width, in_db_data_query, bar_width, color=colors[0], hatch=hatches[0],
           edgecolor='black')
    ax.bar(index - 0.5 * bar_width, in_db_data_preprocess, bar_width, color=colors[2], hatch=hatches[2],
           bottom=in_db_data_query,
           edgecolor='black')
    ax.bar(index - 0.5 * bar_width, in_db_data_copy_gpu, bar_width, color=colors[1], hatch=hatches[1],
           bottom=in_db_data_query + in_db_data_preprocess,
           edgecolor='black')
    ax.bar(index - 0.5 * bar_width, in_db_data_compute, bar_width, color=colors[3], hatch=hatches[3],
           bottom=in_db_data_query + in_db_data_copy_gpu + in_db_data_preprocess,
           edgecolor='black')
    # ax.bar(index - 0.5 * bar_width, in_db_data_others, bar_width, color=colors[4], hatch=hatches[4],
    #        bottom=in_db_data_query + in_db_data_copy_gpu + in_db_data_preprocess + in_db_data_compute,
    #        edgecolor='black')

    # Update the flags to ensure the labels are not set again in the next iterations
    set_label_outdb_gpu_data = False
    set_label_outdb_gpu_inference = False
    set_label_outdb_cpu_data = False
    set_label_outdb_cpu_inference = False
    set_label_indb_data = False
    set_label_indb_inference = False

ax.set_ylabel(".", fontsize=20, color='white')
fig.text(-0.05, 0.5, 'End-to-end Time (ms)', va='center', rotation='vertical', fontsize=20)

# ax.set_ylim(top=1600)

for sub_ax in ax.axs:
    sub_ax.set_xticks(index)
    sub_ax.set_xticklabels(datasets, rotation=0, fontsize=set_font_size)

ax.legend(fontsize=set_lgend_size - 2, ncol=2)

# Since the yaxis formatter is tricky with brokenaxes, you might need to set it for the actual underlying axes:
for ax1 in ax.axs:
    ax1.yaxis.set_major_formatter(thousands_format)

ax.tick_params(axis='y', which='major', labelsize=set_tick_size + 5)

plt.tight_layout()
fig.tight_layout()
# plt.show()
print(f"saving to ./internal/ml/model_slicing/In_DB_inference/filter_latency_memory_bar.pdf")
fig.savefig(f"./internal/ml/model_slicing/In_DB_inference/filter_latency_memory_bar.pdf",
            bbox_inches='tight')
