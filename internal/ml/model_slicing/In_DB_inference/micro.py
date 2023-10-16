import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.ticker import FuncFormatter
from brokenaxes import brokenaxes


def thousands_formatter(x, pos):
    if x >= 1e3:
        return '{:.0f}k'.format(x * 1e-3)
    else:
        return '{:.1f}'.format(x)


thousands_format = FuncFormatter(thousands_formatter)


# Helper function to load data
def load_data(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)


# Set your plot parameters
bar_width = 0.35
opacity = 0.8
set_font_size = 15  # Set the font size
set_lgend_size = 12
set_tick_size = 12
colors = ['#729ECE', '#8E44AD', '#2ECC71', '#3498DB', '#F39C12', '#E74C3C', '#2C3E50', '#27AE60', '#F1C40F', '#9B59B6']
hatches = ['/', '\\', 'x', '.', '*', '//', '\\\\', 'xx', '..', '**']


def scale_to_ms(latencies):
    result = {}
    for key, value in latencies.items():
        value = value * 1000
        result[key] = value
    return result


# here run 10k rows for inference.,
# each sub-list is "compute time" and "data fetch time"

basic_exps = {
    "out_db_cpu": {'data_query_time': 0.6033484935760498, 'py_conver_to_tensor': 2.626680612564087, 'tensor_to_gpu': 0.0002684593200683594, 'py_compute': 1.1743614673614502, 'load_model': 0.14634346961975098, 'overall_query_latency': 4.617777109146118},
}

# Collecting data for plotting
datasets_result = {
    'w/o model cache':
        {'mem_allocate_time': 0.000211036, 'data_query_time_spi': 0.086204318, 'diff': -0.00032627899999937426, 'model_init_time': 3.759357937, 'overall_query_latency': 7.791192761, 'python_compute_time': 5.947771939, 'data_query_time': 1.8352505460000001, 'py_conver_to_tensor': 3.0297083854675293, 'py_compute': 1.6506195068359375, 'py_overall_duration': 5.400642156600952, 'py_diff': 0.4203142642974854},

    '\nw/o SPI':
        {'mem_allocate_time': 0.000211036, 'data_query_time_spi': 0.6033484935760498, 'diff': -0.00032627899999937426, 'model_init_time': 0.007843997, 'overall_query_latency': 7.791192761, 'python_compute_time': 5.947771939, 'data_query_time': 1.8352505460000001, 'py_conver_to_tensor': 3.0297083854675293, 'py_compute': 1.6506195068359375, 'py_overall_duration': 5.400642156600952, 'py_diff': 0.4203142642974854},

    'w/o share memory':
        {'model_init_time': 0.008697525, 'data_query_time_spi': 0.069067049, 'data_query_time': 2.915582513, 'python_compute_time': 5.020754292, 'overall_query_latency': 7.945128665, 'diff': -9.433500000000095e-05, 'py_conver_to_tensor': 3.0656007289886475, 'py_compute': 1.1661615371704102, 'py_overall_duration': 2.560455799102783, 'py_diff': 0.4286935329437256},

    '\nw/ all optims':
        {'mem_allocate_time': 0.000211036, 'data_query_time_spi': 0.086204318, 'diff': -0.00032627899999937426, 'model_init_time': 0.007843997, 'overall_query_latency': 7.791192761, 'python_compute_time': 5.947771939, 'data_query_time': 1.8352505460000001, 'py_conver_to_tensor': 3.0297083854675293, 'py_compute': 1.6506195068359375, 'py_overall_duration': 5.400642156600952, 'py_diff': 0.4203142642974854}
}

datasets = list(datasets_result.keys())

# Plotting
fig, ax = plt.subplots(figsize=(6.4, 4.5))

# Initial flags to determine whether the labels have been set before
set_label_in_db_model_load = True
set_label_in_db_data_query = True
set_label_in_db_data_copy_start_py = True
set_label_in_db_data_preprocess = True
set_label_in_db_data_compute = True
set_label_in_db_data_others = True

indices = []
index = 0
for dataset, valuedic in datasets_result.items():
    indices.append(index)
    indb_med_opt = scale_to_ms(valuedic)

    # set labesl
    label_in_db_model_load = 'Model Loading' if set_label_in_db_model_load else None
    label_in_db_data_query = 'Data Retrieval' if set_label_in_db_data_query else None
    label_in_db_data_copy_start_py = 'Data Copy' if set_label_in_db_data_copy_start_py else None
    label_in_db_data_preprocess = 'Data Preprocessing & Inference' if set_label_in_db_data_preprocess else None
    label_in_db_data_compute = 'Inference' if set_label_in_db_data_compute else None
    label_in_db_data_others = 'Others' if set_label_in_db_data_others else None

    # in-db with optimizization
    in_db_data_model_load = indb_med_opt["model_init_time"]
    # wrapper time - python inner compute time
    in_db_data_query = indb_med_opt["data_query_time_spi"]
    in_db_data_copy_start_py = indb_med_opt["python_compute_time"] - indb_med_opt["py_overall_duration"]
    in_db_data_preprocess = indb_med_opt["py_conver_to_tensor"]
    in_db_data_compute = indb_med_opt["py_compute"]
    in_db_data_others = indb_med_opt["py_diff"]

    ax.bar(index, in_db_data_model_load, bar_width, color=colors[0], hatch=hatches[0], zorder=2,
           label=label_in_db_model_load, edgecolor='black')

    ax.bar(index, in_db_data_query, bar_width, color=colors[1], hatch=hatches[1], zorder=2,
           bottom=in_db_data_model_load,
           label=label_in_db_data_query, edgecolor='black')

    ax.bar(index, in_db_data_copy_start_py, bar_width, color=colors[2], hatch=hatches[2], zorder = 2,
           bottom=in_db_data_query+in_db_data_model_load,
           label = label_in_db_data_copy_start_py,
           edgecolor='black')

    ax.bar(index, in_db_data_preprocess + in_db_data_compute, bar_width, color=colors[3], hatch=hatches[3], zorder=2,
           bottom=in_db_data_query + in_db_data_copy_start_py+in_db_data_model_load,
           label=label_in_db_data_preprocess, edgecolor='black')

    # ax.bar(index, in_db_data_compute, bar_width, color=colors[4], hatch=hatches[4], zorder=2,
    #        bottom=in_db_data_query + in_db_data_copy_start_py + in_db_data_preprocess+in_db_data_model_load,
    #        label=label_in_db_data_compute, edgecolor='black')

    ax.bar(index, in_db_data_others, bar_width, color=colors[5], hatch=hatches[5], zorder = 2,
           bottom=in_db_data_query + in_db_data_copy_start_py + in_db_data_preprocess + in_db_data_compute+in_db_data_model_load,
           label=label_in_db_data_others,
           edgecolor='black')

    # Update the flags to ensure the labels are not set again in the next iterations
    set_label_in_db_model_load = False
    set_label_in_db_data_query = False
    set_label_in_db_data_copy_start_py = False
    set_label_in_db_data_preprocess = False
    set_label_in_db_data_compute = False
    set_label_in_db_data_others = False

    index += 1

# legned etc
ax.set_ylabel(".", fontsize=20, color='white')
fig.text(0.01, 0.5, 'End-to-end Time (ms)', va='center', rotation='vertical', fontsize=20)

ax.set_ylim(top=13000)

ax.set_xticks(indices)
ax.set_xticklabels(datasets, rotation=-0, fontsize=set_font_size)


# ax.legend(fontsize=set_lgend_size - 2, ncol=2, )
ax.legend(fontsize=set_lgend_size, ncol=2, loc='upper left')

# Since the yaxis formatter is tricky with brokenaxes, you might need to set it for the actual underlying axes:
ax.yaxis.set_major_formatter(thousands_format)

ax.tick_params(axis='y', which='major', labelsize=set_tick_size + 5)

ax.grid(True, zorder=1)  # grid in front of bars

plt.tight_layout()
fig.tight_layout()
# plt.show()
print(f"saving to ./internal/ml/model_slicing/In_DB_inference/micro.pdf")
fig.savefig(f"./internal/ml/model_slicing/In_DB_inference/micro.pdf",
            bbox_inches='tight')
