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
colors = ['#729ECE', '#8E44AD', '#2ECC71', '#3498DB', '#F39C12']
hatches = ['/', '\\', 'x', '.', '*', '//', '\\\\', 'xx', '..', '**']


def scale_to_ms(latencies):
    result = {}
    for key, value in latencies.items():
        value = value * 1000
        result[key] = value
    return result


# here run 10k rows for inference.,
# each sub-list is "compute time" and "data fetch time"

# Collecting data for plotting
frappe_datasets_result = {
    "20k": {
        'out-DB-cpu': {'data_query_time': 0.12000298500061035, 'py_conver_to_tensor': 0.48304200172424316,
                       'tensor_to_gpu': 0.00011420249938964844, 'py_compute': 0.155472993850708,
                       'overall_query_latency': 0.8322274684906006},
        'In-Db-opt': {'model_init_time': 0.005696283, 'data_query_time_spi': 0.018192791,
                      'data_query_time': 0.326063297,
                      'python_compute_time': 1.009288424, 'mem_allocate_time': 0.000129372,
                      'overall_query_latency': 1.341232299, 'diff': -0.00018429500000016752,
                      'py_conver_to_tensor': 0.4896477890014648, 'py_compute': 0.15789603233337402,
                      'py_overall_duration': 0.9268307685852051, 'py_diff': 0.1892869472503662},
    },
    "40k": {'out-DB-cpu': {'data_query_time': 0.24573016166687012, 'py_conver_to_tensor': 1.0262854099273682,
                           'tensor_to_gpu': 7.987022399902344e-05, 'py_compute': 0.23546600341796875,
                           'overall_query_latency': 1.6435070037841797},
            'In-Db-opt': {'overall_query_latency': 3.612851412, 'data_query_time_spi': 0.020054077,
                          'model_init_time': 0.002833754, 'python_compute_time': 1.856035929,
                          'mem_allocate_time': 0.000106813, 'diff': -0.00014507599999991072,
                          'data_query_time': 1.753836653,
                          'py_conver_to_tensor': 1.0434872150421143, 'py_compute': 0.23771233558654785,
                          'py_overall_duration': 1.694864273071289, 'py_diff': 0.26366472244262695}, },
    "80k": {'out-DB-cpu': {'data_query_time': 0.5709426403045654, 'py_conver_to_tensor': 1.9019184112548828,
                           'tensor_to_gpu': 0.00023603439331054688, 'py_compute': 0.6903989315032959,
                           'overall_query_latency': 3.3528354167938232},
            'In-Db-opt': {'data_query_time_spi': 0.053391186, 'model_init_time': 0.009251984,
                          'overall_query_latency': 7.577453163, 'mem_allocate_time': 0.000168978,
                          'diff': -0.00024827600000065786, 'data_query_time': 3.580180891,
                          'python_compute_time': 3.9877720119999998, 'py_conver_to_tensor': 1.9604360103607178,
                          'py_compute': 0.6935163307189941, 'py_overall_duration': 3.680163860321045,
                          'py_diff': 0.506211519241333}, },
    "160k": {'out-DB-cpu': {'data_query_time': 1.068756341934204, 'py_conver_to_tensor': 3.675976800918579,
                            'tensor_to_gpu': 0.00016045570373535156, 'py_compute': 1.2924180030822754,
                            'overall_query_latency': 7.511447191238403},
             'In-Db-opt': {'python_compute_time': 6.676874894, 'overall_query_latency': 11.829161511,
                           'diff': -0.00031599099999901625, 'model_init_time': 0.006576756,
                           'data_query_time_spi': 0.153169837, 'mem_allocate_time': 0.000238081,
                           'data_query_time': 5.14539387, 'py_conver_to_tensor': 3.685814380645752,
                           'py_compute': 1.2878854274749756, 'py_overall_duration': 6.1088738441467285,
                           'py_diff': 1.135174036026001}, },
    "320k": {'out-DB-cpu': {'data_query_time': 2.4662599563598633, 'py_conver_to_tensor': 9.113554000854492,
                            'tensor_to_gpu': 0.00025653839111328125, 'py_compute': 2.673182916641235,
                            'overall_query_latency': 18.244637727737427},
             'In-Db-opt': {'model_init_time': 0.008054549, 'python_compute_time': 19.408834322, 'overall_query_latency': 27.048698701,
                           'data_query_time': 7.631448309, 'diff': -0.0003615209999985325, 'data_query_time_spi': 0.252865996,
                           'mem_allocate_time': 0.000264806, 'py_conver_to_tensor': 9.142578220367432, 'py_compute': 2.6727094650268555,
                           'py_overall_duration': 17.520897150039673, 'py_diff': 3.3056094646453857},
             },

}

datasets = list(frappe_datasets_result.keys())

# Plotting
fig, ax = plt.subplots(figsize=(6.4, 4.5))

# Initial flags to determine whether the labels have been set before
set_label_in_db_data_query = True
set_label_in_db_data_copy_start_py = True
set_label_in_db_data_preprocess = True
set_label_in_db_data_compute = True
set_label_in_db_data_others = True


baseline_sys_x_array = []
baseline_sys_y_array = []

sams_sys_x_array = []
sams_sys_y_array = []

indices = []
index = 0
for dataset, valuedic in frappe_datasets_result.items():
    indices.append(index)

    indb_med_opt = scale_to_ms(valuedic["In-Db-opt"])
    outcpudb_med = scale_to_ms(valuedic["out-DB-cpu"])

    # set labesl
    label_in_db_data_query = 'Data Retrieval' if set_label_in_db_data_query else None
    label_in_db_data_copy_start_py = 'Data Copy' if set_label_in_db_data_copy_start_py else None
    label_in_db_data_preprocess = 'Data Preprocess' if set_label_in_db_data_preprocess else None
    label_in_db_data_compute = 'Model Inference' if set_label_in_db_data_compute else None
    label_in_db_data_others = 'Others' if set_label_in_db_data_others else None

    # in-db with optimizization
    in_db_data_copy_start_py = 0
    in_db_data_query = indb_med_opt["data_query_time_spi"]
    in_db_data_preprocess = indb_med_opt["py_conver_to_tensor"]
                            # + indb_med_opt["python_compute_time"] \
                            # - indb_med_opt["py_overall_duration"]
    in_db_data_compute = indb_med_opt["py_compute"]
    in_db_data_others = indb_med_opt["overall_query_latency"] - \
                        indb_med_opt["data_query_time"] - \
                        in_db_data_copy_start_py - \
                        in_db_data_preprocess - \
                        in_db_data_compute

    ax.bar(index + bar_width / 2, in_db_data_query, bar_width, color=colors[0], hatch=hatches[0], zorder=2,
           label=label_in_db_data_query, edgecolor='black')
    # ax.bar(index, in_db_data_copy_start_py, bar_width, color=colors[1], hatch=hatches[1], zorder = 2,
    #        bottom=in_db_data_query,
    #        edgecolor='black')
    ax.bar(index + bar_width / 2, in_db_data_preprocess, bar_width, color=colors[2], hatch=hatches[2], zorder=2,
           bottom=in_db_data_query + in_db_data_copy_start_py,
           label=label_in_db_data_preprocess, edgecolor='black')
    ax.bar(index + bar_width / 2, in_db_data_compute, bar_width, color=colors[3], hatch=hatches[3], zorder=2,
           bottom=in_db_data_query + in_db_data_copy_start_py + in_db_data_preprocess,
           label=label_in_db_data_compute, edgecolor='black')
    # ax.bar(index + bar_width, in_db_data_others, bar_width, color=colors[4], hatch=hatches[4], zorder = 2,
    #        bottom=in_db_data_query + in_db_data_copy_start_py + in_db_data_preprocess + in_db_data_compute,
    #        edgecolor='black')

    sams_sys_x_array.append(index + bar_width / 2)
    sams_sys_y_array.append(
        in_db_data_query + in_db_data_copy_start_py + in_db_data_preprocess + in_db_data_compute)

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

    ax.bar(index - bar_width / 2, in_db_data_query, bar_width, color=colors[0], hatch=hatches[0], zorder=2,
           edgecolor='black')
    ax.bar(index - bar_width / 2, in_db_data_preprocess, bar_width, color=colors[2], hatch=hatches[2], zorder=2,
           bottom=in_db_data_query,
           edgecolor='black')
    ax.bar(index - bar_width / 2, in_db_data_copy_gpu, bar_width, color=colors[1], hatch=hatches[1], zorder=2,
           bottom=in_db_data_query + in_db_data_preprocess,
           edgecolor='black')
    ax.bar(index - bar_width / 2, in_db_data_compute, bar_width, color=colors[3], hatch=hatches[3], zorder=2,
           bottom=in_db_data_query + in_db_data_copy_gpu + in_db_data_preprocess,
           edgecolor='black')
    # ax.bar(index , in_db_data_others, bar_width, color=colors[4], hatch=hatches[4], zorder = 2,
    #        bottom=in_db_data_query + in_db_data_copy_gpu + in_db_data_preprocess + in_db_data_compute,
    #        edgecolor='black')

    baseline_sys_x_array.append(index - bar_width / 2)
    baseline_sys_y_array.append(
        in_db_data_query + in_db_data_copy_gpu + in_db_data_preprocess + in_db_data_compute)

    # Update the flags to ensure the labels are not set again in the next iterations
    set_label_in_db_data_query = False
    set_label_in_db_data_copy_start_py = False
    set_label_in_db_data_preprocess = False
    set_label_in_db_data_compute = False
    set_label_in_db_data_others = False

    index += 1


ax.plot(sams_sys_x_array, sams_sys_y_array, color='red', marker='*', linewidth=2)  # 'o' will add a marker at each point
ax.plot(baseline_sys_x_array, baseline_sys_y_array, color='green', marker='o',linewidth = 2)  # 'o' will add a marker at each point


# legned etc
ax.set_ylabel(".", fontsize=20, color='white')
fig.text(0.01, 0.5, 'End-to-end Time (ms)', va='center', rotation='vertical', fontsize=20)

ax.set_ylim(top=14400)

ax.set_xticks(indices)
ax.set_xticklabels(datasets, rotation=0, fontsize=set_font_size)

# ax.legend(fontsize=set_lgend_size - 2, ncol=2, )
ax.legend(fontsize=set_lgend_size, ncol=1, loc='upper left')

# Since the yaxis formatter is tricky with brokenaxes, you might need to set it for the actual underlying axes:
ax.yaxis.set_major_formatter(thousands_format)

ax.tick_params(axis='y', which='major', labelsize=set_tick_size + 5)

ax.grid(True, zorder=1)  # grid in front of bars

plt.tight_layout()
fig.tight_layout()
# plt.show()
print(f"saving to ./internal/ml/model_slicing/In_DB_inference/macro_data_scale.pdf")
fig.savefig(f"./internal/ml/model_slicing/In_DB_inference/macro_data_scale.pdf",
            bbox_inches='tight')
