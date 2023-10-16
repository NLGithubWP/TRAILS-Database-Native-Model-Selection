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
colors = ['#729ECE','#8E44AD',  '#2ECC71',  '#3498DB','#F39C12']
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
datasets_result = {


    'Adult': {
        'In-Db-opt': {'data_query_time': 1.983637511, 'python_compute_time': 5.668429231,
                      'model_init_time': 0.008373906, 'overall_query_latency': 7.660767723,
                      'diff': -0.0003270750000003986, 'mem_allocate_time': 0.00022936,
                      'data_query_time_spi': 0.104852828, 'py_conver_to_tensor': 3.1869658946990967,
                      'py_compute': 0.6434091854095459, 'py_overall_duration': 5.178176403045654,
                      'py_diff': 0.7278013229370117},
        'out-DB-cpu': {'data_query_time': 0.8945093154907227, 'py_conver_to_tensor': 3.1391489505767822,
                       'tensor_to_gpu': 0.000179290771484375, 'py_compute': 0.6464982032775879,
                       'overall_query_latency': 5.005170583724976},
        'out-DB-gpu': {'data_query_time': 0.8986874103546143, 'py_conver_to_tensor': 3.015498638153076,
                       'tensor_to_gpu': 0.013033390045166016, 'py_compute': 0.024132966995239258,
                       'overall_query_latency': 4.305214881896973},
    },

    'Disease': {
        'In-Db-opt': {'mem_allocate_time': 0.000241846, 'data_query_time_spi': 0.092643221,
                      'python_compute_time': 4.456881872, 'overall_query_latency': 7.531777533,
                      'data_query_time': 3.067697677, 'diff': -0.0003152109999993158, 'model_init_time': 0.006882773,
                      'py_conver_to_tensor': 2.6528313159942627, 'py_compute': 0.7840120792388916,
                      'py_overall_duration': 4.027993440628052, 'py_diff': 0.5911500453948975},

        'out-DB-cpu': {'data_query_time': 0.7599310874938965, 'py_conver_to_tensor': 2.712991952896118,
                       'tensor_to_gpu': 0.0004315376281738281, 'py_compute': 0.7755249500274658,
                       'overall_query_latency': 5.472174644470215},
        'out-DB-gpu': {'data_query_time': 0.7506480598449707, 'py_conver_to_tensor': 2.705919427871704,
                       'tensor_to_gpu': 0.007371664047241211, 'py_compute': 0.028490304946899414,
                       'overall_query_latency': 3.281588554382324},
    },

    'Bank': {
        'In-Db-opt': {'data_query_time': 3.9064207829999997, 'python_compute_time': 4.978618743,
                      'data_query_time_spi': 0.115038494, 'mem_allocate_time': 0.000246575,
                      'overall_query_latency': 8.893539688, 'diff': -0.0003386900000013071,
                      'model_init_time': 0.008161472, 'py_conver_to_tensor': 2.878143072128296,
                      'py_compute': 0.8705038928985596, 'py_overall_duration': 4.330329895019531,
                      'py_diff': 0.8216829299926758},
        'out-DB-cpu': {'data_query_time': 0.9924757480621338, 'py_conver_to_tensor': 2.880948085784912,
                       'tensor_to_gpu': 0.00011372566223144531, 'py_compute': 0.8873722553253174,
                       'overall_query_latency': 5.279063701629639},
        'out-DB-gpu': {'data_query_time': 0.9962258052825928, 'py_conver_to_tensor': 2.896128273010254,
                       'tensor_to_gpu': 0.010221481323242188, 'py_compute': 0.021872520446777344,
                       'overall_query_latency': 4.37221097946167},
    },

    'AppRec': {
        'In-Db-opt':
            {'model_init_time': 0.007445235, 'mem_allocate_time': 0.000228569, 'data_query_time': 1.8191302120000001,
             'python_compute_time': 5.120315227, 'data_query_time_spi': 0.100664045,
             'overall_query_latency': 6.947206643, 'diff': -0.0003159689999998605,
             'py_conver_to_tensor': 2.4618063831329346, 'py_compute': 0.938471794128418,
             'py_overall_duration': 4.697931289672852, 'py_diff': 0.637653112411499},

        'out-DB-cpu':
            {'data_query_time': 0.7075839042663574, 'py_conver_to_tensor': 2.4676432609558105,
             'tensor_to_gpu': 0.0003657341003417969, 'py_compute': 0.9474043369293213,
             'overall_query_latency': 5.193724870681763},

        'out-DB-gpu':
            {'data_query_time': 0.6995127868652344, 'py_conver_to_tensor': 2.3515465259552,
             'tensor_to_gpu': 0.007832050323486328, 'py_compute': 0.03111410140991211,
             'overall_query_latency': 3.233276128768921},
    },
}

datasets = list(datasets_result.keys())

# Plotting
fig, ax = plt.subplots(figsize=(6.4, 4.5))

# Initial flags to determine whether the labels have been set before
set_label_our_system = True
set_label_baseline_system = True


indices = []
index = 0
for dataset, valuedic in datasets_result.items():
    indices.append(index)

    indb_med_opt = scale_to_ms(valuedic["In-Db-opt"])
    outgpudb_med = scale_to_ms(valuedic["out-DB-gpu"])
    outcpudb_med = scale_to_ms(valuedic["out-DB-cpu"])

    # set labesl
    label_our_system = 'INDICES' if set_label_our_system else None
    label_baseline_system = 'INDICES (seperate)' if set_label_baseline_system else None

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

    T1 = in_db_data_query + in_db_data_copy_gpu + in_db_data_preprocess + in_db_data_compute
    ax.bar(index - bar_width/2, in_db_data_query + in_db_data_copy_gpu + in_db_data_preprocess + in_db_data_compute,
           bar_width, color=colors[2], hatch=hatches[2], zorder=2,
           label=label_baseline_system,
           edgecolor='black')


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

    T2 = in_db_data_query + in_db_data_copy_gpu + in_db_data_preprocess + in_db_data_compute
    ax.bar(index + bar_width/2, in_db_data_query + in_db_data_copy_start_py + in_db_data_preprocess+in_db_data_compute,
           bar_width, color=colors[1], hatch=hatches[1], zorder=2,
           label=label_our_system, edgecolor='black')

    # Update the flags to ensure the labels are not set again in the next iterations
    set_label_our_system = False
    set_label_baseline_system = False

    index += 1

    print(dataset, T1 / T2)




# legned etc
ax.set_ylabel(".", fontsize=20, color='white')
fig.text(0.01, 0.5, 'End-to-end Time (ms)', va='center', rotation='vertical', fontsize=20)

ax.set_ylim(top=6000)

ax.set_xticks(indices)
ax.set_xticklabels(datasets, rotation=0, fontsize=set_font_size)

# ax.legend(fontsize=set_lgend_size - 2, ncol=2, )
ax.legend(fontsize=15, ncol=2, loc='upper left')

# Since the yaxis formatter is tricky with brokenaxes, you might need to set it for the actual underlying axes:
ax.yaxis.set_major_formatter(thousands_format)

ax.tick_params(axis='y', which='major', labelsize=set_tick_size + 5)

ax.grid(True, zorder=1)  # grid in front of bars

plt.tight_layout()
fig.tight_layout()
# plt.show()
print(f"saving to ./internal/ml/model_slicing/In_DB_inference/macro_sys.pdf")
fig.savefig(f"./internal/ml/model_slicing/In_DB_inference/macro_sys.pdf",
            bbox_inches='tight')
