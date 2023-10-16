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

# Collecting data for plotting
datasets_result = {
    'Frappe': {
        'In-Db-opt':
            {'diff': -0.0003339110000002421, 'data_query_time': 2.956214174, 'model_init_time': 0.00774435,
             'data_query_time_spi': 0.101406308, 'python_compute_time': 4.419487975, 'mem_allocate_time': 0.000231498,
             'overall_query_latency': 7.38378041, 'py_conver_to_tensor': 2.469083070755005,
             'py_compute': 0.3810811138153076, 'py_overall_duration': 3.9879794120788574,
             'py_diff': 0.6478152275085449},

        'out-DB-cpu':
            {'data_query_time': 1.1919968128204346, 'py_conver_to_tensor': 2.202195644378662,
             'tensor_to_gpu': 0.0030956268310546875, 'py_compute': 1.521008014678955,
             'overall_query_latency': 5.124648094177246},

        'out-DB-gpu':
            {'data_query_time': 1.2131133079528809, 'py_conver_to_tensor': 2.171276569366455,
             'tensor_to_gpu': 0.020495891571044922, 'py_compute': 0.38417649269104004,
             'overall_query_latency': 4.07839560508728},
    },

    'Adult': {
        'In-Db-opt': {'data_query_time': 3.571276695, 'python_compute_time': 5.337580922,
                      'overall_query_latency': 8.915918578, 'diff': -0.00033833100000002503,
                      'mem_allocate_time': 0.000255105, 'model_init_time': 0.00672263,
                      'data_query_time_spi': 0.089652013, 'py_conver_to_tensor': 3.4604952335357666,
                      'py_compute': 0.7244682312011719, 'py_overall_duration': 4.87972092628479,
                      'py_diff': 0.6947574615478516},
        'out-DB-cpu': {'data_query_time': 0.6915404796600342, 'py_conver_to_tensor': 3.29960560798645,
                       'tensor_to_gpu': 0.00020813941955566406, 'py_compute': 0.4946920871734619,
                       'overall_query_latency': 4.719264030456543},
        'out-DB-gpu': {'data_query_time': 0.09808635711669922, 'py_conver_to_tensor': 0.27641721725463867,
                       'tensor_to_gpu': 11.821438789367676, 'py_compute': 0.011312007904052734,
                       'overall_query_latency': 12.34174108505249},
    },

    'Cvd': {
        'In-Db-opt': {'diff': -0.00035345200000058696, 'data_query_time': 3.4941312,
                      'overall_query_latency': 7.492815236, 'data_query_time_spi': 0.104308665,
                      'model_init_time': 0.008329788, 'python_compute_time': 3.990000796,
                      'mem_allocate_time': 0.000259831, 'py_conver_to_tensor': 2.135631799697876,
                      'py_compute': 0.8641166687011719, 'py_overall_duration': 3.4981813430786133,
                      'py_diff': 0.49843287467956543},

        'out-DB-cpu': {'data_query_time': 0.849153995513916, 'py_conver_to_tensor': 2.557361364364624,
                       'tensor_to_gpu': 0.0003402233123779297, 'py_compute': 0.6930482387542725,
                       'overall_query_latency': 4.350241422653198},
        'out-DB-gpu': {'data_query_time': 0.1045682430267334, 'py_conver_to_tensor': 0.2194075584411621,
                       'tensor_to_gpu': 19.62585687637329, 'py_compute': 0.052184343338012695,
                       'overall_query_latency': 20.205774307250977},
    },

    'Bank': {
        'In-Db-opt': {'data_query_time': 3.9064207829999997, 'python_compute_time': 4.978618743,
                      'data_query_time_spi': 0.115038494, 'mem_allocate_time': 0.000246575,
                      'overall_query_latency': 8.893539688, 'diff': -0.0003386900000013071,
                      'model_init_time': 0.008161472, 'py_conver_to_tensor': 2.878143072128296,
                      'py_compute': 0.6305038928985596, 'py_overall_duration': 4.330329895019531,
                      'py_diff': 0.8216829299926758},
        'out-DB-cpu': {'data_query_time': 1.0723905563354492, 'py_conver_to_tensor': 3.5848896503448486,
                       'tensor_to_gpu': 0.0002608299255371094, 'py_compute': 0.8267972469329834,
                       'overall_query_latency': 5.825712203979492},
        'out-DB-gpu': {'data_query_time': 0.1385364532470703, 'py_conver_to_tensor': 0.29821038246154785,
                       'tensor_to_gpu': 20.349432706832886, 'py_compute': 0.045266151428222656,
                       'overall_query_latency': 21.015833854675293},
    },
}

datasets = list(datasets_result.keys())

# Plotting
fig = plt.figure(figsize=(6.4, 4.5))

# Create a broken y-axis within the fig
ax = brokenaxes(ylims=((0, 700), (12150, 12260), (19920, 20200), (20750, 21000)), hspace=.25, fig=fig, d=0)

# Initial flags to determine whether the labels have been set before
set_label_in_db_data_query = True
set_label_in_db_data_copy_start_py = True
set_label_in_db_data_preprocess = True
set_label_in_db_data_compute = True
set_label_in_db_data_others = True

indices = []
index = 0
for dataset, valuedic in datasets_result.items():
    indices.append(index)

    indb_med_opt = scale_to_ms(valuedic["In-Db-opt"])
    outgpudb_med = scale_to_ms(valuedic["out-DB-gpu"])
    outcpudb_med = scale_to_ms(valuedic["out-DB-cpu"])

    # set labesl
    label_in_db_data_query = 'Data Retrievl' if set_label_in_db_data_query else None
    label_in_db_data_copy_start_py = 'Data Copy' if set_label_in_db_data_copy_start_py else None
    label_in_db_data_preprocess = 'Data Preprocess' if set_label_in_db_data_preprocess else None
    label_in_db_data_compute = 'Model Inference' if set_label_in_db_data_compute else None
    label_in_db_data_others = 'Others' if set_label_in_db_data_others else None

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

    ax.bar(index + bar_width, in_db_data_query, bar_width, color=colors[0], hatch=hatches[0], zorder=2,
           label=label_in_db_data_query, edgecolor='black')
    # ax.bar(index, in_db_data_copy_start_py, bar_width, color=colors[1], hatch=hatches[1], zorder = 2,
    #        bottom=in_db_data_query,
    #        edgecolor='black')
    ax.bar(index + bar_width, in_db_data_preprocess, bar_width, color=colors[2], hatch=hatches[2], zorder=2,
           bottom=in_db_data_query + in_db_data_copy_start_py,
           label=label_in_db_data_preprocess, edgecolor='black')
    ax.bar(index + bar_width, in_db_data_compute, bar_width, color=colors[3], hatch=hatches[3], zorder=2,
           bottom=in_db_data_query + in_db_data_copy_start_py + in_db_data_preprocess,
           label=label_in_db_data_compute, edgecolor='black')
    # ax.bar(index + bar_width, in_db_data_others, bar_width, color=colors[4], hatch=hatches[4], zorder = 2,
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

    ax.bar(index - bar_width, in_db_data_query, bar_width, color=colors[0], hatch=hatches[0], zorder=2,
           edgecolor='black')
    ax.bar(index - bar_width, in_db_data_preprocess, bar_width, color=colors[2], hatch=hatches[2], zorder=2,
           bottom=in_db_data_query,
           edgecolor='black')
    ax.bar(index - bar_width, in_db_data_copy_gpu, bar_width, color=colors[1], hatch=hatches[1], zorder=2,
           bottom=in_db_data_query + in_db_data_preprocess,
           label=label_in_db_data_copy_start_py,
           edgecolor='black')
    ax.bar(index - bar_width, in_db_data_compute, bar_width, color=colors[3], hatch=hatches[3], zorder=2,
           bottom=in_db_data_query + in_db_data_copy_gpu + in_db_data_preprocess,
           edgecolor='black')
    # ax.bar(index -  bar_width, in_db_data_others, bar_width, color=colors[4], hatch=hatches[4], zorder = 2,
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

    ax.bar(index, in_db_data_query, bar_width, color=colors[0], hatch=hatches[0], zorder=2,
           edgecolor='black')
    ax.bar(index, in_db_data_preprocess, bar_width, color=colors[2], hatch=hatches[2], zorder=2,
           bottom=in_db_data_query,
           edgecolor='black')
    ax.bar(index, in_db_data_copy_gpu, bar_width, color=colors[1], hatch=hatches[1], zorder=2,
           bottom=in_db_data_query + in_db_data_preprocess,
           edgecolor='black')
    ax.bar(index, in_db_data_compute, bar_width, color=colors[3], hatch=hatches[3], zorder=2,
           bottom=in_db_data_query + in_db_data_copy_gpu + in_db_data_preprocess,
           edgecolor='black')
    # ax.bar(index , in_db_data_others, bar_width, color=colors[4], hatch=hatches[4], zorder = 2,
    #        bottom=in_db_data_query + in_db_data_copy_gpu + in_db_data_preprocess + in_db_data_compute,
    #        edgecolor='black')

    # Update the flags to ensure the labels are not set again in the next iterations
    set_label_in_db_data_query = False
    set_label_in_db_data_copy_start_py = False
    set_label_in_db_data_preprocess = False
    set_label_in_db_data_compute = False
    set_label_in_db_data_others = False

    index += 1

# legned etc
ax.set_ylabel(".", fontsize=20, color='white')
fig.text(-0.05, 0.5, 'End-to-end Time (ms)', va='center', rotation='vertical', fontsize=20)

# ax.set_ylim(top=1600)

for sub_ax in ax.axs:
    sub_ax.set_xticks(indices)
    sub_ax.set_xticklabels(datasets, rotation=0, fontsize=set_font_size)

# ax.legend(fontsize=set_lgend_size - 2, ncol=2, )
ax.legend(fontsize=set_lgend_size - 2, ncol=2, loc='upper left')

# Since the yaxis formatter is tricky with brokenaxes, you might need to set it for the actual underlying axes:
for ax1 in ax.axs:
    ax1.yaxis.set_major_formatter(thousands_format)

ax.tick_params(axis='y', which='major', labelsize=set_tick_size + 5)

ax.grid(True, zorder=1)  # grid in front of bars

plt.tight_layout()
fig.tight_layout()
# plt.show()
print(f"saving to ./internal/ml/model_slicing/In_DB_inference/macro.pdf")
fig.savefig(f"./internal/ml/model_slicing/In_DB_inference/macro.pdf",
            bbox_inches='tight')
