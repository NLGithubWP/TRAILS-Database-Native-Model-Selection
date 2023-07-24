
from internal.common.constant import Config
from query_api.query_model_gt_acc_api import GTMLP


def measure_total_time_usage(N, K, t1, t2, dataset):
    U = 1
    phase1_time = N * t1
    if dataset == "uci":
        # only 1 epoch since fully training is 1 epoch only
        phase2_time = K * U * t2
    else:
        phase2_time = K * U * t2 * 2
    return phase1_time, phase2_time


def get_data_loading_time(dataset):
    if dataset == Config.Frappe:
        t_one_ite = 0.07835125923156738
        t_all = 5.960570335388184
    elif dataset == Config.Criteo:
        t_one_ite = 12.259164810180664
        t_all = 1814.5491938591003
    elif dataset == Config.UCIDataset:
        t_one_ite = 0.11569786071777344
        t_all = 4.2008748054504395
    else:
        raise
    return t_one_ite, t_all


t1 = GTMLP.get_score_one_model_time(Config.Criteo, "cpu")
t2 = GTMLP.get_train_one_epoch_time(Config.Criteo, "cuda:1")
result = {}
print("\n-----------------------------\n")
for nk_pair in [[500, 5], [1000, 10], [2000, 20], [4000, 40]]:
    if str(nk_pair) not in result:
        result[str(nk_pair)] = []
    N, K = nk_pair[0], nk_pair[1]
    s1, s2 = measure_total_time_usage(N, K, t1, t2, Config.Criteo)
    data_load_time_one_ite, data_load_time_one_all = get_data_loading_time(Config.Criteo)

    result[str(nk_pair)].append(s1 + s2 + data_load_time_one_ite)
    result[str(nk_pair)].append(s1 + s2 + data_load_time_one_all)
print(result)

import numpy as np
import matplotlib.pyplot as plt

# lines' mark size
set_marker_size = 10
# points' mark size
set_marker_point = 14
# points' mark size
set_font_size = 25
set_lgend_size = 15
set_tick_size = 25
frontinsidebox = 23

time_usage = {}
datasets = [[500, 5], [1000, 10], [2000, 20], [4000, 40]]
for dataset in datasets:
    time_usage[str(dataset)] = result[str(dataset)]


num_datasets = len(datasets)
num_bars = 4  # CPU-only, GPU-only, Hybrid

bar_width = 0.25
opacity = 0.8

index = np.arange(num_datasets)

fig, ax = plt.subplots(figsize=(6.4, 4.5))
# ax.grid()

colors = ['#729ECE', '#FFB579', '#98DF8A', "#D1A7DC"]  # Softer colors #FF7F7F
hatches = ['/', '\\',  'x', 'o', 'O', '.', '*']

# Set the font size
fontsize = set_font_size

# Plot bars for external dataloader
rects1 = ax.bar(index - bar_width, [time_usage[str(dataset)][0] for dataset in datasets], bar_width,
                alpha=opacity, color=colors[0], hatch=hatches[0], edgecolor='black', label='TRAILS')

# Plot bars for IDMS
rects2 = ax.bar(index, [time_usage[str(dataset)][1] for dataset in datasets], bar_width,
                alpha=opacity, color=colors[1], hatch=hatches[1], edgecolor='black', label='TRAILS w/o RDBMS')


ax.set_ylabel('Latency (s)', fontsize=fontsize)
# ax.set_xlabel('Explored Models $N$', fontsize=fontsize)
ax.set_xticks(index)
ax.set_xticklabels(["500", "1k", "2k", "4k"], fontsize=fontsize)

# linear', 'log', 'symlog', 'logit', 'function', 'functionlog'
ax.set_yscale('linear')  # Set y-axis to logarithmic scale

# ax.set_ylim(ymax=10**3*7)

yticks_positions = [1, 5000, 10000, 15000]
yticks_labels = ['1', '5k', '10k', '15k']
plt.yticks(yticks_positions, yticks_labels)

# ax.legend(fontsize=fontsize)

# Set the font size for tick labels
ax.tick_params(axis='both', labelsize=fontsize)
fig.tight_layout()
plt.tight_layout()
plt.grid()

def export_legend(ori_fig, filename="any_time_legend", colnum=9, unique_labels=None):
    if unique_labels is None:
        unique_labels = []
    fig2 = plt.figure(figsize=(5, 0.3))
    lines_labels = [ax.get_legend_handles_labels() for ax in ori_fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # grab unique labels
    if len(unique_labels) == 0:
        unique_labels = set(labels)
    # assign labels and legends in dict
    legend_dict = dict(zip(labels, lines))
    # query dict based on unique labels
    unique_lines = [legend_dict[x] for x in unique_labels]
    fig2.legend(unique_lines, unique_labels, loc='center',
                ncol=colnum,
                fancybox=True,
                shadow=True, scatterpoints=1, fontsize=set_lgend_size)
    fig2.tight_layout()
    fig2.savefig(f"{filename}.pdf", bbox_inches='tight')

export_legend(ori_fig=fig, colnum=5)

fig.savefig("./exps/main_v2/analysis/IDMS_latency_workloads.pdf", bbox_inches='tight')
