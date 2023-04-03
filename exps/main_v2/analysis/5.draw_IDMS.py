
from src.common.constant import Config
from query_api.query_model_gt_acc_api import GTMLP


def measure_total_time_usage(t1, t2, dataset):
    N = 1500
    K = 10
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


result = {}
print("\n-----------------------------\n")
for dataset in [Config.Criteo, Config.Frappe, Config.UCIDataset]:
    t1 = GTMLP.get_score_one_model_time(dataset, "cpu")
    t2 = GTMLP.get_train_one_epoch_time(dataset, "cuda:1")
    s1, s2 = measure_total_time_usage(t1, t2, dataset)
    if dataset not in result:
        result[dataset] = []
    data_load_time_one_ite, data_load_time_one_all = get_data_loading_time(dataset)
    result[dataset].append(s1 + s2 + data_load_time_one_ite)
    result[dataset].append(s1 + s2 + data_load_time_one_all)
    print(f"--hybrid , Dataset={dataset}, s1={s1}, s2={s2}, total={s1+s2}")
print(result)


import numpy as np
import matplotlib.pyplot as plt

# lines' mark size
set_marker_size = 10
# points' mark size
set_marker_point = 14
# points' mark size
set_font_size = 20
set_lgend_size = 15
set_tick_size = 15
frontinsidebox = 23

time_usage = {}
datasets = [Config.Criteo, Config.Frappe, Config.UCIDataset]
for dataset in datasets:
    time_usage[dataset] = result[dataset]


num_datasets = len(datasets)
num_bars = 3  # CPU-only, GPU-only, Hybrid

bar_width = 0.25
opacity = 0.8

index = np.arange(num_datasets)

fig, ax = plt.subplots(figsize=(6.4, 4.5))
ax.grid()

colors = ['#729ECE', '#FFB579', '#98DF8A']  # Softer colors #FF7F7F
hatches = ['/', '\\',  'x', 'o', 'O', '.', '*']

# Set the font size
fontsize = set_font_size

# Plot bars for external dataloader
rects1 = ax.bar(index - bar_width, [time_usage[dataset][0] for dataset in datasets], bar_width,
                alpha=opacity, color=colors[0], hatch=hatches[0], edgecolor='black', label='TRAILS w/ RDBMS')

# Plot bars for IDMS
rects2 = ax.bar(index, [time_usage[dataset][1] for dataset in datasets], bar_width,
                alpha=opacity, color=colors[1], hatch=hatches[1], edgecolor='black', label='TRAILS w/o RDBMS')


ax.set_ylabel('Latency (s)', fontsize=fontsize)
ax.set_xticks(index)
ax.set_xticklabels(["Criteo", "Frappe", "Diabetes"], fontsize=fontsize)

# linear', 'log', 'symlog', 'logit', 'function', 'functionlog'
ax.set_yscale('log')  # Set y-axis to logarithmic scale

ax.set_ylim(ymax=10**3*7)

ax.legend(fontsize=fontsize)

# Set the font size for tick labels
ax.tick_params(axis='both', labelsize=fontsize)
fig.tight_layout()
plt.tight_layout()

# export_legend(ori_fig=fig, colnum=5)

fig.savefig("./exps/main_v2/analysis/IDMS_latency.pdf", bbox_inches='tight')
