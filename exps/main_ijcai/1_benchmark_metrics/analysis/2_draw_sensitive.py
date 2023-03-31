

from matplotlib import pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.ticker as ticker

def revserList(a):
    return list(reversed(a))


# lines' mark size
marksizeall = 10
# points' mark size
set_marker_point = 14
# points' mark size
frontsizeall = 20
set_lgend_size = 20
set_tick_size = 15

frontinsidebox = 20

# update tick size
matplotlib.rc('xtick', labelsize=set_tick_size)
matplotlib.rc('ytick', labelsize=set_tick_size)

plt.rcParams['axes.labelsize'] = set_tick_size

mark_list = ["o", "*", "<", "^", "s", "d", "D", ">", "h"]
line_shape_list = ['-.', '--', '-', ':']
shade_degree = 0.2


bar_w = 0.4
def plot_channel(axcost):
    axs = axcost.twinx()

    ntk_app = [0.34, 0.35, 0.36, 0.36, 0.37]
    ntk = [0.36, 0.38, 0.39, 0.40, 0.42]
    ntk_condnum = []

    fisher = [0.37, 0.39, 0.40, 0.41, 0.42]

    snip = [0.63, 0.64, 0.65, 0.65, 0.65]

    grasp = [0.45, 0.53, 0.58, 0.59, 0.59]

    synflow = [0.77, 0.78, 0.77, 0.77, 0.76]

    grad_norm = [0.63, 0.64, 0.65, 0.65, 0.65]
    weight_norm = []

    nas_wot = [0.80, 0.79, 0.79, 0.79, 0.78]

    fix32B_flops = [0.936, 3.68924, 14.643, 58.3, 232.9] # in Gb
    fix32B_params = [201.018, 800.746, 3273, 13076, 52288] # in K

    x = [1, 2, 3, 4, 5]

    axcost.bar(x, fix32B_params, width=bar_w, fill=False, hatch='.', edgecolor="black")

    axs.plot(x, ntk_app, mark_list[1 % len(mark_list)] + line_shape_list[1 % len(line_shape_list)],  label="NTKTraceAppx",  markersize=marksizeall)
    axs.plot(x, ntk, mark_list[2 % len(mark_list)] + line_shape_list[2 % len(line_shape_list)],  label="NTKTrace",  markersize=marksizeall)
    axs.plot(x, fisher, mark_list[3 % len(mark_list)] + line_shape_list[3 % len(line_shape_list)],  label="Fisher",  markersize=marksizeall)
    axs.plot(x, snip, mark_list[4 % len(mark_list)] + line_shape_list[4 % len(line_shape_list)],  label="SNIP",  markersize=marksizeall)
    axs.plot(x, grasp, mark_list[5 % len(mark_list)] + line_shape_list[5 % len(line_shape_list)],  label="GraSP",  markersize=marksizeall)
    axs.plot(x, synflow, mark_list[6 % len(mark_list)] + line_shape_list[6 % len(line_shape_list)],  label="SynFlow",  markersize=marksizeall)
    axs.plot(x, grad_norm,mark_list[7 % len(mark_list)] + line_shape_list[7 % len(line_shape_list)],   label="GradNorm",  markersize=marksizeall)
    axs.plot(x, nas_wot,mark_list[8 % len(mark_list)] + line_shape_list[8 % len(line_shape_list)],   label="NASWOT",  markersize=marksizeall)
    axs.plot(x, ntk_app, mark_list[9 % len(mark_list)] + line_shape_list[9 % len(line_shape_list)],  label="NTKTraceAppx",  markersize=marksizeall)
    axs.grid()

    # axs.legend(loc='upper right', fontsize=frontsizeall)

    axcost.set_xticklabels([8, 8, 16, 32, 64, 128], fontsize=frontsizeall)
    axcost.set_yticklabels([0, "10k", "20k", "30k", "40k", "50k"], fontsize=frontsizeall)

    axcost.set_xlabel("(a) Channel-Size", fontsize=frontsizeall)
    axcost.set_ylabel("Params bar / K", fontsize=frontsizeall)
    axs.set_ylabel("SRCC line", fontsize=frontsizeall)


def plot_batchsize(axcost):

    axs = axcost.twinx()

    ntk_app = [0.32, 0.34, 0.35, 0.38, 0.37]
    ntk = [0.37, 0.37, 0.38, 0.38, 0.38]
    ntk_condnum = []

    fisher = [0.38, 0.39, 0.39, 0.38, 0.39]

    snip = [0.64, 0.64, 0.64, 0.64, 0.65]

    grasp = [0.54, 0.55, 0.53, 0.52, 0.48]
    synflow = [0.78, 0.78, 0.78, 0.78, 0.78]

    grad_norm = [0.63, 0.64, 0.64, 0.64, 0.64]
    weight_norm = []

    nas_wot = [0.80, 0.79, 0.79, 0.78, 0.76]

    fix16C_flops = [0.922310, 1.845, 3.689, 7.378, 14.756] # in G
    fix16C_params = [800.746, 800.746, 800.746, 800.746, 800.746] # in M

    x = [1, 2, 3, 4, 5]

    axcost.bar(x, fix16C_flops, width=bar_w, fill=False, hatch='/',)

    axs.plot(x, revserList(ntk_app),
             mark_list[1 % len(mark_list)] + line_shape_list[1 % len(line_shape_list)],
             label="NTKTraceAppx",  markersize=marksizeall)
    axs.plot(x, revserList(ntk), mark_list[1 % len(mark_list)] + line_shape_list[1 % len(line_shape_list)], label="NTKTrace",  markersize=marksizeall)
    axs.plot(x, revserList(fisher), mark_list[2 % len(mark_list)] + line_shape_list[2 % len(line_shape_list)], label="Fisher",  markersize=marksizeall)
    axs.plot(x, revserList(snip), mark_list[3 % len(mark_list)] + line_shape_list[3 % len(line_shape_list)], label="SNIP",  markersize=marksizeall)
    axs.plot(x, revserList(grasp), mark_list[4 % len(mark_list)] + line_shape_list[4 % len(line_shape_list)], label="GraSP",  markersize=marksizeall)
    axs.plot(x, revserList(synflow), mark_list[5 % len(mark_list)] + line_shape_list[5 % len(line_shape_list)], label="SynFlow",  markersize=marksizeall)
    axs.plot(x, revserList(grad_norm), mark_list[6 % len(mark_list)] + line_shape_list[6 % len(line_shape_list)], label="GradNorm",  markersize=marksizeall)
    axs.plot(x, revserList(nas_wot), mark_list[7 % len(mark_list)] + line_shape_list[7 % len(line_shape_list)], label="NASWOT",  markersize=marksizeall)

    axs.grid()
    # axs.legend(loc='upper right', fontsize=frontsizeall)
    axs.set_ylabel("SRCC line", fontsize=frontsizeall)

    axcost.set_xticklabels([8, 8, 16, 32, 64, 128], fontsize=frontsizeall)
    axcost.set_xlabel("(b) Batch-Size", fontsize=frontsizeall)
    axcost.set_ylabel("FLOPs bar / G", fontsize=frontsizeall)


f, allaxs = plt.subplots(1, 2, figsize=(20, 9))

allaxs = allaxs.ravel()

plt.yticks(fontsize=frontsizeall)
plot_channel(allaxs[0])
plt.yticks(fontsize=frontsizeall)
plot_batchsize(allaxs[1])
plt.yticks(fontsize=frontsizeall)

plt.subplots_adjust(hspace=20, wspace=10)

# this is for unique hash
lines_labels = [ax.get_legend_handles_labels() for ax in f.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
# grab unique labels
unique_labels = set(labels)
# assign labels and legends in dict
legend_dict = dict(zip(labels, lines))
# query dict based on unique labels
unique_lines = [legend_dict[x] for x in unique_labels]
f.legend(unique_lines, unique_labels, loc='upper center', bbox_to_anchor=(0.5, 1.05),
         ncol=9, fancybox=True, shadow=True, scatterpoints=1, fontsize=frontsizeall)

plt.tight_layout()
# plt.show()

f.savefig("sen.pdf", bbox_inches='tight')

