

from matplotlib import pyplot as plt

from utilslibs.draw_tools import export_legend
import matplotlib


def revserList(a):
    return list(reversed(a))


frontsizeall = 30
marksizeall = 10
bar_w = 0.4
set_tick_size = 30

# update tick size
matplotlib.rc('xtick', labelsize=set_tick_size)
matplotlib.rc('ytick', labelsize=set_tick_size)

plt.rcParams['axes.labelsize'] = set_tick_size

mark_list = ["o", "*", "<", "^", "s", "h", ">", "+"]

def plot_channel():
    f, axcost = plt.subplots(1, 1, figsize=(9, 7))

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

    axs.plot(x, ntk_app, label="NTKTraceAppx", marker=mark_list[0], markersize=marksizeall)
    axs.plot(x, ntk, label="NTKTrace", marker=mark_list[1], markersize=marksizeall)
    axs.plot(x, fisher, label="Fisher", marker=mark_list[2], markersize=marksizeall)
    axs.plot(x, snip, label="SNIP", marker=mark_list[3], markersize=marksizeall)
    axs.plot(x, grasp, label="GraSP", marker=mark_list[4], markersize=marksizeall)
    axs.plot(x, synflow, label="SynFlow", marker=mark_list[5], markersize=marksizeall)
    axs.plot(x, grad_norm, label="GradNorm", marker=mark_list[6], markersize=marksizeall)
    axs.plot(x, nas_wot, label="NASWOT", marker=mark_list[7], markersize=marksizeall)
    axs.grid()

    # axs.legend(loc='upper right', fontsize=frontsizeall)

    axcost.set_xticklabels([8, 8, 16, 32, 64, 128], fontsize=frontsizeall)
    axcost.set_yticklabels([0, "10", "20", "30", "40", "50"], fontsize=frontsizeall)

    # axcost.set_xlabel("(a) Channel-Size", fontsize=frontsizeall)
    axcost.set_ylabel("Params Bar / M", fontsize=frontsizeall)
    axs.set_ylabel("SRCC Line", fontsize=frontsizeall)
    export_legend(f, filename="sensitive_legend", colnum=4)

    plt.tight_layout()
    f.savefig("plot_channel.pdf", bbox_inches='tight')


def plot_batchsize():
    f, axcost = plt.subplots(1, 1, figsize=(9, 7))

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

    axs.plot(x, revserList(ntk_app), label="NTKTraceAppx", marker=mark_list[0], markersize=marksizeall)
    axs.plot(x, revserList(ntk), label="NTKTrace", marker=mark_list[1], markersize=marksizeall)
    axs.plot(x, revserList(fisher), label="Fisher", marker=mark_list[2], markersize=marksizeall)
    axs.plot(x, revserList(snip), label="SNIP", marker=mark_list[3], markersize=marksizeall)
    axs.plot(x, revserList(grasp), label="GraSP", marker=mark_list[4], markersize=marksizeall)
    axs.plot(x, revserList(synflow), label="SynFlow", marker=mark_list[5], markersize=marksizeall)
    axs.plot(x, revserList(grad_norm), label="GradNorm", marker=mark_list[6], markersize=marksizeall)
    axs.plot(x, revserList(nas_wot), label="NASWOT", marker=mark_list[7], markersize=marksizeall)

    axs.grid()
    # axs.legend(loc='upper right', fontsize=frontsizeall)
    axs.set_ylabel("SRCC Line", fontsize=frontsizeall)

    axcost.set_xticklabels([8, 8, 16, 32, 64, 128], fontsize=frontsizeall)
    # axcost.set_xlabel("(b) Batch-Size", fontsize=frontsizeall)
    axcost.set_ylabel("FLOPs Bar / G", fontsize=frontsizeall)

    export_legend(f, filename="sensitive_legend", colnum=4)
    plt.tight_layout()
    f.savefig("plot_batchsize.pdf", bbox_inches='tight')


plot_batchsize()
plot_channel()




