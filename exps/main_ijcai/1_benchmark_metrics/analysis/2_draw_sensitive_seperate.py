

from matplotlib import pyplot as plt

from utilslibs.draw_tools import export_legend
import matplotlib


def revserList(a):
    return list(reversed(a))

# lines' mark size
set_marker_size = 20
# points' mark size
set_marker_point = 20
# points' mark size
set_font_size = 30
set_lgend_size = 15
set_tick_size = 15

frontinsidebox = 23

# update tick size
matplotlib.rc('xtick', labelsize=set_tick_size)
matplotlib.rc('ytick', labelsize=set_tick_size)

plt.rcParams['axes.labelsize'] = set_tick_size

mark_list = ["o", "*", "<", "^", "s", "d", "D", ">", "h"]
line_shape_list = ['-.', '--', '-', ':']

shade_degree = 0.2
bar_w = 0.4


def plot_channel():
    f, axcost = plt.subplots(1, 1, figsize=(9, 9))

    axs = axcost.twinx()

    ntk_app = [0.34, 0.35, 0.36, 0.36, 0.37]
    ntk = [0.36, 0.38, 0.39, 0.40, 0.42]
    fisher = [0.37, 0.39, 0.40, 0.41, 0.42]
    snip = [0.63, 0.64, 0.65, 0.65, 0.65]
    grasp = [0.45, 0.53, 0.58, 0.59, 0.59]
    synflow = [0.77, 0.78, 0.77, 0.77, 0.76]
    grad_norm = [0.63, 0.64, 0.65, 0.65, 0.65]
    nas_wot = [0.80, 0.79, 0.79, 0.79, 0.78]
    fix32B_flops = [0.936, 3.68924, 14.643, 58.3, 232.9] # in Gb
    fix32B_params = [201.018, 800.746, 3273, 13076, 52288] # in K

    x = [1, 2, 3, 4, 5]

    axcost.bar(x, fix32B_params, width=bar_w, fill=False, hatch='.', edgecolor="black")

    axs.plot(x, ntk_app, mark_list[1 % len(mark_list)] + line_shape_list[1 % len(line_shape_list)],  label="NTKTraceAppx",  markersize=set_marker_size)
    axs.plot(x, ntk, mark_list[2 % len(mark_list)] + line_shape_list[2 % len(line_shape_list)],  label="NTKTrace",  markersize=set_marker_size)
    axs.plot(x, fisher, mark_list[3 % len(mark_list)] + line_shape_list[3 % len(line_shape_list)],  label="Fisher",  markersize=set_marker_size)
    axs.plot(x, snip, mark_list[4 % len(mark_list)] + line_shape_list[4 % len(line_shape_list)],  label="SNIP",  markersize=set_marker_size)
    axs.plot(x, grasp, mark_list[5 % len(mark_list)] + line_shape_list[5 % len(line_shape_list)],  label="GraSP",  markersize=set_marker_size)
    axs.plot(x, synflow, mark_list[6 % len(mark_list)] + line_shape_list[6 % len(line_shape_list)],  label="SynFlow",  markersize=set_marker_size)
    axs.plot(x, grad_norm,mark_list[7 % len(mark_list)] + line_shape_list[7 % len(line_shape_list)],   label="GradNorm",  markersize=set_marker_size)
    axs.plot(x, nas_wot,mark_list[8 % len(mark_list)] + line_shape_list[8 % len(line_shape_list)],   label="NASWOT",  markersize=set_marker_size)
    axs.grid()

    # axs.legend(loc='upper right', fontsize=set_font_size)

    axcost.set_xticklabels([8, 8, 16, 32, 64, 128], fontsize=set_font_size)
    axcost.set_yticklabels([0, "10", "20", "30", "40", "50"], fontsize=set_font_size)

    # axcost.set_xlabel("(a) Channel-Size", fontsize=set_font_size)
    axcost.set_ylabel("Params Bar / M", fontsize=set_font_size)
    axs.set_ylabel("SRCC Line", fontsize=set_font_size)
    export_legend(f, filename="sensitive_legend", colnum=4)

    plt.tight_layout()
    f.savefig("plot_channel.pdf", bbox_inches='tight')


def plot_batchsize():
    f, axcost = plt.subplots(1, 1, figsize=(9, 9))

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
             label="NTKTraceAppx", markersize=set_marker_size)
    axs.plot(x, revserList(ntk), mark_list[1 % len(mark_list)] + line_shape_list[1 % len(line_shape_list)],
             label="NTKTrace", markersize=set_marker_size)
    axs.plot(x, revserList(fisher), mark_list[2 % len(mark_list)] + line_shape_list[2 % len(line_shape_list)],
             label="Fisher", markersize=set_marker_size)
    axs.plot(x, revserList(snip), mark_list[3 % len(mark_list)] + line_shape_list[3 % len(line_shape_list)],
             label="SNIP", markersize=set_marker_size)
    axs.plot(x, revserList(grasp), mark_list[4 % len(mark_list)] + line_shape_list[4 % len(line_shape_list)],
             label="GraSP", markersize=set_marker_size)
    axs.plot(x, revserList(synflow), mark_list[5 % len(mark_list)] + line_shape_list[5 % len(line_shape_list)],
             label="SynFlow", markersize=set_marker_size)
    axs.plot(x, revserList(grad_norm), mark_list[6 % len(mark_list)] + line_shape_list[6 % len(line_shape_list)],
             label="GradNorm", markersize=set_marker_size)
    axs.plot(x, revserList(nas_wot), mark_list[7 % len(mark_list)] + line_shape_list[7 % len(line_shape_list)],
             label="NASWOT", markersize=set_marker_size)

    axs.grid()
    # axs.legend(loc='upper right', fontsize=set_font_size)
    axs.set_ylabel("SRCC Line", fontsize=set_font_size)

    axcost.set_xticklabels([8, 8, 16, 32, 64, 128], fontsize=set_font_size)
    # axcost.set_xlabel("(b) Batch-Size", fontsize=set_font_size)
    axcost.set_ylabel("FLOPs Bar / G", fontsize=set_font_size)

    export_legend(f, filename="sensitive_legend", colnum=4)
    plt.tight_layout()
    f.savefig("plot_batchsize.pdf", bbox_inches='tight')


plot_batchsize()
plot_channel()




