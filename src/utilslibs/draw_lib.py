from typing import List

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.ticker as ticker

# lines' mark size
set_marker_size = 10
# points' mark size
set_marker_point = 14
# points' mark size
set_font_size = 20
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


def Add_one_line(x_time_array: list, y_twod_budget: List[List], namespace: str, index, ax):
    # training-based
    x_ = x_time_array
    y_ = y_twod_budget

    exp = np.array(y_) * 100
    y_h = np.quantile(exp, .75, axis=0)
    y_m = np.quantile(exp, .5, axis=0)
    y_l = np.quantile(exp, .25, axis=0)

    ax.plot(x_, y_m,
             mark_list[index%len(mark_list)]+line_shape_list[index%len(line_shape_list)],
             label=namespace, markersize=set_marker_size)
    ax.fill_between(x_, y_l, y_h, alpha=shade_degree)


def draw_structure_data_anytime(all_lines: List, dataset: str, name_img: str):
    fig, ax = plt.subplots()
    # draw all lines
    for i, each_line_info in enumerate(all_lines):
        _x_array = each_line_info[0]
        _y_2d_array = each_line_info[1]
        _name_space = each_line_info[2]
        Add_one_line(_x_array, _y_2d_array, _name_space, i, ax)

    # plt.xscale("log")
    # plt.grid()
    # plt.xlabel(r"Time Budget $T$ (min)", fontsize=set_font_size)
    # plt.ylabel(f"AUC on {dataset.upper()}", fontsize=set_font_size)

    ax.grid()
    ax.set_xlabel(r"Time Budget $T$ (min)", fontsize=set_font_size)
    ax.set_ylabel(f"AUC on {dataset.upper()}", fontsize=set_font_size)
    ax.set_xscale("log")
    # ax.set_xlim(0.001, 10e4)
    # ax.set_ylim(x1_lim[0], x1_lim[1])
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=5))
    # ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    export_legend(ori_fig=fig, colnum=5)
    plt.tight_layout()

    fig.savefig(f"{name_img}.pdf", bbox_inches='tight')


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


