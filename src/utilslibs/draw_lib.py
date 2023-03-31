from typing import List

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.ticker as ticker

# lines' mark size
set_marker_size = 20
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

    if all(isinstance(item, list) for item in x_):
        expx = np.array(x_)
        x_m = np.quantile(expx, .5, axis=0)
    else:
        x_m = x_

    exp = np.array(y_) * 100
    y_h = np.quantile(exp, .75, axis=0)
    y_m = np.quantile(exp, .5, axis=0)
    y_l = np.quantile(exp, .25, axis=0)

    ax.plot(x_m, y_m,
            mark_list[index % len(mark_list)] + line_shape_list[index % len(line_shape_list)],
            label=namespace, markersize=set_marker_size)
    ax.fill_between(x_m, y_l, y_h, alpha=shade_degree)


def draw_structure_data_anytime(
        all_lines: List,
        dataset: str, name_img: str,
        x_ticks=None, y_ticks=None):
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

    if y_ticks is not None:
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks)
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks)

    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=5))

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


import seaborn as sns
import matplotlib.pyplot as plt


def plot_heatmap(data: List, fontsize: int, x_array_name: str, y_array_name: str, title: str, output_file: str, decimal_places: int):
    labelsize = fontsize
    # Convert the data to a NumPy array
    data_array = np.array(data)

    # Custom annotation function
    def custom_annot(val):
        return "{:.{}f}".format(val, decimal_places) if val > 0 else ""

    # Convert the custom annotations to a 2D array
    annot_array = np.vectorize(custom_annot)(data_array)

    # Create a masked array to hide the cells with values less than or equal to 0
    masked_data = np.ma.masked_array(data_array, data_array <= 0)

    # Set the figure size (width, height) in inches
    fig, ax = plt.subplots(figsize=(6, 4))

    # Use the "viridis" colormap
    cmap = "viridis"

    # Create a heatmap
    sns.heatmap(masked_data, annot=annot_array, fmt='', cmap=cmap, mask=masked_data.mask, ax=ax,
                annot_kws={"size": fontsize, "ha": "center", "va": "center"},
                xticklabels=np.arange(1, masked_data.shape[1] + 1), yticklabels=np.arange(1, masked_data.shape[0] + 1))

    # Set axis labels
    ax.set_xlabel(x_array_name, fontsize=fontsize)
    ax.set_ylabel(y_array_name, fontsize=fontsize)

    # Set x/y-axis tick size
    ax.tick_params(axis='both', which='major', labelsize=labelsize)

    # Set the title
    # ax.set_title(title, fontsize=fontsize)

    # Set tight layout
    plt.tight_layout()

    # Save the plot to a PDF file
    plt.savefig(output_file)
