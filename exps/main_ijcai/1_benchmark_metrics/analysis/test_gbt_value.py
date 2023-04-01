import matplotlib.pyplot as plt
import numpy as np


def plot_graph(lines, bars, x, filename, marker_size=8, line_width=2, font_size=12, tick_font_size=10):
    # Configure plot settings
    plt.rcParams.update({'font.size': font_size})

    # Create figure and axis
    fig, ax1 = plt.subplots()

    # Plot bars
    bar_width = 0.4
    hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
    bar_color = '#1f77b4'

    for idx, value in enumerate(bars):
        ax1.bar(x[idx], value, width=bar_width, color=bar_color, alpha=0.5, hatch=hatches[idx])

    # Create a second y-axis on the right
    ax2 = ax1.twinx()

    # Plot lines
    line_styles = ['-', '--', ':', '-.', '-', '--', ':', '-.']
    marker_styles = ['o', 'v', 's', 'p', 'D', '*', 'h', 'H']

    for idx, line in enumerate(lines):
        ax2.plot(x, line, linestyle=line_styles[idx], marker=marker_styles[idx], markersize=marker_size,
                 linewidth=line_width)

    # Set x-axis label
    ax1.set_xlabel("X-Axis")

    # Set y-axis labels
    ax1.set_ylabel("FLOPs Bar / G")
    ax2.set_ylabel("SRCC Line")

    # Set y-ticks
    ax1.set_yticks(np.arange(0, 16, 2))
    ax2.set_yticks(np.arange(0.3, 0.85, 0.05))

    # Set tick font size
    ax1.tick_params(axis='both', which='major', labelsize=tick_font_size)
    ax2.tick_params(axis='both', which='major', labelsize=tick_font_size)

    # Save the figure as a PDF
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)


# Call the function with the provided data
lines = [
    [0.32, 0.34, 0.35, 0.38, 0.37],
    [0.37, 0.37, 0.38, 0.38, 0.38],
    [0.38, 0.39, 0.39, 0.38, 0.39],
    [0.64, 0.64, 0.64, 0.64, 0.65],
    [0.54, 0.55, 0.53, 0.52, 0.48],
    [0.78, 0.78, 0.78, 0.78, 0.78],
    [0.63, 0.64, 0.64, 0.64, 0.64],
    [0.80, 0.79, 0.79, 0.78, 0.76]
]
bars = [0.922310, 1.845, 3.689, 7.378, 14.756]
x = [1, 2, 3, 4, 5]

plot_graph(lines, bars, x, 'output.pdf')
