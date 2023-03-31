import matplotlib.pyplot as plt
import numpy as np


# lines' mark size
set_marker_size = 10
# points' mark size
set_marker_point = 14
# points' mark size
set_font_size = 20

# Define the data for the bar plot
x_bar = [1, 2, 3, 4, 5]
y_bar = [201.018, 800.746, 3273, 13076, 52288]

# Define the data for the line plots
x_line = [1, 2, 3, 4, 5]

ntk_app = [0.34, 0.35, 0.36, 0.36, 0.37]
ntk = [0.36, 0.38, 0.39, 0.40, 0.42]
fisher = [0.37, 0.39, 0.40, 0.41, 0.42]
snip = [0.63, 0.64, 0.65, 0.65, 0.65]
grasp = [0.45, 0.53, 0.58, 0.59, 0.59]
synflow = [0.77, 0.78, 0.77, 0.77, 0.76]
grad_norm = [0.63, 0.64, 0.65, 0.65, 0.65]
nas_wot = [0.80, 0.79, 0.79, 0.79, 0.78]

# Create the figure and axis objects
fig, ax1 = plt.subplots(figsize=(8, 8))

# Create a second y-axis
ax2 = ax1.twinx()

# Create the bar plot
ax1.bar(x_bar, y_bar, color='b', alpha=0.5)

# Add markers to the line plots
ax2.plot(x_line, ntk_app, 'r-', label='NTKTraceAppx', marker='o', markersize=set_marker_size)
ax2.plot(x_line, ntk, 'g-', label='NTKTrace', marker='s', markersize=set_marker_size)
ax2.plot(x_line, fisher, 'm-', label='Fisher', marker='^', markersize=set_marker_size)
ax2.plot(x_line, snip, 'c-', label='SNIP', marker='*', markersize=set_marker_size)
ax2.plot(x_line, grasp, 'y-', label='GraSP', marker='d', markersize=set_marker_size)
ax2.plot(x_line, synflow, 'k-', label='SynFlow', marker='o', markersize=set_marker_size)
ax2.plot(x_line, grad_norm, 'b--', label='GradNorm', marker='s', markersize=set_marker_size)
ax2.plot(x_line, nas_wot, 'g--', label='NASWOT', marker='^', markersize=set_marker_size)

# Add a grid to the plot
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# Set the line colors
line_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
# Set the bar color
bar_color = '#ff7f0e'

# Define a list of hatch patterns
hatch_patterns = ['//', '++', 'xx', 'oo', '//']

# Plot the bar data with color and hatch
for i in range(len(x_bar)):
    ax1.bar(x_bar[i], y_bar[i], color=bar_color, hatch=hatch_patterns[i], edgecolor='black', alpha=0.2)

# Set the labels and titles
ax1.set_ylabel('Bar Result', fontsize=set_font_size)
ax1.set_yticks([0, 10000, 20000, 30000, 40000, 50000])
ax1.tick_params(axis='y', labelsize=12)

ax2.set_ylabel('Line Result', fontsize=set_font_size)
ax2.set_yticks([0.2, 0.4, 0.6, 0.8])
ax2.tick_params(axis='y', labelsize=12)

ax1.set_xlabel('X Label', fontsize=set_font_size)
ax1.set_xticks([1, 2, 3, 4, 5])
ax1.tick_params(axis='x', labelsize=set_font_size)

ax2.legend(loc='upper left', fontsize=set_font_size)

plt.title('Bar and Line Plot', fontsize=set_font_size)

plt.savefig(f"test.pdf", bbox_inches='tight')

