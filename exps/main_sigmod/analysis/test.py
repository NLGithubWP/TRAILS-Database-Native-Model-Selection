

import matplotlib.pyplot as plt

# create some data
x = [1, 2, 3, 4, 5]
y1 = [1, 3, 2, 4, 5]
y2 = [2, 4, 1, 5, 3]

# create a figure and axis object
fig, ax = plt.subplots()

# plot the data
ax.plot(x, y1, label='Line 1')
ax.plot(x, y2, label='Line 2')

# set the legend outside and above the graph
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2)

# set the x and y axis labels
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')

# set the title of the graph
ax.set_title('Graph Title')

# display the graph
plt.show()





