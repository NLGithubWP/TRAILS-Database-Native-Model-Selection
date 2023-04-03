

import json
from matplotlib import pyplot as plt

total_models = 300
api = None
with open("score_c10_101.json", 'r') as outfile:
    result = json.load(outfile)


x_list = []
y_list = []

# each run evaluates 300 models
for i in range(total_models):
    x_list.append(0)
for run, info in result.items():
    # average x axis
    for i in range(len(info["x_axis_time"])):
        x_list[i] += info["x_axis_time"][i]
x_list = [ele/len(result) for ele in x_list]


diff = []
for i in range(1, len(x_list), 1):
    diff.append((x_list[i] - x_list[i-1]))

print(diff)
print(min(diff))
print(max(diff))

plt.plot(diff, label="diff")
# plt.plot(x_list, label="TFMEM")

plt.show()


