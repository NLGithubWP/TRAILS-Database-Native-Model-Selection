import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# 这里是创建一个数据
vegetables = [2, 3, 4, 5, 6]
farmers = [4, 12, 20, 28, 36, 44, 52]

metrics = [[0.382, 0.46, 0.472, 0.562, 0.656, 0.731, 0.709], [0.409, 0.44, 0.483, 0.479, 0.546, 0.585, 0.737], [0.394, 0.419, 0.465, 0.546, 0.652, 0.627, 0.651], [0.4, 0.431, 0.433, 0.519, 0.639, 0.636, 0.652], [0.376, 0.428, 0.395, 0.416, 0.415, 0.398, 0.443]]

harvest = np.array(metrics)

# 这里是创建一个画布
fig, ax = plt.subplots()
im = ax.imshow(harvest)

# 这里是修改标签
# We want to show all ticks...
ax.set_xticks(np.arange(len(farmers)))
ax.set_yticks(np.arange(len(vegetables)))
# ... and label them with the respective list entries
ax.set_xticklabels(farmers)
ax.set_yticklabels(vegetables)

# 因为x轴的标签太长了，需要旋转一下，更加好看
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# 添加每个热力块的具体数值
# Loop over data dimensions and create text annotations.
for i in range(len(vegetables)):
    for j in range(len(farmers)):
        text = ax.text(j, i, harvest[i, j],
                       ha="center", va="center", color="w")

frontsizeall  =12
fig.tight_layout()
plt.xlabel("mini-eval-epoch", fontsize=frontsizeall)
plt.ylabel(r"$\eta$", fontsize=frontsizeall)
plt.colorbar(im)
plt.show()

base_dr = os.getcwd()
path_gra = os.path.join(base_dr, "result_base/paper_graph/sh_sr.pdf")
fig.savefig(path_gra, bbox_inches='tight')

