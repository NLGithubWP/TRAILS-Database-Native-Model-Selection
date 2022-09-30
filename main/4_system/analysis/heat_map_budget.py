import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# 这里是创建一个数据
vegetables = [2, 3, 4, 5, 6]
farmers = [4, 12, 20, 28, 36, 44, 52]

metrics = [[132.8771237954945, 398.63137138648347, 664.3856189774725, 930.1398665684615, 1195.8941141594505, 1461.6483617504396, 1727.4026093414284], [83.83613097157539, 251.50839291472616, 419.1806548578769, 586.8529168010277, 754.5251787441784, 922.1974406873293, 1089.86970263048], [66.43856189774725, 199.31568569324173, 332.19280948873626, 465.06993328423073, 597.9470570797253, 730.8241808752198, 863.7013046707142], [57.22706232293574, 171.6811869688072, 286.1353116146787, 400.58943626055014, 515.0435609064216, 629.4976855522931, 743.9518101981646], [51.403888357538754, 154.21166507261626, 257.0194417876938, 359.8272185027713, 462.63499521784877, 565.4427719329263, 668.2505486480038]]

harvest = np.rint(np.array(metrics))

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

base_dr =os.getcwd()
path_gra = os.path.join(base_dr, "result_base/paper_graph/sh_budget.pdf")
fig.savefig(path_gra, bbox_inches='tight')

