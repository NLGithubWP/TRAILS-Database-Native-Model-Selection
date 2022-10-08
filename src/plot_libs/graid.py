import os

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import palettable

acc = [[0.9347, 0.9363, 0.9382, 0.9379, 0.9382, 0.9382], [0.9363, 0.9382, 0.9382, 0.9379, 0.9382, 0.9389], [0.9332, 0.9356, 0.9382, 0.9379, 0.9375, 0.9389], [0.9345, 0.9374, 0.9421, 0.9379, 0.9421, 0.9421], [0.9436, 0.9436, 0.9421, 0.9412, 0.9437, 0.9437], [0.9436, 0.9436, 0.9412, 0.9421, 0.9437, 0.9437]]

bt = [[1.7218967777777778, 1.8421218888888888, 1.926109777777778, 3.0912538888888887, 4.489907777777778, 7.239692555555555], [1.7286255555555559, 1.9892567777777779, 2.2744125555555557, 4.463216777777777, 7.2416455555555554, 12.815212111111112], [1.8002391111111111, 2.2531465555555554, 2.8026261111111115, 7.270723555555556, 12.80045611111111, 23.925889222222224], [1.992945777777778, 3.096895888888889, 4.474283777777778, 15.55501488888889, 29.47211377777778, 57.244249555555555], [2.2401265555555554, 4.450847777777778, 7.2679025555555565, 29.45323477777778, 57.256401555555556, 112.80175811111113], [2.7850491111111113, 7.211048555555555, 12.800673111111111, 57.248589555555554, 112.77832211111111, 223.9269742222222]]

mask = np.array(acc)
mask[mask > 0] = 0
mask[mask < 0] = 1

bt = np.round(np.array(bt), 2).tolist()

mask2 = np.array(bt)
mask2[mask2 > 0] = 0
mask2[mask2 < 0] = 1

fig, ax = plt.subplots(1, 2, figsize=(21, 9))

sns.heatmap(
    data=acc,
    vmax=0.97,
    vmin=0.92,
    cmap=palettable.cmocean.diverging.Curl_10.mpl_colors,
    annot=True,
    fmt=".4f",
    annot_kws={'size': 12, 'weight': 'normal', 'color': '#253D24', 'va': 'bottom'},
    mask=mask,
    square=True, linewidths=.5,  # 每个方格外框显示，外框宽度设置
    cbar_kws={"shrink": .5},
    ax=ax[0]
)

sns.heatmap(
    data=bt,
    vmax=200,
    vmin=-100,
    cmap=palettable.cmocean.diverging.Curl_10.mpl_colors,
    annot=True,
    fmt=".2f",
    annot_kws={'size': 12, 'weight': 'normal', 'color': '#253D24', 'va': 'top'},
    mask=mask2,
    square=True, linewidths=.5,  # 每个方格外框显示，外框宽度设置
    cbar_kws={"shrink": .5},
    ax=ax[1]
)

vegetables = [5, 10, 20, 50, 100, 200]
farmers = [1, 5, 10, 50, 100, 200]

frontsizeall = 12
plt.tight_layout()
plt.xlabel("mini-eval-epoch", fontsize=frontsizeall)
plt.ylabel("# models", fontsize=frontsizeall)

for i in [0, 1]:
    ax[i].set_xticklabels(farmers)
    ax[i].set_yticklabels(vegetables)
    ax[i].set_xlabel("mini-eval-epoch", fontsize=frontsizeall)
    ax[i].set_ylabel("# models", fontsize=frontsizeall)

# plt.colorbar()

plt.show()
fig.delaxes(ax[0])
base_dr = os.getcwd()
path_gra = os.path.join(base_dr, "256all.pdf")
fig.savefig(path_gra, bbox_inches='tight')







