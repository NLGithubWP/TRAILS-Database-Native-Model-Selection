
import os
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import palettable


def draw_grid_graph(
        acc: list,
        bt: list,
        img_name: str, y_array: list, x_array: list):
    """
    :param acc: Two array list
    :param bt:  Two array list
    :param img_name: img name string
    :return:
    """

    mask = np.array(acc)
    mask[mask > 0] = 0
    mask[mask < 0] = 1

    bt = np.round(np.array(bt), 2).tolist()

    mask2 = np.array(bt)
    mask2[mask2 > 0] = 0
    mask2[mask2 < 0] = 1

    fig, ax = plt.subplots(1, 2, figsize=(15, 9))
    frontsizeall = 12

    sns.heatmap(
        data=acc,
        vmax=0.99,
        vmin=0.93,
        cmap=palettable.cmocean.diverging.Curl_10.mpl_colors,
        annot=True,
        fmt=".4f",
        annot_kws={'size': frontsizeall, 'weight': 'normal', 'color': 'w', 'va': 'bottom'},
        mask=mask,
        square=True, linewidths=.5,  # 每个方格外框显示，外框宽度设置
        cbar_kws={"shrink": .5},
        ax=ax[0]
    )

    sns.heatmap(
        data=bt,
        # vmax=,
        vmin=-9,
        cmap=palettable.cmocean.diverging.Curl_10.mpl_colors,
        annot=True,
        fmt=".2f",
        annot_kws={'size': frontsizeall, 'weight': 'normal', 'color': 'w', 'va': 'top'},
        mask=mask2,
        square=True, linewidths=.5,  # 每个方格外框显示，外框宽度设置
        cbar_kws={"shrink": .5},
        ax=ax[1]
    )

    plt.tight_layout()
    plt.xlabel("mini-eval-epoch", fontsize=frontsizeall)
    plt.ylabel("# models", fontsize=frontsizeall)

    ax[0].set_xticklabels(x_array)
    ax[0].set_yticklabels(y_array)
    ax[1].set_xticklabels(x_array)
    ax[1].set_yticklabels(y_array)

    base_dr = os.getcwd()
    path_gra = os.path.join(base_dr, f"{img_name}.pdf")

    plt.show()
    fig.savefig(path_gra, bbox_inches='tight')


def draw_grid_graph_with_budget(
        acc, bt, b1, b2,
        img_name: str, y_array: list, x_array: list):
    """
    :param acc: Two array list
    :param bt:  Two array list
    :param img_name: img name string
    :return:
    """

    mask = np.array(acc)
    mask[mask > 0] = 0
    mask[mask < 0] = 1

    bt = np.round(np.array(bt), 2).tolist()
    mask2 = np.array(bt)
    mask2[mask2 > 0] = 0
    mask2[mask2 < 0] = 1

    mask3 = np.array(b1)
    mask3[mask3 > 0] = 0
    mask3[mask3 < 0] = 1

    mask4 = np.array(b2)
    mask4[mask4 > 0] = 0
    mask4[mask4 < 0] = 1

    fig, ax = plt.subplots(2, 2, figsize=(15, 14))
    frontsizeall = 12

    sns.heatmap(
        data=acc,
        vmax=0.99,
        vmin=0.93,
        cmap=palettable.cmocean.diverging.Curl_10.mpl_colors,
        annot=True,
        fmt=".4f",
        annot_kws={'size': frontsizeall, 'weight': 'normal', 'color': 'w', 'va': 'bottom'},
        mask=mask,
        square=True, linewidths=.5,  # 每个方格外框显示，外框宽度设置
        cbar_kws={"shrink": .5},
        ax=ax[0, 0]
    )

    sns.heatmap(
        data=bt,
        # vmax=,
        vmin=-9,
        cmap=palettable.cmocean.diverging.Curl_10.mpl_colors,
        annot=True,
        fmt=".2f",
        annot_kws={'size': frontsizeall, 'weight': 'normal', 'color': 'w', 'va': 'top'},
        mask=mask2,
        square=True, linewidths=.5,  # 每个方格外框显示，外框宽度设置
        cbar_kws={"shrink": .5},
        ax=ax[0, 1]
    )

    sns.heatmap(
        data=b1,
        # vmax=,
        # vmin=-9,
        cmap=palettable.cmocean.diverging.Curl_10.mpl_colors,
        annot=True,
        fmt=".2f",
        annot_kws={'size': frontsizeall, 'weight': 'normal', 'color': 'w', 'va': 'top'},
        mask=mask4,
        square=True, linewidths=.5,  # 每个方格外框显示，外框宽度设置
        cbar_kws={"shrink": .5},
        ax=ax[1, 0]
    )

    sns.heatmap(
        data=b2,
        # vmax=,
        # vmin=-9,
        cmap=palettable.cmocean.diverging.Curl_10.mpl_colors,
        annot=True,
        fmt=".2f",
        annot_kws={'size': frontsizeall, 'weight': 'normal', 'color': 'w', 'va': 'top'},
        mask=mask4,
        square=True, linewidths=.5,  # 每个方格外框显示，外框宽度设置
        cbar_kws={"shrink": .5},
        ax=ax[1, 1]
    )

    plt.tight_layout()
    plt.xlabel("U (epoch)", fontsize=frontsizeall)
    plt.ylabel("K (# models)", fontsize=frontsizeall)

    for i in [0, 1]:
        for j in [0, 1]:
            ax[i, j].set_xticklabels(x_array)
            ax[i, j].set_yticklabels(y_array)
            ax[i, j].set_xlabel("U (epoch)", fontsize=frontsizeall)
            ax[i, j].set_ylabel("K (# models)", fontsize=frontsizeall)

    plt.show()
    base_dr = os.getcwd()
    path_gra = os.path.join(base_dr, f"{img_name}.pdf")
    fig.savefig(path_gra, bbox_inches='tight')


if __name__ == "__main__":
    acc = [[0.9347, 0.9363, 0.9382, 0.9379, 0.9382, 0.9382], [0.9363, 0.9382, 0.9382, 0.9379, 0.9382, 0.9389],
           [0.9332, 0.9356, 0.9382, 0.9379, 0.9375, 0.9389], [0.9345, 0.9374, 0.9421, 0.9379, 0.9421, 0.9421],
           [0.9436, 0.9436, 0.9421, 0.9412, 0.9437, 0.9437], [0.9436, 0.9436, 0.9412, 0.9421, 0.9437, 0.9437]]

    bt = [[1.7218967777777778, 1.8421218888888888, 1.926109777777778, 3.0912538888888887, 4.489907777777778,
           7.239692555555555],
          [1.7286255555555559, 1.9892567777777779, 2.2744125555555557, 4.463216777777777, 7.2416455555555554,
           12.815212111111112],
          [1.8002391111111111, 2.2531465555555554, 2.8026261111111115, 7.270723555555556, 12.80045611111111,
           23.925889222222224],
          [1.992945777777778, 3.096895888888889, 4.474283777777778, 15.55501488888889, 29.47211377777778,
           57.244249555555555],
          [2.2401265555555554, 4.450847777777778, 7.2679025555555565, 29.45323477777778, 57.256401555555556,
           112.80175811111113],
          [2.7850491111111113, 7.211048555555555, 12.800673111111111, 57.248589555555554, 112.77832211111111,
           223.9269742222222]]

    y_k_array = [5, 10, 20, 50, 100, 200]
    x_epoch_array = [1, 5, 10, 50, 100, 200]

    draw_grid_graph(acc, bt, "test", y_k_array, x_epoch_array)
