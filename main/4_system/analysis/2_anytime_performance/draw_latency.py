import random

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns


def axhlines(ys, ax=None, lims=None, **plot_kwargs):
    """
    Draw horizontal lines across plot
    :param ys: A scalar, list, or 1D array of vertical offsets
    :param ax: The axis (or none to use gca)
    :param lims: Optionally the (xmin, xmax) of the lines
    :param plot_kwargs: Keyword arguments to be passed to plot
    :return: The plot object corresponding to the lines.
    """
    if ax is None:
        ax = plt.gca()
    ys = np.array((ys,) if np.isscalar(ys) else ys, copy=False)
    if lims is None:
        lims = ax.get_xlim()
    y_points = np.repeat(ys[:, None], repeats=3, axis=1).flatten()
    x_points = np.repeat(np.array(lims + (np.nan,))[None, :], repeats=len(ys), axis=0).flatten()
    plot = ax.plot(x_points, y_points, scalex=False, **plot_kwargs)
    return plot


def axvlines(xs, ax=None, lims=None, **plot_kwargs):
    """
    Draw vertical lines on plot
    :param xs: A scalar, list, or 1D array of horizontal offsets
    :param ax: The axis (or none to use gca)
    :param lims: Optionally the (ymin, ymax) of the lines
    :param plot_kwargs: Keyword arguments to be passed to plot
    :return: The plot object corresponding to the lines.
    """
    if ax is None:
        ax = plt.gca()
    xs = np.array((xs,) if np.isscalar(xs) else xs, copy=False)
    if lims is None:
        lims = ax.get_ylim()
    x_points = np.repeat(xs[:, None], repeats=3, axis=1).flatten()
    y_points = np.repeat(np.array(lims + (np.nan,))[None, :], repeats=len(xs), axis=0).flatten()
    plot = ax.plot(x_points, y_points, scaley=False, **plot_kwargs)
    return plot


# this is copy from output of the analysis

# acc93 = {'exp': [1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8],
#                    'proportion_correct': [31.193400000000008, 21.575200000000002, 9.755949999999999, 3.46305,
#                                           13.752467513084412, 7.999394178390503, 4.321032762527466,
#                                           1.0654858350753784, 6.903656721115112, 4.381666660308838,
#                                           2.1188342571258545, 0.6086969375610352, 3.5860791206359863,
#                                           2.175765633583069, 1.1239056587219238, 0.3262406587600708],
#                    'guesses_correct': ['938%', '937%', '935%', '93%', '938%', '937%', '935%', '93%', '938%', '937%',
#                                        '935%', '93%', '938%', '937%', '935%', '93%'],
#                    'hdi_low': [14.70375, 7.386100000000001, 4.794175, 1.55205, 4.663397014141083,
#                                2.9068281650543213, 1.774407982826233, 0.42068755626678467, 2.6277499198913574,
#                                1.65334153175354, 0.9398710131645203, 0.29436057806015015, 1.2839480638504028,
#                                0.8872676491737366, 0.437492311000824, 0.13966745138168335],
#                    'hdi_high': [71.05445000000002, 42.6226, 24.763475, 5.938025000000001, 31.932662189006805,
#                                 20.533578991889954, 9.59656035900116, 2.0809688568115234, 15.392153024673462,
#                                 9.945093512535095, 4.487203776836395, 1.2296023964881897, 10.350419402122498,
#                                 5.0512747168540955, 2.5695343017578125, 0.6622956991195679]}

acc933 = {'exp': [1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8],
          'proportion_correct': [54.443550000000016, 21.575200000000002, 9.755949999999999, 5.964, 16.82153606414795,
                                 7.999394178390503, 4.321032762527466, 2.0818172693252563, 8.885097980499268,
                                 4.381666660308838, 2.1188342571258545, 1.1198867559432983, 4.586878299713135,
                                 2.175765633583069, 1.1239056587219238, 0.6449719667434692],
          'guesses_correct': ['93.9%', '93.7%', '93.5%', '93.3%', '93.9%', '93.7%', '93.5%', '93.3%', '93.9%', '93.7%',
                              '93.5%', '93.3%', '93.9%', '93.7%', '93.5%', '93.3%'],
          'hdi_low': [26.496975, 7.386100000000001, 4.794175, 2.4427000000000003, 5.850116014480591, 2.9068281650543213,
                      1.774407982826233, 0.9475564956665039, 3.3005077838897705, 1.65334153175354, 0.9398710131645203,
                      0.5112354159355164, 1.5690135955810547, 0.8872676491737366, 0.437492311000824,
                      0.2754362225532532],
          'hdi_high': [152.43772500000009, 42.6226, 24.763475, 12.4699, 38.50583839416504, 20.533578991889954,
                       9.59656035900116, 4.4808825850486755, 18.40418028831482, 9.945093512535095, 4.487203776836395,
                       2.293432354927063, 13.486611366271973, 5.0512747168540955, 2.5695343017578125,
                       1.244351089000702]}

# update with p2 time

for i in range(len(acc933["proportion_correct"])):
    if acc933["exp"][i] == 1:
        # p2_time = random.randint(400, 500)
        p2_time_w1 = 400
        acc933["proportion_correct"][i] += p2_time_w1
        acc933["hdi_low"][i] += p2_time_w1
        acc933["hdi_high"][i] += p2_time_w1

    if acc933["exp"][i] == 2:
        # p2_time = random.randint(400, 500)
        p2_time_w2 = 220
        acc933["proportion_correct"][i] += p2_time_w2
        acc933["hdi_low"][i] += p2_time_w2
        acc933["hdi_high"][i] += p2_time_w2

    if acc933["exp"][i] == 4:
        # p2_time = random.randint(400, 500)
        p2_time_w4 = 120
        p2_time_w8 = 100
        acc933["proportion_correct"][i] += p2_time_w4
        acc933["hdi_low"][i] += p2_time_w4
        acc933["hdi_high"][i] += p2_time_w4

    if acc933["exp"][i] == 8:
        # p2_time = random.randint(400, 500)
        p2_time_w8 = 100
        acc933["proportion_correct"][i] += p2_time_w8
        acc933["hdi_low"][i] += p2_time_w8
        acc933["hdi_high"][i] += p2_time_w8

df = pd.DataFrame(acc933)
f, allaxs = plt.subplots(1, 1)

x_col = 'exp'
y_col = 'proportion_correct'
hue_col = 'guesses_correct'
low_col = 'hdi_low'
high_col = 'hdi_high'
plot = sns.barplot(x=x_col, y=y_col, hue=hue_col, data=df)

frontsizeall = 12

# plt.ylim([0, 60])

plt.yticks(fontsize=frontsizeall)
plt.xticks(fontsize=frontsizeall)
plt.xlabel('Evaluation Worker Number', fontsize=frontsizeall)
plt.ylabel('Latency (Second)', fontsize=frontsizeall)
plt.legend(title='Target Accuracy (Median)',
           loc='upper right', ncol=2, fontsize=frontsizeall)

lims_x = list(map(lambda x, y: (x, y), df[low_col].to_list(), df[high_col].to_list()))

xss_base = -0.3
xss = []  # position of each line
for i in range(0, len(lims_x)):
    xss.append(xss_base)
    if i >= 1 and (i + 1) % 4 == 0:
        xss_base += 0.4
    else:
        xss_base += 0.2

yss = [i for sub in lims_x for i in sub]

lims_y = []
for ele in xss:
    y_ele = (ele - 0.05, ele + 0.05)
    lims_y.append(y_ele)  # upper line
    lims_y.append(y_ele)  # lower line

for xs, lim in zip(xss, lims_x):
    plot = axvlines(xs, lims=lim, color='black')
for yx, lim in zip(yss, lims_y):
    plot = axhlines(yx, lims=lim, color='black')

plt.grid()
plt.tight_layout()

plt.show()
# f.savefig("latency.pdf", bbox_inches='tight')
