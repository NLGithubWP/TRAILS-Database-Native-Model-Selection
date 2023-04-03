
import pickle
from scipy import stats
from prettytable import PrettyTable

from utilslibs.measure_tools import CorCoefficient

t = None
all_ds = {}
all_acc = {}
allc = {}
all_metrics = {}
all_runs = {}
metric_names = ['grad_norm', 'snip', 'grasp', 'fisher', 'synflow', 'jacob_cov']
for fname, rname in [(
        '../trails/data/nasbench2/nb2_cf10_seed42_dlrandom_dlinfo1_initwnone_initbnone.p',
        'CIFAR10'),
    (
            '../trails/data/nasbench2/nb2_cf100_seed42_dlrandom_dlinfo1_initwnone_initbnone.p',
                     'CIFAR100'),
                     (
                     '../trails/data/nasbench2/nb2_im120_seed42_dlrandom_dlinfo1_initwnone_initbnone.p',
                     'ImageNet16-120')]:
    runs = []
    f = open(fname, 'rb')
    while (1):
        try:
            runs.append(pickle.load(f))
        except EOFError:
            break
    f.close()
    print(fname, len(runs))

    all_runs[fname] = runs
    all_ds[fname] = {}
    metrics = {}
    for k in metric_names:
        metrics[k] = []
    acc = []

    if t is None:
        hl = ['Dataset']
        hl.extend(metric_names)
        t = PrettyTable(hl)

    for r in runs:
        for k, v in r['logmeasures'].items():
            if k in metrics:
                metrics[k].append(v)
        acc.append(r['testacc'])

    all_ds[fname]['metrics'] = metrics
    all_ds[fname]['acc'] = acc

    res = []
    crs = {}
    for k in hl:
        if k == 'Dataset':
            continue
        v = metrics[k]
        cr = abs(stats.spearmanr(acc, v, nan_policy='omit').correlation)
        # print(f'{k} = {cr}')
        res.append(round(cr, 3))
        crs[k] = cr

    ds = rname
    all_acc[ds] = acc
    allc[ds] = crs
    t.add_row([ds] + res)

    all_metrics[ds] = metrics
print(t)

#
# from tqdm import tqdm
# votes = {}
# def vote(mets, gt):
#     numpos = 0
#     for m in mets:
#         numpos += 1 if m > 0 else 0
#     if numpos >= len(mets)/2:
#         sign = +1
#     else:
#         sign = -1
#     return sign*gt
#
# for ds in all_acc.keys():
#     num_pts = 15625
#     #num_pts = 1000
#     tot=0
#     right=0
#     for i in tqdm(range(num_pts)):
#         for j in range(num_pts):
#             if i!=j:
#                 diff = all_acc[ds][i] - all_acc[ds][j]
#                 if diff == 0:
#                     continue
#                 diffsyn = []
#                 for m in ['synflow', 'jacob_cov', 'snip']:
#                     diffsyn.append(all_metrics[ds][m][i] - all_metrics[ds][m][j])
#                 same_sign = vote(diffsyn, diff)
#                 right += 1 if same_sign > 0 else 0
#                 tot += 1
#     votes[ds.lower() if 'CIFAR' in ds else ds] = right/tot
# print('votes correlation: ', votes)

c10a = all_acc["CIFAR10"]
s10m = all_metrics["CIFAR10"]

from statistic_lib import get_rank_after_sort
synflow = get_rank_after_sort(s10m["synflow"])
jacob_cov = get_rank_after_sort(s10m["jacob_cov"])
snip = get_rank_after_sort(s10m["snip"])



res_mtr = []
res_acc = []
for i in range(15625):
    vote_score = (synflow[i] + jacob_cov[i] + snip[i])/3

    res_mtr.append(vote_score)
    res_acc.append(c10a[i])


print(CorCoefficient.measure(res_acc, res_mtr, "Spearman"))









