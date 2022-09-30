import os.path
from random import randint
from matplotlib import pyplot as plt
from api_local.parse_pre_res import FetchGroundTruth
from common.constant import Config
from eva_engine.phase2.p2evaluator import P2Evaluator
from eva_engine.phase2.sh import SH
from utilslibs.tools import read_json
import scipy.stats as ss


fgt = FetchGroundTruth(Config.NB201)

evaluator = P2Evaluator(fgt)
sh = SH(evaluator)

base_d = os.getcwd()
pre_score_dir = os.path.join(base_d,
                             "result_base/result_append/CIFAR10_15625/union/201_15625_c10_bs32_ic16_unionBest.json")

pre_scored_data = read_json(pre_score_dir)


def get_pre_score(arch_id, alg_name):
    score_ = float(pre_scored_data[str(arch_id)][alg_name])
    return score_


def get_rank_sc1ore(score_list):
    rank_index_list = ss.rankdata(score_list)
    return rank_index_list


def get_rank_score(candidate):

    naswot_list = []
    synflow_list = []
    acc_list = []

    for cand in candidate:
        naswot_list.append(get_pre_score(cand, "nas_wot"))
        synflow_list.append(get_pre_score(cand, "synflow"))
        acc_gt_v, _ = fgt.get_ground_truth(cand, epoch_num=12)
        acc_list.append(acc_gt_v)

    nll = get_rank_sc1ore(naswot_list)
    syl = get_rank_sc1ore(synflow_list)
    acl = get_rank_sc1ore(acc_list)

    res = -1
    max_index = 0
    for i in range(len(nll)):
        rank_score = nll[i] + syl[i] + acl[i]
        if rank_score > res:
            res = rank_score
            max_index = i
    return candidate[max_index]


B = 500
r = 1
mu = 2

correct_rank = 0
correct_sh = 0
total = 0

acc_rk = []
acc_found_sh = []
acc_gt = []

diff_rank = 0
diff_sh = 0

for _ in range(1000):
    candidates = [randint(1, 15600) for i in range(10)]

    # found by rank-score
    rank_best_arch_id = get_rank_score(candidates)
    acc_rk_v, _ = fgt.get_ground_truth(rank_best_arch_id)

    # found by sh
    best_arch = sh.SuccessiveHalving(B, r, candidates, mu)
    acc_sh_v, _ = fgt.get_ground_truth(best_arch)

    # the real best one
    real_best_arch = fgt.get_best_arch_id(candidates)
    acc_gt_v, _ = fgt.get_ground_truth(real_best_arch)

    # draw the graph
    acc_rk.append(acc_rk_v)
    acc_found_sh.append(acc_sh_v)
    acc_gt.append(acc_gt_v)

    # record diff
    diff_rank += abs(acc_rk_v - acc_gt_v) * 100
    diff_sh += abs(acc_sh_v - acc_gt_v) * 100

    total += 1
    if int(best_arch) == int(real_best_arch):
        correct_sh += 1

    if int(rank_best_arch_id) == int(real_best_arch):
        correct_rank += 1

print("diff_rank", diff_rank/len(acc_rk))
print("diff_sh", diff_sh/len(acc_rk))

print("rank correct", correct_rank / total)
print("sh correct", correct_sh / total)


plt.plot(acc_rk, label="acc_found_rank")
plt.plot(acc_found_sh, label="acc_found_sh")
plt.plot(acc_gt, label="acc_gt")

plt.legend()
plt.show()














