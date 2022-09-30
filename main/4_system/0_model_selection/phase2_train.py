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

n = 10

harvest_acc = []
harvest_budget = []

for mu in [2, 3, 4, 5, 6]:
# for mu in [4, 5]:

    inner_map_acc = []
    inner_map_budget = []

    x_plot = []
    y_successful_rate = []

    # print(f"when mu={mu}, r={min_epoch}, latest Budget usage is {least_time_usage}")

    for min_epoch in [4, 12, 20, 28, 36, 44, 52]:
        _, min_budget_required, _ = SH.allocate_min_total_budget(n, mu, min_epoch)
        inner_map_budget.append(min_budget_required)

        x_plot.append(min_epoch)
        correct_sh = 0
        total = 0

        for _ in range(1000):
            candidates = [randint(1, 15600) for i in range(n)]

            # found by sh
            best_arch = sh.SuccessiveHalving(min_epoch, candidates, mu)
            acc_sh_v, _ = fgt.get_ground_truth(best_arch)

            # the real best one
            real_best_arch = fgt.get_best_arch_id(candidates)
            acc_gt_v, _ = fgt.get_ground_truth(real_best_arch)

            total += 1
            if int(best_arch) == int(real_best_arch):
                correct_sh += 1

        y_successful_rate.append(correct_sh / total)
        inner_map_acc.append(correct_sh / total)

    harvest_acc.append(inner_map_acc)
    harvest_budget.append(inner_map_budget)

    f, allaxs = plt.subplots(1, 1)

    print(x_plot)
    plt.plot(x_plot, y_successful_rate, "*-")

    frontsizeall = 12

    plt.yticks(fontsize=frontsizeall)
    plt.xticks(fontsize=frontsizeall)
    plt.xlabel('Min Budget for each model (epoch)', fontsize=frontsizeall)
    plt.ylabel('Successful Rate', fontsize=frontsizeall)

    plt.title(mu)
    plt.grid()
    plt.legend()
    plt.show()

plt.show()
# f.savefig("201_sh.pdf", bbox_inches='tight')

print(harvest_acc)
print(harvest_budget)













