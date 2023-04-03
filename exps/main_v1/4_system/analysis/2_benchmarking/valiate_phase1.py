import numpy as np

from eva_engine.phase1.run_phase1 import RunPhase1
from utilslibs.parse_pre_res import SimulateTrain
from matplotlib import pyplot as plt


search_space = "nasbench201"
dataset = "cifar10"
# dataset = "cifar100"
# dataset = "ImageNet16-120"

fgt = SimulateTrain(space_name=search_space, total_epoch=200)

y_acc_list_arr = []
x_T_list = list(range(1, 300))
real_time_used_arr = []
planed_time_used_arr = []

for run_id in range(1, 100, 1):

    y_each_run = []

    for N in x_T_list:
        K_models, B1_actual_time_use = RunPhase1.run_phase1_simulate(
            search_space,
            dataset,
            run_id,
            N,
            1)
        best_arch = K_models[0]
        fgt = SimulateTrain(space_name=search_space, total_epoch=200)
        acc_sh_v, _ = fgt.get_ground_truth(arch_id=best_arch, dataset=dataset, epoch_num=None)
        y_each_run.append(acc_sh_v)
    y_acc_list_arr.append(y_each_run)


accuracy_exp = np.array(y_acc_list_arr)
accuracy_q_75 = np.quantile(accuracy_exp, .75, axis=0)
accuracy_q_25 = np.quantile(accuracy_exp, .25, axis=0)
accuracy_mean = np.quantile(accuracy_exp, .5, axis=0)
# plot accuracy
plt.plot(accuracy_mean, label="test_acc")
plt.fill_between(range(len(accuracy_q_25)), accuracy_q_25, accuracy_q_75, alpha=0.1)
plt.tight_layout()
plt.grid()
plt.show()

