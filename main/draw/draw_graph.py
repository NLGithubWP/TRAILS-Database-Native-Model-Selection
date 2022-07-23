
import json

from statistic_lib import get_rank_after_sort, sort_update

with open('./Logs/201_15k_c10_128_noBN.json', 'r') as readfile:
    data = json.load(readfile)


ori_test_accuracy = []
# scores
fisher = []
grad_norm = []
grad_plain = []
grasp = []
jacob_conv = []
nas_wot = []
ntk_trace_approx = []
snip = []
synflow = []
weight_norm = []

ntk_trace = []
ntk_cond_num = []


for arch_id, info in data.items():
    ori_test_accuracy.append(info["validation_accuracy"])

    fisher.append(info["scores"]["fisher"]["score"])
    grad_norm.append(info["scores"]["grad_norm"]["score"])
    grad_plain.append(info["scores"]["grad_plain"]["score"])
    grasp.append(info["scores"]["grasp"]["score"])
    jacob_conv.append(info["scores"]["jacob_conv"]["score"])
    nas_wot.append(info["scores"]["nas_wot"]["score"])
    ntk_trace_approx.append(info["scores"]["ntk_trace_approx"]["score"])
    snip.append(info["scores"]["snip"]["score"])
    synflow.append(info["scores"]["synflow"]["score"])
    weight_norm.append(info["scores"]["weight_norm"]["score"])
    # ntk_cond_num.append(info["scores"]["ntk_cond_num"]["score"])
    # ntk_trace.append(info["scores"]["ntk_trace"]["score"])

# draw with rank
fisher = get_rank_after_sort(fisher)
grad_norm = get_rank_after_sort(grad_norm)
grad_plain = get_rank_after_sort(grad_plain)
grasp = get_rank_after_sort(grasp)
jacob_conv = get_rank_after_sort(jacob_conv)
nas_wot = get_rank_after_sort(nas_wot)
ntk_trace_approx = get_rank_after_sort(ntk_trace_approx)
snip = get_rank_after_sort(snip)
synflow = get_rank_after_sort(synflow)
weight_norm = get_rank_after_sort(weight_norm)

ntk_cond_num = get_rank_after_sort(ntk_cond_num)
ntk_trace = get_rank_after_sort(ntk_trace)

# batch by b samples
fisher, test_accuracy1 = sort_update(fisher, ori_test_accuracy)
grad_norm, test_accuracy2 = sort_update(grad_norm, ori_test_accuracy)
grad_plain, test_accuracy3 = sort_update(grad_plain, ori_test_accuracy)
grasp, test_accuracy4 = sort_update(grasp, ori_test_accuracy)
jacob_conv, test_accuracy5 = sort_update(jacob_conv, ori_test_accuracy)
nas_wot, test_accuracy6 = sort_update(nas_wot, ori_test_accuracy)
ntk_trace_approx, test_accuracy7 = sort_update(ntk_trace_approx, ori_test_accuracy)
snip, test_accuracy8 = sort_update(snip, ori_test_accuracy)
synflow, test_accuracy9 = sort_update(synflow, ori_test_accuracy)
weight_norm, test_accuracy10 = sort_update(weight_norm, ori_test_accuracy)
ntk_cond_num, test_accuracy11 = sort_update(ntk_cond_num, ori_test_accuracy)
ntk_trace, test_accuracy12 = sort_update(ntk_trace, ori_test_accuracy)

import matplotlib.pyplot as plt

f = plt.figure()

# begin = 31
# end = 45

# plt.scatter(fisher, test_accuracy1, color = 'g', marker='o', label='fisher')
# plt.scatter(grad_norm, test_accuracy2, color = 'r', marker='o', label='grad_norm')
# plt.scatter(grad_plain, test_accuracy3, color = 'c', marker='o', label='grad_plain')
# plt.scatter(grasp, test_accuracy4, color = 'm', marker='o', label='grasp')
# plt.scatter(jacob_conv, test_accuracy5, color = 'y', marker='o', label='jacob_conv')
# plt.scatter(nas_wot, test_accuracy6, color = 'k', marker='o', label='nas_wot')
# plt.scatter(ntk_trace_approx, test_accuracy7, color = 'k', marker='o', label='ntk_trace_approx')
# plt.scatter(snip, test_accuracy8, color = 'lime', marker='o', label='snip')
plt.scatter(synflow, test_accuracy9, color = 'orange', marker='o', label='synflow')
# plt.scatter(weight_norm, test_accuracy10, color = 'indigo', marker='o', label='weight_norm')
# plt.scatter(ntk_cond_num, test_accuracy11, color = 'indigo', marker='o', label='ntk_cond_num')
# plt.scatter(ntk_trace, test_accuracy12, color = 'indigo', marker='o', label='ntk_trace')


# plt.xlim(-3000, 3000)
# plt.ylim(0.8, 1)
plt.grid(axis = "y")
plt.legend()
plt.show()

# f.savefig("foo4.pdf", bbox_inches='tight')



