
import json
from pprint import pprint


with open('./Logs/101_5k_c10_32.json', 'r') as readfile:
    data = json.load(readfile)


key_list = []
for k, v in data.items():
    pprint(k)
    pprint(v)
    key_list.append(v["architecture_id"])
    break
print(key_list)

test_accuracy = []
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


for arch_id, info in data.items():
    test_accuracy.append(info["test_accuracy"])

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


# draw with rank
fisher = [i[0] for i in sorted(enumerate(fisher), key=lambda x:x[1])]
grad_norm = [i[0] for i in sorted(enumerate(grad_norm), key=lambda x:x[1])]
grad_plain = [i[0] for i in sorted(enumerate(grad_plain), key=lambda x:x[1])]
grasp = [i[0] for i in sorted(enumerate(grasp), key=lambda x:x[1])]
jacob_conv = [i[0] for i in sorted(enumerate(jacob_conv), key=lambda x:x[1])]
nas_wot = [i[0] for i in sorted(enumerate(nas_wot), key=lambda x:x[1])]
ntk_trace_approx = [i[0] for i in sorted(enumerate(ntk_trace_approx), key=lambda x:x[1])]
snip = [i[0] for i in sorted(enumerate(snip), key=lambda x:x[1])]
synflow = [i[0] for i in sorted(enumerate(synflow), key=lambda x:x[1])]
weight_norm = [i[0] for i in sorted(enumerate(weight_norm), key=lambda x:x[1])]


# fisher = [1/ele for ele in fisher]
# grad_norm = [1/ele for ele in grad_norm]
# grad_plain = [1/ele for ele in grad_plain]
# jacob_conv = [1/ele for ele in jacob_conv]
# ntk_trace_approx = [1/ele for ele in ntk_trace_approx]
# snip = [1/ele for ele in snip]


# fisher = [100 * (float(i)-min(fisher))/(max(fisher)-min(fisher)) for i in fisher]
# grad_norm = [100 * (float(i)-min(grad_norm))/(max(grad_norm)-min(grad_norm)) for i in grad_norm]
# grad_plain = [100 * (float(i)-min(grad_plain))/(max(grad_plain)-min(grad_plain)) for i in grad_plain]
# grasp = [100 * (float(i)-min(grasp))/(max(grasp)-min(grasp)) for i in grasp]
# jacob_conv = [100 * (float(i)-min(jacob_conv))/(max(jacob_conv)-min(jacob_conv)) for i in jacob_conv]
# nas_wot = [100 * (float(i)-min(nas_wot))/(max(nas_wot)-min(nas_wot)) for i in nas_wot]
# ntk_trace_approx = [100 * (float(i)-min(ntk_trace_approx))/(max(ntk_trace_approx)-min(ntk_trace_approx)) for i in ntk_trace_approx]
# snip = [100 * (float(i)-min(snip))/(max(snip)-min(snip)) for i in snip]
# synflow = [100 * (float(i)-min(synflow))/(max(synflow)-min(synflow)) for i in synflow]
# weight_norm = [100 * (float(i)-min(weight_norm))/(max(weight_norm)-min(weight_norm)) for i in weight_norm]


fisher = [x for _, x in sorted(zip(test_accuracy, fisher))]
grad_norm = [x for _, x in sorted(zip(test_accuracy, grad_norm))]
grad_plain = [x for _, x in sorted(zip(test_accuracy, grad_plain))]
grasp = [x for _, x in sorted(zip(test_accuracy, grasp))]
jacob_conv = [x for _, x in sorted(zip(test_accuracy, jacob_conv))]
nas_wot = [x for _, x in sorted(zip(test_accuracy, nas_wot))]
ntk_trace_approx = [x for _, x in sorted(zip(test_accuracy, ntk_trace_approx))]
snip = [x for _, x in sorted(zip(test_accuracy, snip))]
synflow = [x for _, x in sorted(zip(test_accuracy, synflow))]
weight_norm = [x for _, x in sorted(zip(test_accuracy, weight_norm))]

test_accuracy = sorted(test_accuracy)



begin = 31
end = 45
x = range(end-begin)

import matplotlib.pyplot as plt

f = plt.figure()


# plt.plot(x, [ele* 100 for ele in test_accuracy[begin: end]], color = 'b', marker='o', label='test_accuracy')
plt.plot(x, fisher[begin: end], color = 'g', marker='o', label='fisher')
plt.plot(x, grad_norm[begin: end], color = 'r', marker='o', label='grad_norm')
plt.plot(x, grad_plain[begin: end], color = 'c', marker='o', label='grad_plain')
plt.plot(x, grasp[begin: end], color = 'm', marker='o', label='grasp')

plt.plot(x, jacob_conv[begin: end], color = 'y', marker='o', label='jacob_conv')

plt.plot(x, nas_wot[begin: end], color = 'k', marker='o', label='nas_wot')
plt.plot(x, ntk_trace_approx[begin: end], color = 'yellow', marker='o', label='ntk_trace_approx')

plt.plot(x, snip[begin: end], color = 'lime', marker='o', label='snip')
plt.plot(x, synflow[begin: end], color = 'orange', marker='o', label='synflow')
plt.plot(x, weight_norm[begin: end], color = 'indigo', marker='o', label='weight_norm')



# plt.plot(x, [ele* 100 for ele in test_accuracy[begin: end]], color = 'b', marker='o', label='test_accuracy')
# plt.plot(x, [ele * 100 for ele in fisher[begin: end]], color = 'g', marker='o', label='fisher')
# plt.plot(x, [ele * 10 for ele in grad_norm[begin: end]], color = 'r', marker='o', label='grad_norm')
# plt.plot(x, jacob_conv[begin: end], color = 'y', marker='o', label='jacob_conv')
# plt.plot(x, [ele * 100 for ele in ntk_trace_approx[begin: end]], color = 'yellow', marker='o', label='ntk_trace_approx')
# plt.plot(x, [ele * 10 for ele in snip[begin: end]], color = 'lime', marker='o', label='snip')
# plt.plot(x, [ele * 100 for ele in synflow[begin: end]], color = 'orange', marker='o', label='synflow')


plt.xlim(-8, 14)
plt.grid(axis = "y")
plt.legend()
plt.show()

f.savefig("foo3.pdf", bbox_inches='tight')

