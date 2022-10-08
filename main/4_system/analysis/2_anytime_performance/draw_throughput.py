
from matplotlib import pyplot as plt

x = [ 2, 3, 4, 7]
y = [6.091656235713765, 11.897212744279793, 23.36669667470669, 45.37072296051299]

f, allaxs = plt.subplots(1, 1)


plt.plot(x, y, "D-")

plt.xticks(x, [2, 4, 8, 16])

plt.yticks(fontsize=12)

plt.tight_layout()

frontsizeall = 12
# plt.ylim([0, 40])

plt.yticks(fontsize=frontsizeall)
plt.xticks(fontsize=frontsizeall)
plt.xlabel('Evaluation Worker Number', fontsize=frontsizeall)
plt.ylabel('Throughput (# models/second)', fontsize=frontsizeall)
# plt.legend(title='Target Accuracy (Median)',
#            loc='upper right', ncol=2, fontsize=frontsizeall)

plt.grid()
# plt.show()
f.savefig("tps.pdf", bbox_inches='tight')
