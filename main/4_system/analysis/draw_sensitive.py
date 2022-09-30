

from matplotlib import pyplot as plt


def revserList(a):
    return list(reversed(a))


frontsizeall = 20
def plot_channel(axcost):
    axs = axcost.twinx()

    ntk_app = [0.34, 0.35, 0.36, 0.36, 0.37]
    ntk = [0.36, 0.38, 0.39, 0.40, 0.42]
    ntk_condnum = []

    fisher = [0.37, 0.39, 0.40, 0.41, 0.42]

    snip = [0.63, 0.64, 0.65, 0.65, 0.65]

    grasp = [0.45, 0.53, 0.58, 0.59, 0.59]

    synflow = [0.77, 0.78, 0.77, 0.77, 0.76]

    grad_norm = [0.63, 0.64, 0.65, 0.65, 0.65]
    weight_norm = []

    nas_wot = [0.80, 0.79, 0.79, 0.79, 0.78]

    fix32B_flops = [0.936, 3.68924, 14.643, 58.3, 232.9] # in Gb
    fix32B_params = [201.018, 800.746, 3273, 13076, 52288] # in K

    x = [1, 2, 3, 4, 5]

    axcost.bar(x, fix32B_params, width=0.1)

    axs.plot(x, ntk_app, "o--", label="NTKTraceAppx")
    axs.plot(x, ntk, "o--", label="NTKTrace")
    axs.plot(x, fisher, "o--", label="Fisher")
    axs.plot(x, snip, "o--", label="SNIP")
    axs.plot(x, grasp, "o--", label="GraSP")
    axs.plot(x, synflow, "o--", label="SynFlow")
    axs.plot(x, grad_norm, "o--", label="GradNorm")
    axs.plot(x, nas_wot, "o--", label="NASWOT")

    axs.grid()

    axs.legend(loc='upper right', fontsize=frontsizeall)

    axcost.set_xticklabels([8, 8, 16, 32, 64, 128], fontsize=frontsizeall)
    axcost.set_yticklabels([0, "10k", "20k", "30k", "40k", "50k"], fontsize=frontsizeall)

    axcost.set_xlabel("(a) Channel-Size", fontsize=frontsizeall)
    axcost.set_ylabel("Params bar / K", fontsize=frontsizeall)
    axs.set_ylabel("SRCC line", fontsize=frontsizeall)


def plot_batchsize(axcost):

    axs = axcost.twinx()

    ntk_app = [0.32, 0.34, 0.35, 0.38, 0.37]
    ntk = [0.37, 0.37, 0.38, 0.38, 0.38]
    ntk_condnum = []

    fisher = [0.38, 0.39, 0.39, 0.38, 0.39]

    snip = [0.64, 0.64, 0.64, 0.64, 0.65]

    grasp = [0.54, 0.55, 0.53, 0.52, 0.48]
    synflow = [0.78, 0.78, 0.78, 0.78, 0.78]

    grad_norm = [0.63, 0.64, 0.64, 0.64, 0.64]
    weight_norm = []

    nas_wot = [0.80, 0.79, 0.79, 0.78, 0.76]

    fix16C_flops = [0.922310, 1.845, 3.689, 7.378, 14.756] # in G
    fix16C_params = [800.746, 800.746, 800.746, 800.746, 800.746] # in M

    x = [1, 2, 3, 4, 5]

    axcost.bar(x, fix16C_flops, width=0.1)

    axs.plot(x, revserList(ntk_app), "o--", label="NTKTraceAppx" )
    axs.plot(x, revserList(ntk), "o--", label="NTKTrace" )
    axs.plot(x, revserList(fisher), "o--", label="Fisher" )
    axs.plot(x, revserList(snip), "o--", label="SNIP" )
    axs.plot(x, revserList(grasp), "o--", label="GraSP" )
    axs.plot(x, revserList(synflow), "o--", label="SynFlow" )
    axs.plot(x, revserList(grad_norm), "o--", label="GradNorm" )
    axs.plot(x, revserList(nas_wot), "o--", label="NASWOT" )

    axs.grid()
    axs.legend(loc='upper right', fontsize=frontsizeall)
    axs.set_ylabel("SRCC line", fontsize=frontsizeall)

    axcost.set_xticklabels([8, 8, 16, 32, 64, 128], fontsize=frontsizeall)
    axcost.set_xlabel("(b) Batch-Size", fontsize=frontsizeall)
    axcost.set_ylabel("FLOPs bar / G", fontsize=frontsizeall)


f, allaxs = plt.subplots(1, 2, figsize=(20, 9))

allaxs = allaxs.ravel()

plt.yticks(fontsize=frontsizeall)
plot_channel(allaxs[0])
plt.yticks(fontsize=frontsizeall)
plot_batchsize(allaxs[1])
plt.yticks(fontsize=frontsizeall)

plt.subplots_adjust(hspace=20, wspace=10)
plt.tight_layout()
plt.show()

f.savefig("sen.pdf", bbox_inches='tight')

