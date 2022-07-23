
from tqdm import tqdm


def vote_between_two_arch(arch1_info: dict, arch2_info: dict, metric: list):

    gt_differ = arch1_info["accuracy"] - arch2_info["accuracy"]

    metrics_differ = []
    for m_name in metric:
        metrics_differ.append(arch1_info[m_name] - arch1_info[m_name])

    vote_res = sum([1 for ele in metrics_differ if ele > 0])

    if vote_res >= len(metrics_differ)/2:
        return 1
    else:
        return 0
























