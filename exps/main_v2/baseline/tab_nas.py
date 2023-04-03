import os


os.environ.setdefault("base_dir", "../exp_data/")
from common.constant import Config
from query_api.query_model_gt_acc_api import GTMLP

gtm = GTMLP()

potention_lst = [
"48-384-16-176", "32-144-24-112", "48-112-8-80"
"96-144-32-240", "96-384-48-16", "48-128-16-112",
"48-128-24-32", "48-144-16-48", "48-112-16-144",
"48-128-16-112", "48-80-32-112", "48-160-8-48",

"48-112-16-24", "32-112-24-224", "48-160-32-144",
"48-256-24-112", "80-208-32-64"
]

for ele in [
    [Config.Criteo, 9],
    [Config.Frappe, 19],
    [Config.UCIDataset, 0]]:

    dataset = ele[0]
    epoch = ele[1]

    acclst = []
    for best_arch in potention_lst:
        try:
            acc, time_use = gtm.get_valid_auc(
                best_arch, dataset, epoch)
        except:
            continue
        acclst.append(acc)
    Facc = max(acclst)

    # todo: 5.8 is got from the figure12 in TabNAS paper.
    print(f"TabNAS dataset={dataset}, time_usage={5.8*time_use}, acc={Facc}")





