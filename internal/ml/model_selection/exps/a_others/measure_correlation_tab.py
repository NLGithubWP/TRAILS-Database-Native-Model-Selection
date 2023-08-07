import os

from src.common.constant import CommonVars
from src.tools.io_tools import read_json
from src.tools.correlation import CorCoefficient
import numpy as np

# Criteo
# train_dir = "../exp_data/tab_data/criteo/all_train_baseline_criteo_only_8k.json"
# score_dir = "../exp_data/tab_data/criteo/score_criteo_batch_size_32.json"
# epoch_train = "9"

# Frappe
train_dir = "../exp_data/tab_data/frappe/all_train_baseline_frappe.json"
score_dir = "../exp_data/tab_data/frappe/score_frappe_batch_size_32_local_finish_all_models.json"
epoch_train = "19"

# UCI
# train_dir = "../exp_data/tab_data/uci_diabetes/all_train_baseline_uci_160k_40epoch.json"
# score_dir = "../exp_data/tab_data/uci_diabetes/score_uci_diabetes_batch_size_32_all_metrics.json"
# epoch_train = "0"

# Read the training and scoring results
train_res = read_json(train_dir)
score_res = read_json(score_dir)

# Get the dataset
dataset = list(train_res.keys())[0]

# Check and print the number of trained models and scored models
num_train_models = len(list(train_res[dataset].keys()))
num_scored_models = len(list(score_res.keys()))
print(f"num_train_models={num_train_models}, num_scored_models={num_scored_models}")

# Decide which models to use based on the number of models trained and scored
if num_train_models > num_scored_models:
    all_models_ids = list(score_res.keys())
else:
    all_models_ids = list(train_res[dataset].keys())

# Print the maximum training epoch
print(f"1. epoch train max is {epoch_train}")

# Initialize dictionary for algorithm scores
all_alg_score_dic = {}
for alg, _ in score_res[all_models_ids[0]].items():
    all_alg_score_dic[alg] = []
all_alg_score_dic["nas_wot_syn_flow"] = []
print(f"2. all alg list {all_alg_score_dic}")

# Populate the algorithm scores and training results list
model_train_res_lst = []
for model_id in all_models_ids:
    score_value = score_res[model_id]
    for alg, value in score_value.items():
        all_alg_score_dic[alg].append(value)
    all_alg_score_dic["nas_wot_syn_flow"].append(score_value["nas_wot"] + score_value["synflow"])
    model_train_res_lst.append(train_res[dataset][model_id][epoch_train]["valid_auc"])

# Calculate and print the correlation for each algorithm
for alg in all_alg_score_dic.keys():
    scores = all_alg_score_dic[alg]
    ground_truth = model_train_res_lst
    sorted_indices = np.argsort(scores)
    sorted_ground_truth = [ground_truth[i] for i in sorted_indices]
    sorted_scores = [scores[i] for i in sorted_indices]
    print(f"3. measure the correlation between {alg} and training_result ")
    res = CorCoefficient.measure(sorted_scores, sorted_ground_truth)
    print(alg, res[CommonVars.Spearman])

# Set environment for JACFLOW
os.environ.setdefault("base_dir", "../exp_data")
from src.query_api.query_api_mlp import GTMLP
gt_mlp = GTMLP()

# Get global ranks and measure the correlation for JACFLOW
global_rank = []
for model_id in all_models_ids:
    global_rank.append(gt_mlp.get_global_rank_score(model_id, dataset)["nas_wot_synflow"])
sorted_indices = np.argsort(global_rank)
sorted_ground_truth = [model_train_res_lst[i] for i in sorted_indices]
sorted_scores = [global_rank[i] for i in sorted_indices]
print(f"3. measure the correlation between JACFLOW and training_result ")
res = CorCoefficient.measure(sorted_scores, sorted_ground_truth)
print("JACFLOW", res[CommonVars.Spearman])
