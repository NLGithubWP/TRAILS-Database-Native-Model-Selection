import os

from common.constant import Config
from utilslibs.io_tools import read_json

base_dir_folder = os.environ.get("base_dir")
if base_dir_folder is None:base_dir_folder = os.getcwd()
base_dir = os.path.join(base_dir_folder, "result_base/mlp_training_append")

base_dir_folder = os.environ.get("base_dir")
if base_dir_folder is None:base_dir_folder = os.getcwd()
base_dir = os.path.join(base_dir_folder, "result_base/mlp_training_append")

print("mlp gt_api running at {}".format(base_dir))
train_frappe = os.path.join(base_dir, "frappe_train.json")
train_uci_diabetes = os.path.join(base_dir, "uci_diabetes_train.json")
train_criteo = os.path.join(base_dir, "criteo_train.json")

# training-free result
score_frappe = os.path.join(base_dir, "frappe_score_all.json")
score_uci_diabetes = os.path.join(base_dir, "uci_diabetes_score_all.json")
score_criteo = os.path.join(base_dir, "criteo_score_all.json")


def get_mlp_training_res(dataset: str):

    if dataset == Config.Frappe:
        data = read_json(train_frappe)
    elif dataset == Config.Criteo:
        data = read_json(train_criteo)
    elif dataset == Config.UCIDataset:
        data = read_json(train_uci_diabetes)
    else:
        print(f"Cannot read dataset {dataset} of file")
        raise

    return data


def get_mlp_score_res(dataset: str):
    if dataset == Config.Frappe:
        data = read_json(score_frappe)
    elif dataset == Config.Criteo:
        data = read_json(score_uci_diabetes)
    elif dataset == Config.UCIDataset:
        data = read_json(score_criteo)
    else:
        print(f"Cannot read dataset {dataset} of file")
        raise

    return data








