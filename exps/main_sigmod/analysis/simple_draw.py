import numpy as np

from utilslibs.draw_lib import draw_structure_data_anytime
from utilslibs.io_tools import read_json


# 'frappe, criteo, uci_diabetes'
dataset = "frappe"

train_result19 = read_json(f"./res_train_base_line_{dataset}_epoch_19.json")
train_result5 = read_json(f"./res_train_base_line_{dataset}_epoch_5.json")
# train_result1 = read_json(f"./res_train_base_line_{dataset}_epoch_1.json")
system_result_de_duplication = read_json(f"./res_end_2_end_{dataset}_1e4_de_dup.json")
# system_result_with_duplication = read_json(f"./res_end_2_end_{dataset}_1e4_with_dup.json")
training_free_result = read_json(f"./res_end_2_end_{dataset}_p1.json")


all_lines = [
    [[ele/60 for ele in train_result19["baseline_time_budget"][0]], train_result19["baseline_acc"], "Train-Based-FullyTrain"],
    # [[ele/60 for ele in train_result5["baseline_time_budget"][0]], train_result5["baseline_acc"], "Train-Based-EarlyStopping"],
    # [system_result_with_duplication["sys_time_budget"], system_result_with_duplication["sys_acc"], "FIRMEST_dup"],
    [system_result_de_duplication["sys_time_budget"], system_result_de_duplication["sys_acc"], "FIRMEST"],
    [training_free_result["sys_time_budget"], training_free_result["sys_acc"], "Train-Free"],
]

draw_structure_data_anytime(all_lines, "frappe", "frappe-test")





