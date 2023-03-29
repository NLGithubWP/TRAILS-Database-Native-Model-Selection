import numpy as np

from utilslibs.draw_lib import draw_structure_data_anytime
from utilslibs.io_tools import read_json


# 'frappe, criteo, uci_diabetes'
dataset = "frappe"

train_result19 = read_json(f"./res_train_base_line_{dataset}_epoch_19.json")
system_result_de_duplication = read_json(f"./res_end_2_end_{dataset}_1000.0.json")


all_lines = [
    [train_result19["baseline_time_budget"][0], train_result19["baseline_acc"], "Train-Based-FullyTrain"],
    [system_result_de_duplication["sys_time_budget"], system_result_de_duplication["sys_acc"], "FIRMEST"],
]

draw_structure_data_anytime(all_lines, "frappe", "frappe-test")





