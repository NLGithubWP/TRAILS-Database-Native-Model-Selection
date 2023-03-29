import numpy as np

from exps.main_sigmod.common.shared_args import parse_arguments
from utilslibs.draw_lib import draw_structure_data_anytime
from utilslibs.io_tools import read_json

args = parse_arguments()

print(f"reading from {args.saved_result}")

# 'frappe, criteo, uci_diabetes'
dataset = "frappe"

train_result19 = read_json(f"./exps/main_sigmod/analysis/res_train_base_line_{dataset}_epoch_19.json")
system_result_de_duplication = read_json(args.saved_result)


all_lines = [
    [train_result19["baseline_time_budget"][0], train_result19["baseline_acc"], "Train-Based-FullyTrain"],
    [system_result_de_duplication["sys_time_budget"], system_result_de_duplication["sys_acc"], "FIRMEST"],
]

draw_structure_data_anytime(all_lines, "frappe", f"{args.saved_result[:-4]}")





