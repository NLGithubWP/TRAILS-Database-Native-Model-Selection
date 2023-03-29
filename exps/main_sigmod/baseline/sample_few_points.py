from utilslibs.compute import log_scale_x_array
from utilslibs.io_tools import read_json


dataset = "frappe"
train_epoch = 19

checkpoint_file = f"./exps/main_sigmod/analysis/res_train_base_line_{dataset}_epoch_{train_epoch}.json"

read_json(checkpoint_file)

budget_array = log_scale_x_array(num_points=25, max_minute=max_minute)

result_with_fixed_budget = {
    "baseline_time_budget": [],
    "baseline_acc": []
}

for run_id in range(len(result["baseline_time_budget"])):
    record_time_list = []
    record_acc_list = []
    for _time_use in budget_array:
        record_acc = -1
        for index in range(len(result["baseline_time_budget"][run_id])-1, 0,-1):
            cur_time_usage = result["baseline_time_budget"][run_id][index]
            cur_acc = result["baseline_acc"][run_id][index]

            if _time_use > cur_time_usage:
                record_acc = cur_acc
                break
        if record_acc != -1:
            record_time_list.append(_time_use)
            record_acc_list.append(record_acc)

    result_with_fixed_budget["baseline_time_budget"].append(record_time_list)
    result_with_fixed_budget["baseline_acc"].append(record_acc_list)

