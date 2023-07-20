
from src.query_api.api_query import SimulateTrain
from src.utilslibs.io_tools import read_json


def gen_list_run_infos( data):
    result = []
    for run_id, value in data.items():
        res = EachRunInfo(run_id=run_id,
                          x_axis_time=data[run_id]["x_axis_time"],
                          y_axis_top10_model=data[run_id]["y_axis_top10_models"])
        result.append(res)
    return result


def get_current_best_acc(acc_list):
    res = []
    for ele in acc_list:
        if len(res) == 0:
            res.append(ele)
            continue
        if ele > res[-1]:
            res.append(ele)
        else:
            res.append(res[-1])
    return res


# accepting output of online system
# each run info, {x_list: time list, y: acc list,}
class EachRunInfo:
    def __init__(self, run_id, x_axis_time, y_axis_top10_model=None, y_current_best_accs=None):
        self.run_id = run_id
        self.x_axis_time = x_axis_time
        self.y_axis_top10_model = y_axis_top10_model
        self.y_current_best_accs = y_current_best_accs

    def get_current_best_acc(self, index, fgt: SimulateTrain):
        if self.y_current_best_accs is None:
            current_top_10 = self.y_axis_top10_model[index]
            high_acc, _ = fgt.get_high_acc_top_10(current_top_10)
            return high_acc
        else:
            return self.y_current_best_accs[index] * 0.01


# accepting output of online system
def measure_time_usage(file_paths_list):
    total_t = []
    model_score_t = []
    model_gene_t = []

    for file in file_paths_list:
        data = read_json(file)

        total_t.extend(data["total_t"])
        model_score_t.extend(data["model_score_t"])
        model_gene_t.extend(data["model_gene_t"])

    print("Generating model using ", sum(model_gene_t) / sum(total_t))  # 0.533766836811139
    print("Scoring model using", sum(model_score_t) / sum(total_t))  # 0.46619504843604753

