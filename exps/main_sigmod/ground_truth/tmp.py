import os.path

from utilslibs.io_tools import read_json, write_json
import os.path
import os
import json

# fetch result from server. rename base_line_res to base_line_res_2k5
import os
import json


parent_folder = "../firmest_data/result_base/mlp_results/uci_diabetes/score_uci_diabetes_batch_size_32_all_metrics.json"

res = read_json(parent_folder)

print(len(res.keys()))



