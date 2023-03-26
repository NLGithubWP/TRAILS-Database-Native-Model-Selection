import os
import json

from utilslibs.io_tools import write_json

# this is to parse the log into the result.json
dir = "./base_line_res_uci/Logs"
all_files = os.listdir(dir)


def parse_worker_id(s):
    parts = s.split("_")

    for i, part in enumerate(parts):
        if part == "wkid":
            value_between_wkid_and_underscore = parts[i + 1]
            break
    return value_between_wkid_and_underscore


for log_file in all_files:
    worker_id = parse_worker_id(log_file)
    base_line_log = {"uci_diabetes": {}}
    save_file = f"./base_line_res_uci/train_baseline_uci_diabetes_wkid_{worker_id}.json"

    with open(os.path.join(dir, log_file), 'r') as f:
        for line in f:
            if '---- info: {' in line:
                # extract the contents after "---- info: {"
                contents = line.split('---- info: ')[1]
                # do something with the contents
                obj = json.loads(contents)
                base_line_log["uci_diabetes"].update(obj)

    write_json(save_file, base_line_log)




