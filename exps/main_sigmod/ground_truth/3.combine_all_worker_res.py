import os.path

from utilslibs.io_tools import read_json, write_json
import os.path
import os
import json

# fetch result from server. rename base_line_res to base_line_res_2k5
import os
import json

def combine_json_files(folder_path):
    combined_data = {}
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    data = json.load(f)
                    for key, value in data.items():
                        if key in combined_data:
                            combined_data[key] = {**combined_data[key], **value}
                        else:
                            combined_data[key] = value
    return combined_data


parent_folder = "./A_structure_dataexp_res/criteo/"
output_folder = os.path.dirname(parent_folder)
combined_json = combine_json_files(parent_folder)
print(len(combined_json["criteo"].keys()))

write_json(os.path.join(output_folder, "all_train_baseline_criteo.json"), combined_json)


