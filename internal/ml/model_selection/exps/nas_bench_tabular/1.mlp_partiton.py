from utilslibs.io_tools import read_json, write_json

all_trained = read_json("./exps/main_v2/ground_truth/sampled_models_10000_models.json")
all_model_id = read_json("./exps/main_v2/ground_truth/sampled_models_all.json")


print("trained results = ", len(list(all_trained.keys())))
print("all_model_id = ", len(list(all_model_id.keys())))

result = {}
index = 0

for key, value in all_model_id.items():
    if value not in all_trained.values():
        result[index] = value
        index += 1

write_json("./all_models_uci_left_159k.json", result)












