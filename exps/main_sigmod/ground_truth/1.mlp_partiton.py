from utilslibs.io_tools import read_json, write_json

all_trained = read_json("./A_structure_dataexp_res/criteo0401/all_train_baseline_criteo.json")
all_model_id = read_json("./exps/main_sigmod/ground_truth/sampled_models_10000_models.json")


print("trained results = ", len(list(all_trained["criteo"].keys())))
print("all_model_id = ", len(list(all_model_id.keys())))

result = {}
index = 0

for key, value in all_model_id.items():
    if value not in all_trained["criteo"].keys():
        result[index] = value
        index += 1

write_json("./all_models_criteo_left.json", result)











