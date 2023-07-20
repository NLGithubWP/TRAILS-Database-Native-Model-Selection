# Imports

import numpy as np
import os


# Initialize function to calculate correlation
def calculate_correlation(dataset, search_space, epoch_train):
    print("================================================================")
    print(f" {dataset} + {search_space}")
    print("================================================================")
    # Initialize query objects
    acc_query = SimulateTrain(space_name=search_space)
    score_query = SimulateScore(space_name=search_space, dataset_name=dataset)

    # Get list of all model IDs that have been trained and scored
    trained_models = acc_query.get_all_model_ids(dataset)
    scored_models = score_query.get_all_model_ids(dataset)

    # Find the intersection between trained_models and scored_models
    trained_scored_models = list(set(trained_models) & set(scored_models))

    # Initialize storage for algorithm scores and training results
    model_train_res_lst = []
    all_alg_score_dic = {"nas_wot_add_syn_flow": []}

    # Populate algorithm scores and training results
    for model_id in trained_scored_models:
        score_value = score_query.get_all_tfmem_score_res(model_id)
        acc, _ = acc_query.get_ground_truth(arch_id=model_id, dataset=dataset, epoch_num=epoch_train)

        for alg, value in score_value.items():
            # If the algorithm is not in the dict, initialize its list
            if alg not in all_alg_score_dic:
                all_alg_score_dic[alg] = []

            all_alg_score_dic[alg].append(float(value))

        all_alg_score_dic["nas_wot_add_syn_flow"].append(float(score_value["nas_wot"]) + float(score_value["synflow"]))
        model_train_res_lst.append(acc)

    # Measure the correlation for each algorithm and print the result
    for alg in all_alg_score_dic.keys():
        scores = all_alg_score_dic[alg]
        sorted_indices = np.argsort(scores)

        sorted_ground_truth = [model_train_res_lst[i] for i in sorted_indices]
        sorted_scores = [scores[i] for i in sorted_indices]

        res = CorCoefficient.measure(sorted_scores, sorted_ground_truth)
        print(alg, res[CommonVars.Spearman])

    # Get global ranks, measure correlation and print result for JACFLOW
    if dataset in [Config.Frappe, Config.UCIDataset, Config.UCIDataset]:
        global_rank_score = [score_query.get_score_res(model_id)['nas_wot_synflow']
                             for model_id in trained_scored_models]

        sorted_indices = np.argsort(global_rank_score)
        sorted_ground_truth = [model_train_res_lst[i] for i in sorted_indices]
        sorted_scores = [global_rank_score[i] for i in sorted_indices]

        res = CorCoefficient.measure(sorted_scores, sorted_ground_truth)
        print("JACFLOW", res[CommonVars.Spearman])


# Call the main function
if __name__ == "__main__":
    os.environ.setdefault("base_dir", "../exp_data")
    from src.query_api.api_query import SimulateTrain, SimulateScore
    from utilslibs.measure_tools import CorCoefficient
    from src.common.constant import CommonVars, Config
    # Criteo configuration
    calculate_correlation(Config.Criteo, Config.MLPSP, 9)

    # Frappe configuration
    calculate_correlation(Config.Frappe, Config.MLPSP, 19)

    # UCI configuration
    calculate_correlation(Config.UCIDataset, Config.MLPSP, 0)

    # NB201 + C10
    calculate_correlation(Config.c10, Config.NB201, None)

    # NB201 + C100
    calculate_correlation(Config.c100, Config.NB201, None)

    # NB201 + imageNet
    calculate_correlation(Config.imgNet, Config.NB201, None)

    # NB101 + C10
    calculate_correlation(Config.c10, Config.NB101, None)

