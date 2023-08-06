import numpy as np
import os
import numpy as np


# Initialize function to calculate correlation
def calculate_correlation(dataset, search_space, epoch_train):
    print("\n================================================================")
    print(f" {dataset} + {search_space}")
    print("================================================================")
    # Initialize query objects
    acc_query = SimulateTrain(space_name=search_space)
    score_query = SimulateScore(space_name=search_space, dataset_name=dataset)

    # Get list of all model IDs that have been trained and scored
    trained_models = acc_query.query_all_model_ids(dataset)
    scored_models = score_query.query_all_model_ids(dataset)

    # Find the intersection between trained_models and scored_models
    trained_scored_models = list(set(trained_models) & set(scored_models))

    # Initialize storage for algorithm scores and training results
    model_train_res_lst = []
    all_alg_score_dic = {"nas_wot_add_syn_flow": []}

    # Populate algorithm scores and training results
    for model_id in trained_scored_models:
        score_value = score_query.query_all_tfmem_score(model_id)
        acc, _ = acc_query.get_ground_truth(arch_id=model_id, dataset=dataset, epoch_num=epoch_train)

        for alg, value in score_value.items():
            # If the algorithm is not in the dict, initialize its list
            if alg not in all_alg_score_dic:
                all_alg_score_dic[alg] = []

            all_alg_score_dic[alg].append(float(value))

        if "nas_wot" in score_value:
            all_alg_score_dic["nas_wot_add_syn_flow"].append(
                float(score_value["nas_wot"]) + float(score_value["synflow"]))
        else:
            all_alg_score_dic["nas_wot_add_syn_flow"].append(0)

        model_train_res_lst.append(acc)

    # Measure the correlation for each algorithm and print the result
    print("--------------------------------------------------")
    for alg in all_alg_score_dic.keys():
        scores = all_alg_score_dic[alg]
        sorted_indices = np.argsort(scores)

        sorted_ground_truth = [model_train_res_lst[i] for i in sorted_indices]
        sorted_scores = [scores[i] for i in sorted_indices]
        res = CorCoefficient.measure(sorted_scores, sorted_ground_truth)
        print(alg, res[CommonVars.Spearman])

    # Get global ranks, measure correlation and print result for JACFLOW
    # if dataset in [Config.Frappe, Config.UCIDataset, Config.Criteo]:
    try:
        global_rank_score = [score_query.query_tfmem_rank_score(model_id)['nas_wot_synflow']
                             for model_id in trained_scored_models]

        sorted_indices = np.argsort(global_rank_score)
        sorted_ground_truth = [model_train_res_lst[i] for i in sorted_indices]
        sorted_scores = [global_rank_score[i] for i in sorted_indices]

        res = CorCoefficient.measure(sorted_scores, sorted_ground_truth)
        print("JACFLOW", res[CommonVars.Spearman])
    except:
        print("JACFLOW not provided")


def average_rank(data):
    # Take the absolute value of each element
    abs_data = np.abs(data)

    # Rank each value within its row based on the absolute value, from large to small
    row_ranks = abs_data.argsort(axis=1)[:, ::-1].argsort(axis=1)

    # Add 1 to each rank so that the ranking starts from 1
    row_ranks += 1

    # Compute the average rank for each column
    avg_ranks = row_ranks.mean(axis=0)

    return avg_ranks


# Call the main function
if __name__ == "__main__":
    # this is pre-computed
    data = np.array([
        [0.45, 0.61, -0.77, 0.54, 0.13, 0.48, -0.27, 0.68, 0.77, 0.80],
        [0.39, 0.63, -0.56, 0.37, 0.31, 0.21, -0.23, 0.62, 0.68, 0.76],
        [0.32, 0.69, -0.66, 0.46, 0.01, 0.41, -0.18, 0.78, 0.74, 0.77]
    ])

    avg_ranks = average_rank(data)
    print(avg_ranks)

    os.environ.setdefault("base_dir", "../exp_data")
    from src.query_api.interface import SimulateTrain, SimulateScore
    from src.tools.correlation import CorCoefficient
    from src.common.constant import CommonVars, Config

    # Frappe configuration
    calculate_correlation(Config.Frappe, Config.MLPSP, 19)

    # UCI configuration
    calculate_correlation(Config.UCIDataset, Config.MLPSP, 0)

    # Criteo configuration
    calculate_correlation(Config.Criteo, Config.MLPSP, 9)

    # NB101 + C10
    calculate_correlation(Config.c10, Config.NB101, None)

    # NB201 + C10
    calculate_correlation(Config.c10, Config.NB201, None)

    # NB201 + C100
    calculate_correlation(Config.c100, Config.NB201, None)

    # NB201 + imageNet
    calculate_correlation(Config.imgNet, Config.NB201, None)
