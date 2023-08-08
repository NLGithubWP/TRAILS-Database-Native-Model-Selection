import numpy as np
import os
import numpy as np
from typing import List


# Initialize function to calculate correlation
def calculate_correlation(dataset, search_space, epoch_train, srcc_top_k: List = [1]):
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
    for topp in srcc_top_k:
        print("--------------------------------------------------")
        for alg in all_alg_score_dic.keys():
            scores = all_alg_score_dic[alg]
            sorted_indices = np.argsort(scores)[- int(topp * len(scores)):]
            sorted_ground_truth = [model_train_res_lst[i] for i in sorted_indices]
            sorted_scores = [scores[i] for i in sorted_indices]
            res = CorCoefficient.measure(sorted_scores, sorted_ground_truth)
            print(f"Top {topp, len(sorted_indices)}, {alg}, {res[CommonVars.Spearman]}")

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

    # Frappe configuration, here also measure SRCC of top 0.2% -> 0.8%
    calculate_correlation(Config.Frappe, Config.MLPSP, 19, srcc_top_k=[0.002, 0.004, 0.006, 0.008, 1])

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

    """
    Here is the output:
    ================================================================
     frappe + mlp_sp
    ================================================================
    Loading ../exp_data/tab_data/frappe/all_train_baseline_frappe.json...
    Loading ../exp_data/tab_data/frappe/score_frappe_batch_size_32_local_finish_all_models.json...
    Loading ../exp_data/tab_data/expressflow_score_mlp_sp_frappe_batch_size_32_cpu.json...
    Loading ../exp_data/tab_data/weight_share_nas_frappe.json...
    --------------------------------------------------
    Top (0.002, 320), nas_wot_add_syn_flow, 0.09176383558433188
    Top (0.002, 320), grad_norm, -0.05221181301609281
    Top (0.002, 320), grad_plain, 0.010659033743417297
    Top (0.002, 320), nas_wot, 0.17182172240830892
    Top (0.002, 320), ntk_cond_num, 0.047184176603287144
    Top (0.002, 320), ntk_trace, -0.19255177854077166
    Top (0.002, 320), ntk_trace_approx, -0.03251309912000771
    Top (0.002, 320), fisher, -0.11514936834521941
    Top (0.002, 320), grasp, 0.08584539745341016
    Top (0.002, 320), snip, 0.11704523741755678
    Top (0.002, 320), synflow, 0.09176383558433188
    Top (0.002, 320), weight_norm, 0.07500883959917261
    Top (0.002, 320), express_flow, 0.17456750065918614
    Top (0.002, 4), weight_share, -0.7999999999999999
    --------------------------------------------------
    Top (0.004, 640), nas_wot_add_syn_flow, 0.11882548541378278
    Top (0.004, 640), grad_norm, 0.025383640373591164
    Top (0.004, 640), grad_plain, -0.005632709232963978
    Top (0.004, 640), nas_wot, 0.1366327900980159
    Top (0.004, 640), ntk_cond_num, -0.15135665635044113
    Top (0.004, 640), ntk_trace, -0.1648806092409808
    Top (0.004, 640), ntk_trace_approx, -0.11244283819712042
    Top (0.004, 640), fisher, -0.13527564278345275
    Top (0.004, 640), grasp, 0.08921729237998346
    Top (0.004, 640), snip, 0.10973746428064046
    Top (0.004, 640), synflow, 0.11882763690829326
    Top (0.004, 640), weight_norm, 0.1397656899871464
    Top (0.004, 640), express_flow, 0.1770325672181817
    Top (0.004, 8), weight_share, -0.523809523809524
    --------------------------------------------------
    Top (0.006, 960), nas_wot_add_syn_flow, 0.14857992738707398
    Top (0.006, 960), grad_norm, 0.06158907308879807
    Top (0.006, 960), grad_plain, -0.019561768676986176
    Top (0.006, 960), nas_wot, 0.11271946601821807
    Top (0.006, 960), ntk_cond_num, -0.14027904408911906
    Top (0.006, 960), ntk_trace, -0.17359733075154035
    Top (0.006, 960), ntk_trace_approx, -0.13465470597426205
    Top (0.006, 960), fisher, -0.10129184676317263
    Top (0.006, 960), grasp, 0.0773428333918988
    Top (0.006, 960), snip, 0.08243519772399532
    Top (0.006, 960), synflow, 0.14858087682386803
    Top (0.006, 960), weight_norm, 0.15950466273961506
    Top (0.006, 960), express_flow, 0.18887676473173254
    Top (0.006, 12), weight_share, -0.6083916083916086
    --------------------------------------------------
    Top (0.008, 1280), nas_wot_add_syn_flow, 0.17996247022245493
    Top (0.008, 1280), grad_norm, 0.06243788518998898
    Top (0.008, 1280), grad_plain, -0.011124894709764077
    Top (0.008, 1280), nas_wot, 0.12083829746089553
    Top (0.008, 1280), ntk_cond_num, -0.1371495342350461
    Top (0.008, 1280), ntk_trace, -0.18678413362379745
    Top (0.008, 1280), ntk_trace_approx, -0.11327261499399259
    Top (0.008, 1280), fisher, -0.1335225388835811
    Top (0.008, 1280), grasp, 0.08311089873786662
    Top (0.008, 1280), snip, 0.11391344623983912
    Top (0.008, 1280), synflow, 0.1799591113794625
    Top (0.008, 1280), weight_norm, 0.17665909042477856
    Top (0.008, 1280), express_flow, 0.19513000381775134
    Top (0.008, 16), weight_share, -0.18529411764705883
    --------------------------------------------------
    Top (1, 160000), nas_wot_add_syn_flow, 0.7722624738302497
    Top (1, 160000), grad_norm, 0.5022881504556264
    Top (1, 160000), grad_plain, 0.04248195958085427
    Top (1, 160000), nas_wot, 0.6378172532674963
    Top (1, 160000), ntk_cond_num, -0.7226973187313154
    Top (1, 160000), ntk_trace, 0.5339715485407944
    Top (1, 160000), ntk_trace_approx, 0.15199042944834584
    Top (1, 160000), fisher, 0.503764400646809
    Top (1, 160000), grasp, -0.32104009686859214
    Top (1, 160000), snip, 0.7038939006969223
    Top (1, 160000), synflow, 0.7721962598662835
    Top (1, 160000), weight_norm, 0.7473390604383027
    Top (1, 160000), express_flow, 0.8028456677612497
    Top (1, 2000), weight_share, -0.03136309834077458
    JACFLOW 0.7406547545980877
    
    ================================================================
     uci_diabetes + mlp_sp
    ================================================================
    Loading ../exp_data/tab_data/uci_diabetes/all_train_baseline_uci_160k_40epoch.json...
    Loading ../exp_data/tab_data/uci_diabetes/score_uci_diabetes_batch_size_32_all_metrics.json...
    Loading ../exp_data/tab_data/expressflow_score_mlp_sp_uci_diabetes_batch_size_32_cpu.json...
    Loading ./not_exist...
    ./not_exist is not exist
    --------------------------------------------------
    Top (1, 160000), nas_wot_add_syn_flow, 0.6855057216575722
    Top (1, 160000), grad_norm, 0.3999081089630585
    Top (1, 160000), grad_plain, 0.02451448778377102
    Top (1, 160000), nas_wot, 0.635540008950723
    Top (1, 160000), ntk_cond_num, -0.5654103067100021
    Top (1, 160000), ntk_trace, 0.3774899968561059
    Top (1, 160000), ntk_trace_approx, 0.31808993358325754
    Top (1, 160000), fisher, 0.21598774748021798
    Top (1, 160000), grasp, -0.23202305383871977
    Top (1, 160000), snip, 0.629837846386711
    Top (1, 160000), synflow, 0.6855051126181101
    Top (1, 160000), weight_norm, 0.6927936726919207
    Top (1, 160000), express_flow, 0.6978445139608305
    JACFLOW 0.692050239116883
    
    ================================================================
     criteo + mlp_sp
    ================================================================
    Loading ../exp_data/tab_data/criteo/all_train_baseline_criteo.json...
    Loading ../exp_data/tab_data/criteo/score_criteo_batch_size_32.json...
    Loading ../exp_data/tab_data/expressflow_score_mlp_sp_criteo_batch_size_32_cpu.json...
    Loading ./not_exist...
    ./not_exist is not exist
    --------------------------------------------------
    Top (1, 10000), nas_wot_add_syn_flow, 0.7464429461404294
    Top (1, 10000), grad_norm, 0.3471953123725521
    Top (1, 10000), grad_plain, 0.023434944830985988
    Top (1, 10000), nas_wot, 0.7128521207543811
    Top (1, 10000), ntk_cond_num, -0.6335174238677821
    Top (1, 10000), ntk_trace, 0.49024945003576803
    Top (1, 10000), ntk_trace_approx, 0.00890247410055012
    Top (1, 10000), fisher, 0.4302424148655023
    Top (1, 10000), grasp, -0.2026179640580912
    Top (1, 10000), snip, 0.7978576087791914
    Top (1, 10000), synflow, 0.7464395938803958
    Top (1, 10000), weight_norm, 0.8134301266060824
    Top (1, 10000), express_flow, 0.8276736303927363
    JACFLOW 0.7602146996144342
    
    ================================================================
     cifar10 + nasbench101
    ================================================================
    Loading ../exp_data/img_data/score_101_15k_c10_128.json...
    Loading pickel ../exp_data/img_data/ground_truth/nasbench1_accuracy.p...
    Loading ../exp_data/img_data/ground_truth/nb101_id_to_hash.json...
    Loading ../exp_data/img_data/ground_truth/101_allEpoch_info_json...
    --------------------------------------------------
    Top (1, 15625), nas_wot_add_syn_flow, 0.3689520399131206
    Top (1, 15625), grad_norm, -0.24028712904763305
    Top (1, 15625), grad_plain, -0.37437473256641196
    Top (1, 15625), jacob_conv, -0.004427148034070742
    Top (1, 15625), nas_wot, 0.36881036232313413
    Top (1, 15625), ntk_cond_num, -0.30221532514959765
    Top (1, 15625), ntk_trace, -0.30751131315805114
    Top (1, 15625), ntk_trace_approx, -0.4195178764767932
    Top (1, 15625), fisher, -0.27330982397300113
    Top (1, 15625), grasp, 0.27891820857555477
    Top (1, 15625), snip, -0.1668017404578911
    Top (1, 15625), synflow, 0.3689520399131206
    Top (1, 15625), weight_norm, 0.5332072603621669
    JACFLOW 0.412358290561836
    
    ================================================================
     cifar10 + nasbench201
    ================================================================
    Loading ../exp_data/img_data/score_201_15k_c10_bs32_ic16.json...
    Loading ../exp_data/img_data/ground_truth/201_allEpoch_info...
    --------------------------------------------------
    Top (1, 15625), nas_wot_add_syn_flow, 0.778881359475867
    Top (1, 15625), grad_norm, 0.6407726431247389
    Top (1, 15625), grad_plain, -0.12987923450265587
    Top (1, 15625), nas_wot, 0.7932888939510635
    Top (1, 15625), ntk_cond_num, -0.48387478988448707
    Top (1, 15625), ntk_trace, 0.37783839330319213
    Top (1, 15625), ntk_trace_approx, 0.346026013974282
    Top (1, 15625), fisher, 0.3880671330868219
    Top (1, 15625), grasp, 0.5301491874432275
    Top (1, 15625), snip, 0.6437364743868734
    Top (1, 15625), weight_norm, 0.005565168122791671
    Top (1, 15625), synflow, 0.7788685936507451
    JACFLOW 0.8339798847659702
    
    ================================================================
     cifar100 + nasbench201
    ================================================================
    Loading ../exp_data/img_data/score_201_15k_c100_bs32_ic16.json...
    --------------------------------------------------
    Top (1, 15000), nas_wot_add_syn_flow, 0.7671956422900841
    Top (1, 15000), grad_norm, 0.638398024328767
    Top (1, 15000), grad_plain, -0.16701447428313634
    Top (1, 15000), nas_wot, 0.8089325143676851
    Top (1, 15000), ntk_cond_num, -0.39182378815354696
    Top (1, 15000), ntk_trace, 0.37724922855703374
    Top (1, 15000), ntk_trace_approx, 0.38385292527377407
    Top (1, 15000), fisher, 0.3845332624562634
    Top (1, 15000), grasp, 0.5462288460061152
    Top (1, 15000), snip, 0.6375851983100865
    Top (1, 15000), weight_norm, 0.011918450096024165
    Top (1, 15000), synflow, 0.7671950896881894
    JACFLOW 0.836747529703176
    
    ================================================================
     ImageNet16-120 + nasbench201
    ================================================================
    Loading ../exp_data/img_data/score_201_15k_imgNet_bs32_ic16.json...
    --------------------------------------------------
    Top (1, 15000), nas_wot_add_syn_flow, 0.7450159967363084
    Top (1, 15000), grad_norm, 0.566172650250217
    Top (1, 15000), grad_plain, -0.16454617540967373
    Top (1, 15000), nas_wot, 0.7769715502067605
    Top (1, 15000), ntk_cond_num, -0.41263954976382056
    Top (1, 15000), ntk_trace, 0.310570269337782
    Top (1, 15000), ntk_trace_approx, 0.3566322129734418
    Top (1, 15000), fisher, 0.3202230462329743
    Top (1, 15000), grasp, 0.5093070840387243
    Top (1, 15000), snip, 0.5688946225005688
    Top (1, 15000), weight_norm, 0.005571648346911519
    Top (1, 15000), synflow, 0.7450170886565295
    JACFLOW 0.8077842182522329

    """