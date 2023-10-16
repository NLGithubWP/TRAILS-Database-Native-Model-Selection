# Collecting data for plotting
datasets_result_640k = \
    {'data_query_time': 5.470755338668823,
     'py_conver_to_tensor': 19.043654918670654,
     'tensor_to_gpu': 0.00024271011352539062,
     'py_compute': 12.584440469741821,
     'load_model': 0.1481478214263916,
     'overall_query_latency': 38.474358320236206}

datasets_result_10k = {'data_query_time': 0.07095789909362793, 'py_conver_to_tensor': 0.2404768466949463, 'tensor_to_gpu': 2.1219253540039062e-05, 'py_compute': 0.06025338172912598, 'load_model': 0.12145113945007324, 'overall_query_latency': 0.5489771366119385}
datasets_result_20k = {'data_query_time': 0.1730802059173584, 'py_conver_to_tensor': 0.44614291191101074, 'tensor_to_gpu': 0.000118255615234375, 'py_compute': 0.5517294406890869, 'load_model': 0.16239595413208008, 'overall_query_latency': 1.4929616451263428}
datasets_result_40k = {'data_query_time': 0.2943258285522461, 'py_conver_to_tensor': 1.0942583084106445, 'tensor_to_gpu': 0.00019288063049316406, 'py_compute': 0.8386809825897217, 'load_model': 0.17323684692382812, 'overall_query_latency': 2.5823915004730225}

for datasets_result_used in [datasets_result_10k, datasets_result_20k, datasets_result_40k]:
    print("===" * 10)
    total_usage = datasets_result_used["data_query_time"] + \
      datasets_result_used["py_conver_to_tensor"] + \
      datasets_result_used["tensor_to_gpu"] + \
      datasets_result_used["py_compute"] + \
      datasets_result_used["load_model"]

    print(total_usage - datasets_result_used["overall_query_latency"])

    print(f"load_model = "
          f"{100*datasets_result_used['load_model']/total_usage}")
    print(f"data_query_time = "
          f"{100*datasets_result_used['data_query_time']/total_usage}")
    print(f"py_conver_to_tensor = "
          f"{100*datasets_result_used['py_conver_to_tensor']/total_usage}")
    print(f"py_compute & compute = "
          f"{100*(datasets_result_used['py_compute'] + datasets_result_used['py_conver_to_tensor'])/total_usage}")




