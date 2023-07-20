

# TRAILS: A Database Native Model Selection System

![image-20230702035806963](documents/imgs/image-20230702035806963.png)

# Config Environments

```bash
# Create virtual env
conda config --set ssl_verify false
conda create -n "trails" python=3.8.10
conda activate trails
pip install -r requirement.txt

cd TRAILS

# make a dir to store all results.
mkdir ../exp_data
```

# Reproduce the results

## NAS-Bench-Tabular

 NAS-Bench-Tabular can be either **download** or build from scratch.

### Download NAS-Bench-Tabular

1. **Download** the dataset using the following link, and extract them to `result_base`

```bash
https://drive.google.com/file/d/1fpKAqvvVooiJh2EIfz18UMsBE4demHL2/view?usp=sharing
```

### Build NAS-Bench-Tabular

2. Build the **NAS-Bench-Tabular** from scratch

```python
# Construct NAS-Bench-Tabular:
## 1. Training all models.
bash internal/ml/model_selection/scripts/nas-bench-tabular/score_all_modesl_frappe.sh
bash internal/ml/model_selection/scripts/nas-bench-tabular/score_all_modesl_uci.sh
bash internal/ml/model_selection/scripts/nas-bench-tabular/score_all_modesl_criteo.sh

## 2. Scoring all models using all TFMEMs.
bash internal/ml/model_selection/scripts/nas-bench-tabular/score_all_modesl_frappe.sh
bash internal/ml/model_selection/scripts/nas-bench-tabular/score_all_modesl_uci.sh
bash internal/ml/model_selection/scripts/nas-bench-tabular/score_all_modesl_criteo.sh
```

3. Build the **NAS-Bench-Img** from scratch

   To facilitate the experiments and query speed (NASBENCH API is slow)

   1. We retrieve all results from NASBENCH API and store them as a json file.
   2. We score all models in NB201 and 28K models in NB101.
   3. We search with  EA + Score and record the searching process in terms of
       `run_id,  current_explored_model, top_400 highest scored model, time_usage`
    to SQLLite.

```python
# 1. Record NASBENCH API data into json file
## This requires to install nats_bench: pip install nats_bench
bash internal/ml/model_selection/scripts/nas_bench_img/convert_api_2_json.sh

# 2. Scoring all models using all TFMEMs.
bash internal/ml/model_selection/scripts/nas_bench_img/score_all_models.sh

# 3. Explore with EA ans score result and store exploring process into SQLLite
bash internal/ml/model_selection/scripts/nas_bench_img/explore_all_models.sh
```

The following experiment could then query filtering phase results based on `run_id`.

## SLO-Aware 2Phase-MS

With the above **NAS-Bench-Tabular**, we could run various experiments.

```bash
# 1. Generate the results for drawing the figure
## tabular data: training-base-ms
bash internal/ml/model_selection/scripts/baseline_system_tab.sh
## tabular data: training-free-ms, 2phase-ms
bash internal/ml/model_selection/scripts/anytime_tab.sh
## image data: training-base-ms, training-free-ms, 2phase-ms
bash internal/ml/model_selection/scripts/anytime_img_w_baseline.sh

# 2. Draw figure
python internal/ml/model_selection/exps/macro/anytime_tab_draw.py
python internal/ml/model_selection/exps/macro/anytime_img_draw.py
```

![image-20230702035554579](documents/imgs/image-20230702035554579.png)

## Benchmark TFMEMs

```bash
export PYTHONPATH=$PYTHONPATH:./internal/ml/model_selection
python ./internal/ml/model_selection/exps/micro/measure_correlation.py
```

![image-20230421214835152](./documents/imgs/image-20230421214835152.png)

## System Motivation Experiments

asdf

## Reproduce Figure7

```bash
python exps/main_v2/analysis/2.\ cost_draw.py
python exps/main_v2/analysis/3.\ cost_train_based.py
```

![image-20230702035622198](documents/imgs/image-20230702035622198.png)

## Reproduce Figure8

```bash
# draw figure 8(a)
python exps/main_v2/analysis/5.draw_IDMS_var_workloads.py
# draw figure 8(b)
python exps/main_v2/analysis/6.draw_IDMS_dataloading.py
```

![image-20230702035639502](documents/imgs/image-20230702035639502.png)

## Reproduce Figure9

```bash
# generate results
python main/4_system/analysis/2_benchmarking/1_micro_phase2.py
# draw with the following cmd
python main/4_system/analysis/2_benchmarking/1_micro_phase2_only_draw.py
```

![image-20230421214753155](./documents/imgs/image-20230421214753155.png)

## Reproduce Figure 10

```bash
# draw results with the following:
python main/4_system/analysis/1_sys_design/plot_1.py
python main/4_system/analysis/1_sys_design/plot_2.py
```

![image-20230421214807878](./documents/imgs/image-20230421214807878.png)


# Run end2end model selection

download the dataset and put it in the `exp_data/data/structure_data`

```
python main.py --budget=100 --dataset=frappe
```

Check the log at the `Logs`

![image-20230421220338391](./documents/imgs/image-20230421220338391.png)

![image-20230421220443231](./documents/imgs/image-20230421220443231.png)

