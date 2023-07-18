

# TRAILS: A Database Native Model Selection System

![image-20230702035806963](documents/imgs/image-20230702035806963.png)

# Config Environments

```bash
# Create virtual env
conda config --set ssl_verify false
conda create -n "trails" python=3.8.10
conda activate trails
pip install -r requirement.txt 

# Init env
source init_env

# make a dir to store all results. 
mkdir exp_data
```

# Reproduce the results

1. Download the dataset using the following link, and extract them to `result_base`

```bash
https://drive.google.com/file/d/1fpKAqvvVooiJh2EIfz18UMsBE4demHL2/view?usp=sharing
```

## Reproduce Figure6

Update the dataset_name and then run.

```bash
# generate pre-calculated results.

# generate the results for draw the figure
bash internal/ml/model_selection/scripts/anytime_tab.sh
bash internal/ml/model_selection/scripts/anytime_img.sh
# draw figure
python internal/ml/model_selection/exps/macro/anytime_tab_draw.py
python internal/ml/model_selection/exps/macro/anytime_img_draw.py
```

![image-20230702035554579](documents/imgs/image-20230702035554579.png)

## Reproduce Table 2

```bash
python exps/main_v2/analysis/4.\ measure_correlation.py
```

![image-20230421214835152](./documents/imgs/image-20230421214835152.png)

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
