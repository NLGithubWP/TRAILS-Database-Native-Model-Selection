
# measure time usaeg

python exps/main_v2/system/measure_data_loading_time.py --base_dir="../exp_data/data/structure_data/" --dataset=frappe --nfield=10 --batch_size=1024 --workers=0 --max_load=1024

    #time fulling loading:
    #5.960570335388184

    #time loading 1 iteration
    0.07835125923156738

python exps/main_v2/system/measure_data_loading_time.py --base_dir="../exp_data/data/structure_data/" --dataset=criteo --nfield=39 --batch_size=1024 --workers=0 --max_load=1024

    #time fulling loading:
    #1814.5491938591003

    #time loading 1 iteration
    12.259164810180664

python exps/main_v2/system/measure_data_loading_time.py --base_dir="../exp_data/data/structure_data/" --dataset=uci_diabetes --nfield=43 --batch_size=1024 --workers=0 --max_load=1024

    #time fulling loading:
    # 4.2008748054504395

    #time loading 1 iteration
    0.11569786071777344

