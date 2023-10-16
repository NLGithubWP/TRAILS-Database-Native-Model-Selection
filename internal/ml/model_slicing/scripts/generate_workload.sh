

python generate_workload.py \
    --dataset cvd --nfield 11\
    --max_select_col 4 \
    --n 100 --output_name random_100


python generate_workload.py \
    --dataset adult --nfield 13\
    --max_select_col 4 \
    --n 100 --output_name random_100


python generate_workload.py \
    --dataset bank --nfield 16\
    --max_select_col 4 \
    --n 100 --output_name random_100

python generate_workload.py \
    --dataset frappe --nfield 10\
    --max_select_col 4 \
    --n 100 --output_name random_100