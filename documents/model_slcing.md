# Envs

```bash
unset PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/project/TRAILS/internal/ml/
export PYTHONPATH=$PYTHONPATH:/project/TRAILS/internal/ml/model_slicing
echo $PYTHONPATH
```

# Save data 

4 datasets are used here.

```
adult  bank  cvd  frappe
```

Save the statistics

```bash
# save the data cardinalities, run in docker

# frappe
python3 ./internal/ml/model_slicing/save_satistics.py --dataset frappe --data_dir /hdd1/sams/data/ --nfeat 5500 --nfield 10 --max_filter_col 10 --train_dir ./

# adult
python3 ./internal/ml/model_slicing/save_satistics.py --dataset adult --data_dir /hdd1/sams/data/ --nfeat 140 --nfield 13 --max_filter_col 13 --train_dir ./

# cvd
python3 ./internal/ml/model_slicing/save_satistics.py --dataset cvd --data_dir /hdd1/sams/data/ --nfeat 110 --nfield 11 --max_filter_col 11 --train_dir ./

# bank
python3 ./internal/ml/model_slicing/save_satistics.py --dataset bank --data_dir /hdd1/sams/data/ --nfeat 80 --nfield 16 --max_filter_col 16 --train_dir ./
```

# Run docker

```bash
# in server
ssh panda17

# goes to /home/xingnaili/firmest_docker/TRAILS
git submodule update --recursive --remote

# run container
docker run -d --name trails \
  --network="host" \
  -v $(pwd)/TRAILS:/project/TRAILS \
  -v /hdd1/xingnaili/exp_data/:/project/exp_data \
  -v /hdd1/sams/tensor_log/:/project/tensor_log \
  -v /hdd1/sams/data/:/project/data_all \
  trails
    
# Enter the docker container.
docker exec -it trails bash 
```

# Run in database

Config the database runtime

```sql
cargo pgrx run
```

Load data into RDBMS

```bash

psql -h localhost -p 28814 -U postgres 

# frappe
bash /project/TRAILS/internal/ml/model_selection/scripts/database/load_data_to_db.sh /project/data_all/frappe frappe

# adult
bash ./internal/ml/model_selection/scripts/database/load_data_to_db.sh /project/data_all/adult adult

# cvd 
bash /project/TRAILS/internal/ml/model_selection/scripts/database/load_data_to_db.sh /project/data_all/cvd cvd

# bank
bash /project/TRAILS/internal/ml/model_selection/scripts/database/load_data_to_db.sh /project/data_all/bank bank

```

Verify data is in the DB

```sql
# check table status
\dt
\d frappe_train
SELECT * FROM frappe_train LIMIT 10;
```

Config

```sql
# after run the pgrx, then edie the sql
# generate schema
cargo pgrx schema >> /home/postgres/.pgrx/14.9/pgrx-install/share/extension/pg_extension--0.1.0.sql

-- src/lib.rs:173
-- pg_extension::sams_inference
CREATE  FUNCTION "sams_inference"(
        "dataset" TEXT, /* alloc::string::String */
        "condition" TEXT, /* alloc::string::String */
        "config_file" TEXT, /* alloc::string::String */
        "col_cardinalities_file" TEXT, /* alloc::string::String */
        "model_path" TEXT, /* alloc::string::String */
        "sql" TEXT, /* alloc::string::String */
        "batch_size" INT /* i32 */
) RETURNS TEXT /* alloc::string::String */
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'run_sams_inference_wrapper';

# record the necessary func above and then copy it to following
rm /home/postgres/.pgrx/14.9/pgrx-install/share/extension/pg_extension--0.1.0.sql
vi /home/postgres/.pgrx/14.9/pgrx-install/share/extension/pg_extension--0.1.0.sql

# then drop/create extension
DROP EXTENSION IF EXISTS pg_extension;
CREATE EXTENSION pg_extension;

pip install einops
```

Examples

```sql

# this is database name, columns used, time budget, batch size, and config file
SELECT count(*) FROM frappe_train WHERE col2='973:1' LIMIT 1000;
SELECT col2, count(*) FROM frappe_train group by col2 order by count(*) desc;

# query with two conditions
SELECT sams_inference(
    'frappe', 
    '{"1":266, "2":1244}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    'WHERE col1=''266:1'' and col2=''1244:1''', 
    32
);

# query with 1 conditions
SELECT sams_inference(
    'frappe', 
    '{"2":977}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    'WHERE col2=''977:1''', 
    8000
); 

# query with no conditions
SELECT sams_inference(
    'frappe', 
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    '', 
    8000
); 

# explaination
EXPLAIN (ANALYZE, BUFFERS) SELECT sams_inference(
    'frappe', 
    '{"2":977}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    'WHERE col2=''977:1''', 
    8000
); 


```

# Clear cache

```sql
DISCARD ALL;
```

# Benchmark Latency over all datasets

## Adult

```sql
SELECT sams_inference(
    'adult', 
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/adult_col_cardinalities', 
    '/project/tensor_log/adult/Ednn_K16_alpha2-5', 
    '', 
    10000
); 
```

## Frappe

```sql
SELECT sams_inference(
    'frappe', 
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    '', 
    10000
); 

SELECT sams_inference(
    'frappe', 
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    '', 
    20000
); 

SELECT sams_inference(
    'frappe', 
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    '', 
    40000
); 

SELECT sams_inference_shared(
    'frappe', 
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    '', 
    40000
); 




SELECT sams_inference(
    'frappe', 
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    '', 
    80000
); 


SELECT sams_inference(
    'frappe', 
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    '', 
    160000
); 
```

## CVD

```sql
SELECT sams_inference(
    'cvd', 
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/cvd_col_cardinalities', 
    '/project/tensor_log/cvd/dnn_K16_alpha2-5', 
    '', 
    10000
); 
```

## Bank

```sql
SELECT sams_inference(
    'bank', 
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/bank_col_cardinalities', 
    '/project/tensor_log/bank/dnn_K16_alpha2-3_beta1e-3', 
    '', 
    10000
); 
```





















