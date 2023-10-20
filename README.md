# Database-Native Model Selection 

​																																																		-- based on Singa



![image-20231020174425377](documents/image-20231020174425377.png)

## Build Docker Image

```bash
git clone https://github.com/NLGithubWP/TRAILS-Database-Native-Model-Selection.git
cd TRAILS-Database-Native-Model-Selection
docker build -t trails-singa .
```

## Run Docker Image

```bash
docker run -d --name trails-singa \
  --network="host" \
  -v /hdd1/xingnaili/exp_data/:/project/exp_data \
  trails
```

## Start PostgreSQL Instance

```bash
# 1. Run docker container
docker exec -it trails-singa bash 
# 2. Clone the code
cd ~
git clone https://github.com/NLGithubWP/TRAILS-Database-Native-Model-Selection.git
# 3. Load data into RDBMS
bash /project/TRAILS-Database-Native-Model-Selection/internal/ml/model_selection/scripts/database/load_data_to_db.sh /project/exp_data/data/structure_data/frappe frappe
# 4. Run database server
cd TRAILS-Database-Native-Model-Selection/internal/pg_extension
cargo pgrx run

```


## Register Stored Procedure

```sql
CREATE OR REPLACE
PROCEDURE model_selection_sp(
    dataset TEXT,               --dataset name
    selected_columns TEXT[],    --used columns
    N INTEGER,                  --number of models to evaluate
    batch_size INTEGER,         --batch size, for profiling, filtering
    config_file TEXT            --config file path
)
LANGUAGE plpgsql
AS $$
DECLARE
    -- global inputs/outputs
    result_status TEXT;
    column_list TEXT;
BEGIN
    -- combine the columns into a string
    column_list := array_to_string(selected_columns, ', ');

    -- 4. Run filtering phase to get top K models.
    EXECUTE format('
                WITH batch_rows AS (
                    SELECT %s
                    FROM %I
                    ORDER BY RANDOM()
                    LIMIT %s OFFSET 0
                )
                SELECT filtering_phase(
                    json_agg(row_to_json(t))::text, %s, %s, %L
                )
                FROM batch_rows AS t', column_list, dataset, batch_size, N, 1, config_file) INTO result_status;
    RAISE NOTICE '4. run filtering phase, k models = %', result_status;

END; $$;
```

# Compile the UDF

```bash
# Try compile the UDF
DROP EXTENSION IF EXISTS pg_extension;
CREATE EXTENSION pg_extension;
# If the above faile, open another terminal and repeat the step 1 and 4 of "Start PostgreSQL Instance"
# Then run hose 
rm /home/postgres/.pgrx/14.9/pgrx-install/share/extension/pg_extension--0.1.0.sql
vi /home/postgres/.pgrx/14.9/pgrx-install/share/extension/pg_extension--0.1.0.sql
# copy the following to the /home/postgres/.pgrx/14.9/pgrx-install/share/extension/pg_extension--0.1.0.sql
-- src/lib.rs:66
-- pg_extension::filtering_phase
CREATE  FUNCTION "filtering_phase"(
    "mini_batch" TEXT, /* alloc::string::String */
    "n" INT, /* i32 */
    "k" INT, /* i32 */
    "config_file" TEXT /* alloc::string::String */
) RETURNS TEXT /* alloc::string::String */
    IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'filtering_phase_wrapper';
```

## Run Model Selection 

```sql
-- Template for calling 'model_selection_sp' stored procedure
CALL model_selection_sp(
    <TABLE_NAME>,             -- The name of the table or dataset from which data should be retrieved.
    <COLUMN_NAMES_ARRAY>,     -- An array of column names to be considered in the model selection process.
    <PARAMETER_1>,            -- Number of models to explore
    <PARAMETER_2>,            -- Batch size
    <CONFIG_FILE_PATH>        -- The file path to a configuration file needed for the process.
);


# For example
CALL model_selection_sp(
  	'frappe_train',
  	ARRAY['col1', 'col2', 'col3', 'label'], 
    10, 
    32, 
  '/home/postgres/TRAILS-Database-Native-Model-Selection/internal/ml/model_selection/config.ini');
```

# Example Result

![image-20231020174945226](documents/image-20231020174945226.png)
