

CREATE EXTENSION IF NOT EXISTS plpython3u;

# create table
CREATE TABLE data_table (
    id serial PRIMARY KEY,
    F1 double precision,
    F2 double precision,
    F3 double precision,
    F4 double precision
);

# insert dummy data
INSERT INTO data_table (F1, F2, F3, F4)
SELECT
    random()::double precision,
    random()::double precision,
    random()::double precision,
    random()::double precision
FROM generate_series(1, 2000);

# create a customer type
CREATE TYPE data_table_fields AS (
    F1 double precision,
    F2 double precision,
    F3 double precision,
    F4 double precision
);

CREATE OR REPLACE FUNCTION rows_to_json_string(rows data_table_fields[]) RETURNS text
LANGUAGE plpython3u
AS $$
    import json
    json_rows = []
    for row in rows:
        json_rows.append([row.f1, row.f2, row.f3, row.f4])

    return json.dumps(json_rows)
$$;




CREATE OR REPLACE PROCEDURE model_selection(rows data_table_fields[])
LANGUAGE plpython3u
AS $$
import plpy
import subprocess

plpy.notice("1. Starting stored procedure...")
plpy.notice(f"2. Reading dataset {rows}")
plpy.notice(f"3. Calling sub-process to run")

script_path = './/src/database/run.sh'
result = subprocess.run(['bash', script_path], capture_output=True, text=True)
if result.returncode == 0:
    plpy.notice(result.stdout)
else:
    plpy.error(result.stderr)

plpy.notice("Finished stored procedure.")
return 0.123
$$;



CREATE OR REPLACE PROCEDURE process_rows_in_batches()
LANGUAGE plpgsql
AS $$
DECLARE
    batch_size CONSTANT integer := 1024;
    total_rows integer;
    num_batches integer;
    json_result text;
BEGIN
    SELECT COUNT(*) INTO total_rows FROM data_table;
    num_batches := CEIL(total_rows::double precision / batch_size)::integer;

    FOR i IN 1..num_batches
    LOOP
        RAISE NOTICE 'Processing batch % of %', i, num_batches;

        -- Retrieve 1024 rows (or the remaining rows for the last batch) using LIMIT and OFFSET
        WITH batch_rows AS (
            SELECT F1, F2, F3, F4
            FROM data_table
            ORDER BY id
            LIMIT batch_size OFFSET (i - 1) * batch_size
        )
        SELECT rows_to_json_string(ARRAY(SELECT ROW(F1, F2, F3, F4)::data_table_fields FROM batch_rows)) INTO json_result;

        RAISE NOTICE 'Batch % JSON result: %', i, json_result;
    END LOOP;
END;
$$;




CREATE OR REPLACE PROCEDURE compute_forward_pytorch()
LANGUAGE plpgsql
AS $$
DECLARE
    row_data data_table%ROWTYPE;
    result double precision;
BEGIN
    FOR row_data IN SELECT * FROM data_table
    LOOP
        result := forward_computation_pytorch(row_data.input1, row_data.input2);
        RAISE NOTICE 'Output for id %: %', row_data.id, result;
    END LOOP;
END;
$$;


CALL process_rows_in_batches();













