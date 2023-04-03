

CREATE EXTENSION IF NOT EXISTS plpython3u;

CREATE TABLE criteo_dataset (
    id serial PRIMARY KEY,
    F1 double precision,
    F2 double precision,
    F3 double precision,
    F4 double precision
);

INSERT INTO criteo_dataset (F1, F2, F3, F4)
SELECT
    random()::double precision,
    random()::double precision,
    random()::double precision,
    random()::double precision
FROM generate_series(1, 2000);


CREATE OR REPLACE FUNCTION profile_t1(batch_data TEXT) RETURNS text
LANGUAGE plpython3u
AS $$
    import subprocess
    script_path = './/src/database/run_filter_phase_with_batch_data.sh'
    subprocess.run(['bash', script_path, batch_data])
    return "OK"
$$;


CREATE OR REPLACE FUNCTION profile_t2(batch_data TEXT) RETURNS text
LANGUAGE plpython3u
AS $$
    from urllib import request, parse
    url = "http://0.0.0.0:8002/profile"
    data = parse.urlencode({"data": batch_data}).encode()
    req = request.Request(url, data=data)
    resp = request.urlopen(req)
    return "OK"
$$;

CREATE OR REPLACE FUNCTION coordination(budgets float) RETURNS text
LANGUAGE plpython3u
AS $$
    import subprocess
    script_path = './/src/database/run_filter_phase_with_batch_data.sh'
    subprocess.run(['bash', script_path, str(budgets)])
    return "OK"
$$;


CREATE OR REPLACE FUNCTION filtering(batch_data TEXT) RETURNS text
LANGUAGE plpython3u
AS $$
    import subprocess
    script_path = './/src/database/run_filter_phase_with_batch_data.sh'
    subprocess.run(['bash', script_path, batch_data])
    return "OK"
$$;


CREATE OR REPLACE FUNCTION refinement() RETURNS text
LANGUAGE plpython3u
AS $$
    from urllib import request, parse
    url = "http://0.0.0.0:8002/start_refinement"
    req = request.Request(url)
    resp = request.urlopen(req)
$$;


CREATE OR REPLACE
PROCEDURE model_selection(dataset_name TEXT, selected_columns TEXT[], budget float)
LANGUAGE plpgsql
AS $$
DECLARE
    row_count INTEGER;
    num_batches INTEGER;
    i INTEGER;
    result_status TEXT;
    batch_size CONSTANT integer := 10;
    column_list TEXT;
BEGIN
    column_list := array_to_string(selected_columns, ', ');

    EXECUTE format('SELECT COUNT(*) FROM %I', dataset_name) INTO row_count;
    num_batches := CEIL(row_count::float / batch_size)::integer;

    -- 1. profile t1
    RAISE NOTICE '1. profile t1';
    EXECUTE format('
        WITH batch_rows AS (
            SELECT %s
            FROM %I
            ORDER BY id
            LIMIT 3 OFFSET 0
        )
        SELECT profile_t1(
            json_agg(row_to_json(t))::text
        )
        FROM batch_rows AS t', column_list, dataset_name) INTO result_status;

    -- 2. profile t2
    RAISE NOTICE '2. profile t2';
    FOR i IN 1..3
    LOOP
        RAISE NOTICE 'running profile t2 phase with batch id % of %', i, num_batches;
        EXECUTE format('
        WITH batch_rows AS (
            SELECT %s
            FROM %I
            ORDER BY id
            LIMIT %s OFFSET (%s - 1) * %s
        )
        SELECT profile_t2(
            json_agg(row_to_json(t))::text
        )
        FROM batch_rows AS t', column_list,
            dataset_name,
            batch_size::text,
            i::text,
            batch_size::text) INTO result_status;
    END LOOP;

    -- 3. coordinating
    RAISE NOTICE '3. coordinating';
    SELECT coordination(budget) INTO result_status;

    -- 4. run filter phase with user defined batch size.
    RAISE NOTICE '4. run filter phase with user defined batch size.';
    EXECUTE format('
        WITH batch_rows AS (
            SELECT %s
            FROM %I
            ORDER BY id
            LIMIT 1 OFFSET 0
        )
        SELECT filtering(
            json_agg(row_to_json(t))::text
        )
        FROM batch_rows AS t', column_list, dataset_name) INTO result_status;

    -- 5. run refinement phase with user defined batch size.
    RAISE NOTICE '5. run refinement phase with user defined batch size.';
    SELECT refinement() INTO result_status;

END; $$;



CALL model_selection(
    'criteo_dataset',
    ARRAY['F1', 'F2'],
    10
);






