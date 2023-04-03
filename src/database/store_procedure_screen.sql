

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


        -- 1. In-Database profiling
        CREATE OR REPLACE
        FUNCTION profile_t1(batch_data TEXT)
        RETURNS text
        LANGUAGE plpython3u
        AS $$
            import subprocess
            subprocess.run(['bash',
                'profile_t1.sh', batch_data])
            return "OK" $$;

        CREATE OR REPLACE
        FUNCTION profile_t2(batch_data TEXT)
        RETURNS text
        LANGUAGE plpython3u
        AS $$
          from urllib import request, parse
          url = "REFINEMENTURL/profile"
          data = parse.urlencode({
            "data": batch_data}).encode()
          req = request.Request(url, data=data)
          resp = request.urlopen(req)
          return "OK" $$;

        -- 2. In-Database Coordination
        CREATE OR REPLACE
        FUNCTION coordination(T float)
        RETURNS text
        LANGUAGE plpython3u
        AS $$
            import subprocess
            subprocess.run(['bash',
             'coordinating.sh',
             str(T)])
            return "OK"$$;

        -- 3. In-Database Filtering
        CREATE OR REPLACE
        FUNCTION filtering(batch_data TEXT)
        RETURNS text
        LANGUAGE plpython3u
        AS $$
          import subprocess
          subprocess.run(['bash',
            'filtering.sh', batch_data])
          return "OK" $$;
        -- 4. In-Database Refinement
        CREATE OR REPLACE
        FUNCTION refinement()
        RETURNS text
        LANGUAGE plpython3u
        AS $$
          from urllib import request, parse
          url = "REFINEMENTURL/start_refinement"
          req = request.Request(url)
          resp = request.urlopen(req)
          return "OK" $$;

        CALL model_selection(
            'CriteoDataset',   -- dataset
             ARRAY['F1', 'F2'],-- used columns
             2);               -- target SLO


        -- 5. In-Database Model Selection
        CREATE OR REPLACE
        PROCEDURE model_selection(
                    dataset_name TEXT,
                    selected_columns TEXT[],
                    budget float)
        LANGUAGE plpgsql
        AS $$
        DECLARE
          row_count INTEGER;
          num_batches INTEGER;
          i INTEGER;
          result_status TEXT;
          batch_size CONSTANT integer := 1024;
          column_list TEXT;
        BEGIN
          column_list := array_to_string(
                           selected_columns, ', ');
          EXECUTE format('SELECT COUNT(*) FROM %I',
            dataset_name) INTO row_count;
          num_batches := CEIL(
            row_count::float/batch_size)::integer;
          -- 5.1. Profile time t1
          EXECUTE format('
            WITH batch_rows AS (SELECT %s FROM %I ORDER BY id LIMIT 3 OFFSET 0)
                SELECT profile_t1(json_agg(row_to_json(t))::text)
                FROM batch_rows AS t', column_list, dataset_name) INTO result_status;
          -- 5.2. Profile time t2
          FOR i IN 1..num_batches
          LOOP
            EXECUTE format('
            WITH batch_rows AS (SELECT %s FROM %I ORDER BY id LIMIT %s OFFSET (%s - 1) * %s)
            SELECT profile_t2(json_agg(row_to_json(t))::text)
            FROM batch_rows AS t', column_list, dataset_name, batch_size::text, i::text,
                batch_size::text) INTO result_status;
          END LOOP;
          -- 5.3. Coordination
          SELECT coordination(budget) INTO result_status;
          -- 5.4. Run filtering phase with mini-batch of data.
          EXECUTE format('
            WITH batch_rows AS ( SELECT %s FROM %I ORDER BY id LIMIT 32 OFFSET 0)
            SELECT filtering(json_agg(row_to_json(t))::text)
            FROM batch_rows AS t', column_list, dataset_name) INTO result_status;
          -- 5.5. Run refinement phase.
          SELECT refinement() INTO result_status;
          END; $$;


