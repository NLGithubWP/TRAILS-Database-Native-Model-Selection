

CREATE OR REPLACE
PROCEDURE model_selection_sp(dataset TEXT, selected_columns TEXT[], budget TEXT)
LANGUAGE plpgsql
AS $$
DECLARE
    -- global_result place holder
    result_status TEXT;
    -- user defined params
    batch_size CONSTANT integer := 8;
    column_list TEXT;
    -- function outputs
    score_time TEXT := 0.12;
    train_time TEXT := 0.56;
    coordinator_k integer;
    coordinator_u integer;
    coordinator_n integer;
BEGIN

    column_list := array_to_string(selected_columns, ', ');
    EXECUTE format('
            WITH batch_rows AS (
                SELECT %s
                FROM %I
                ORDER BY RANDOM()
                LIMIT %s OFFSET 0
            )
            SELECT profiling_filtering_phase(
                json_agg(row_to_json(t))::text
            )
            FROM batch_rows AS t', column_list, dataset, batch_size) INTO result_status;
    score_time := json_extract_path_text(result_status::json, 'time');
    RAISE NOTICE '1. profiling_filtering_phase, get score_time: %', score_time;

    EXECUTE format('
            WITH batch_rows AS (
                SELECT %s
                FROM %I
                ORDER BY RANDOM()
                LIMIT %s OFFSET 0
            )
            SELECT profiling_refinement_phase(
                json_agg(row_to_json(t))::text
            )
            FROM batch_rows AS t', column_list, dataset, batch_size) INTO result_status;
    train_time := json_extract_path_text(result_status::json, 'time');
    RAISE NOTICE '2. profiling_refinement_phase, get train_time: %', train_time;

    EXECUTE format('SELECT "coordinator"(%L, %L, %L, true)', score_time, train_time, budget) INTO result_status;

    coordinator_k := (json_extract_path_text(result_status::json, 'k'))::integer;
    coordinator_u := (json_extract_path_text(result_status::json, 'u'))::integer;
    coordinator_n := (json_extract_path_text(result_status::json, 'n'))::integer;
    RAISE NOTICE '3. coordinator result: k = %, u = %, n = %', coordinator_k, coordinator_u, coordinator_n;

    EXECUTE format('
        WITH batch_rows AS (
            SELECT %s
            FROM %I
            ORDER BY RANDOM()
            LIMIT %s OFFSET 0
        )
        SELECT filtering_phase(
            json_agg(row_to_json(t))::text, %s, %s
        )
        FROM batch_rows AS t', column_list, dataset, batch_size, coordinator_n, coordinator_k) INTO result_status;
    RAISE NOTICE '4. run filtering phase, k models = %', result_status;

END; $$;
