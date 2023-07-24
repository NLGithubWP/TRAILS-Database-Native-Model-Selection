

CREATE OR REPLACE
PROCEDURE model_selection_sp(dataset TEXT, selected_columns TEXT[], budget float)
LANGUAGE plpgsql
AS $$
DECLARE
    result_status TEXT;
    batch_size CONSTANT integer := 8;
    column_list TEXT;
BEGIN
    column_list := array_to_string(selected_columns, ', ');
    -- 4. run filter phase with user defined batch size.
    RAISE NOTICE '4. run filter phase with user defined batch size.';
    EXECUTE format('
        WITH batch_rows AS (
            SELECT %s
            FROM %I
            ORDER BY RANDOM()
            LIMIT %s OFFSET 0
        )
        SELECT filtering_phase(
            json_agg(row_to_json(t))::text
        )
        FROM batch_rows AS t', column_list, dataset, batch_size) INTO result_status;
    RAISE NOTICE 'Result status: %', result_status;
END; $$;
