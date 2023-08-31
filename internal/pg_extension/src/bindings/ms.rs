use serde_json::json;
use std::collections::HashMap;
use pgrx::Spi;
use crate::bindings::ml_register::PY_MODULE;
use crate::bindings::ml_register::run_python_function;


pub fn profiling_filtering_phase(
    task: &String
) -> serde_json::Value {
    run_python_function(&PY_MODULE, task, "profiling_filtering_phase")
}


pub fn profiling_refinement_phase(
    task: &String
) -> serde_json::Value {
    run_python_function(&PY_MODULE, task, "profiling_refinement_phase")
}


pub fn coordinator(
    task: &String
) -> serde_json::Value {
    run_python_function(&PY_MODULE, task, "coordinator")
}


pub fn filtering_phase(
    task: &String
) -> serde_json::Value {
    run_python_function(&PY_MODULE, task, "filtering_phase")
}


pub fn refinement_phase() -> serde_json::Value {
    let task = "refinement_phase".to_string();
    run_python_function(&PY_MODULE, &task, "refinement_phase")
}


// this two are filtering + refinement in UDF runtime
pub fn model_selection(
    task: &String
) -> serde_json::Value {
    run_python_function(&PY_MODULE, task, "model_selection")
}


pub fn model_selection_workloads(
    task: &String
) -> serde_json::Value {
    run_python_function(&PY_MODULE, task, "model_selection_workloads")
}


// this two are filtering + refinement in GPU server
pub fn model_selection_trails(
    task: &String
) -> serde_json::Value {
    run_python_function(&PY_MODULE, task, "model_selection_trails")
}


pub fn model_selection_trails_workloads(
    task: &String
) -> serde_json::Value {
    run_python_function(&PY_MODULE, task, "model_selection_trails_workloads")
}

// micro benchmarks

pub fn benchmark_filtering_phase_latency(
    task: &String
) -> serde_json::Value {
    run_python_function(&PY_MODULE, task, "benchmark_filtering_phase_latency")
}

pub fn benchmark_filtering_latency_in_db(
    explore_models: i32, config_file: &String) -> serde_json::Value {

    let dataset_name = "pg_extension";
    let mut last_id = 0;
    let mut eva_results = serde_json::Value::Null; // Initializing the eva_results

    for i in 1..explore_models {

        // Step 1: Initialize State in Python
        let mut task_map = HashMap::new();
        task_map.insert("config_file", config_file.clone());
        task_map.insert("eva_results", eva_results.to_string());
        let task_json = json!(task_map).to_string();

        let sample_result = run_python_function(
            &PY_MODULE,
            &task_json,
            "in_db_filtering_state_init");

        // Step 2: Data Retrieval in Rust using SPI
        // Construct the SQL query
        let (results, max_id) = Spi::connect(|client| {
            let mut inner_results = Vec::new(); // Assuming row type matches with this

            // Construct the SQL query
            let query = format!(
                "SELECT * FROM frappe_train WHERE id > {} ORDER BY id ASC LIMIT 32",
                , last_id
            );

            let mut tup_table = client.select(&query, None, None)?;

            while let Some(row) = tup_table.next() {
                let mut map = HashMap::new();
                map.insert("id".to_string(), row["id"].value::<i64>().unwrap_or(0));
                // Continue for other columns...
                // map.insert("columnName".to_string(), row["columnName"].value::<ColumnType>().unwrap_or_default());
                inner_results.push(map);
            }

            if let Some(max_id_value) = inner_results.iter().map(|row| row["id"].value::<i64>().unwrap_or(0)).max() {
                Ok((inner_results, max_id_value))
            } else {
                Ok((inner_results, -1i64))
            }
        }).expect("TODO: panic message");

        last_id = max_id as i32;

        // Step 3: Data Processing in Python
        let mut eva_task_map = HashMap::new();
        eva_task_map.insert("config_file", config_file.clone());
        eva_task_map.insert("sample_result", sample_result.to_string());
        let mini_batch_json = serde_json::json!(results).to_string();
        eva_task_map.insert("mini_batch", mini_batch_json);
        let eva_task_json = json!(eva_task_map).to_string(); // Corrected this line

        eva_results = run_python_function(
            &PY_MODULE,
            &eva_task_json,
            "in_db_filtering_evaluate");
    }

    // Step 4: Return to PostgreSQL
    return serde_json::json!("Done");
}
