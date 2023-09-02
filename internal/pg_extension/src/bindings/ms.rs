use serde_json::json;
use std::collections::HashMap;
use pgrx::prelude::*;
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

        // here it cache a state
        let sample_result = run_python_function(
            &PY_MODULE,
            &task_json,
            "in_db_filtering_state_init");

        // 2. query data via SPI
        let results: Result<Vec<Vec<String>>, String> = Spi::connect(|client| {
            let query = format!("SELECT * FROM frappe_train LIMIT 3");
            let mut cursor = client.open_cursor(&query, None);
            let table = match cursor.fetch(32) {
                Ok(table) => table,
                Err(e) => return Err(e.to_string()), // Convert the error to a string and return
            };

            let mut mini_batch = Vec::new();

            for row in table.into_iter() {
                let mut each_row = Vec::new();
                // add primary key
                let col0 = match row.get::<i32>(1) {
                    Ok(val) => val.map(|i| i.to_string()).unwrap_or_default(),
                    Err(e) => e.to_string(),
                };
                each_row.push(col0);
                // add label
                let col1 = match row.get::<i32>(2) {
                    Ok(val) => val.map(|i| i.to_string()).unwrap_or_default(),
                    Err(e) => e.to_string(),
                };
                each_row.push(col1);
                // add fields
                let texts: Vec<String> = (3..13)
                    .filter_map(|i| {
                        match row.get::<&str>(i) {
                            Ok(Some(s)) => Some(s.to_string()),
                            Ok(None) => None,
                            Err(e) => Some(e.to_string()),  // Convert error to string
                        }
                    }).collect();
                each_row.extend(texts);
                mini_batch.push(each_row)
            }
            // return
            Ok(mini_batch)
        });
        // serialize the mini-batch data
        let tup_table = match results {
            Ok(data) => {
                serde_json::json!({
                        "status": "success",
                        "data": data
                    })
            }
            Err(e) => {
                serde_json::json!({
                    "status": "error",
                    "message": format!("Error while connecting: {}", e)
                })
            }
        };


        // Step 3: Data Processing in Python
        let mut eva_task_map = HashMap::new();
        eva_task_map.insert("config_file", config_file.clone());
        eva_task_map.insert("sample_result", sample_result.to_string());
        let mini_batch_json = tup_table.to_string();
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

