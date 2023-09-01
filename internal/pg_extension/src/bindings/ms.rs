use serde_json::json;
use std::collections::HashMap;
use pgrx::prelude::*;
use crate::bindings::ml_register::PY_MODULE;
use crate::bindings::ml_register::run_python_function;
use crate::bindings::model::Frappe;
use std::fs;

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
        let results = Spi::connect(|client| {
            let query = format!("SELECT * FROM frappe_train WHERE id > {} ORDER BY id ASC LIMIT 32", last_id);
            let spi_result = client.select(&query, None, None)?;

            let mut logger_ap = HashMap::new();
            logger_ap.insert("spi_result".to_string(), spi_result);
            let json_data = serde_json::to_string(&logger_ap).unwrap();
            fs::write("/project/TRAILS/log_score_time_frappe/map.json", json_data).expect("Unable to write to file");

            let rows = spi_result.collect::<Vec<_>>();
            Ok(rows)
        });


        let tup_table = match results {
            Ok(Some(table)) => table,  // Handle the case where we have some data
            Ok(None) => {
                // Handle the case where there's no data (this depends on your needs)
                eprintln!("No data fetched.");
                return serde_json::json!({
            "status": "error",
            "message": "No data fetched from the database"
        });
            },
            Err(e) => {
                // Handle error case and extract the error message
                let error_msg = format!("Error while fetching data: {:?}", e);
                eprintln!("{}", &error_msg);
                return serde_json::json!({
            "status": "error",
            "message": error_msg
        });
            }
        };

        // Now serialize the success case
        let mini_batch_json = serde_json::json!(tup_table).to_string();

        // Step 3: Data Processing in Python
        let mut eva_task_map = HashMap::new();
        eva_task_map.insert("config_file", config_file.clone());
        eva_task_map.insert("sample_result", sample_result.to_string());
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

