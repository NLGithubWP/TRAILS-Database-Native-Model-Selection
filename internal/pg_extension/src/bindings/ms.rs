use std::ffi::CString;
use log::error;
use once_cell::sync::Lazy;
use pgrx::spi;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use serde_json::json;
use std::collections::HashMap;
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

    let dataset_name = "your_dataset_name"; // Adjust this as per your requirement.
    let mut last_id = 0;
    let mut eva_results = serde_json::Value::Null; // Initializing the eva_results

    // Step 1: Initialize State in Python
    for i in 1..explore_models {

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
        let sql = format!(
            "SELECT * FROM {}_train WHERE id > {} ORDER BY id ASC LIMIT 32",
            dataset_name, last_id
        );
        let c_sql = CString::new(sql).unwrap();

        // SPI call to execute the SQL
        let data;
        unsafe {
            data = spi::SPI_execute(c_sql.as_ptr(), true, 0);
        }

        // Process the result
        if let Some(rows) = data {
            if !rows.is_empty() {
                last_id = rows.iter().map(|row| row[0]).max().unwrap_or(last_id);
            } else {
                last_id = -1;
            }
        }

        let mut eva_task_map = HashMap::new();
        eva_task_map.insert("config_file", config_file.clone());
        eva_task_map.insert("sample_result", sample_result.to_string());
        let mini_batch_json = serde_json::json!(data).to_string();
        eva_task_map.insert("mini_batch", mini_batch_json);
        let eva_task_json = json!(eva_task_map).to_string(); // Corrected this line

        // Step 3: Data Processing in Python
        eva_results = run_python_function(
            &PY_MODULE,
            &eva_task_json,
            "in_db_filtering_evaluate");
    }

    // Step 4: Return to PostgreSQL
    return serde_json::json!("Done");
}

