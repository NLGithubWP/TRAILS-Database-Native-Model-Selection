use pgrx::prelude::*;
pgrx::pg_module_magic!();
use serde_json::json;
use std::collections::HashMap;

pub mod bindings;


/*
 * @param mini_batch: mini_batch of data. Assume all columns are string type in
 * libsvm codding
 */
#[cfg(feature = "python")]
#[pg_extern(immutable, parallel_safe, name = "profiling_filtering_phase")]
#[allow(unused_variables)]
pub fn profiling_filtering_phase(mini_batch: String, config_file: String) -> String {
    let mut task_map = HashMap::new();
    task_map.insert("mini_batch", mini_batch);
    task_map.insert("config_file", config_file);
    let task_json = json!(task_map).to_string();
    crate::bindings::model_selection::profiling_filtering_phase(&task_json).to_string()
}

/*
 * @param mini_batch: training for one iteration.
 * libsvm codding
 */
#[cfg(feature = "python")]
#[pg_extern(immutable, parallel_safe, name = "profiling_refinement_phase")]
#[allow(unused_variables)]
pub fn profiling_refinement_phase(mini_batch: String, config_file: String) -> String {
    let mut task_map = HashMap::new();
    task_map.insert("mini_batch", mini_batch);
    task_map.insert("config_file", config_file);
    let task_json = json!(task_map).to_string();
    crate::bindings::model_selection::profiling_refinement_phase(&task_json).to_string()
}

/*
 * @param mini_batch: training for one iteration.
 * libsvm codding
 */
#[cfg(feature = "python")]
#[pg_extern(immutable, parallel_safe, name = "coordinator")]
#[allow(unused_variables)]
pub fn coordinator(time_score: String, time_train: String, time_budget: String, only_phase1: bool,
                   config_file: String) -> String {
    let mut task_map = HashMap::new();
    task_map.insert("budget", time_budget);
    task_map.insert("score_time_per_model", time_score);
    task_map.insert("train_time_per_epoch", time_train);
    task_map.insert("only_phase1", only_phase1.to_string());
    task_map.insert("config_file", config_file);
    let task_json = json!(task_map).to_string();
    crate::bindings::model_selection::coordinator(&task_json).to_string()
}


/*
 * @param mini_batch: mini_batch of data. Assume all columns are string type in
 * libsvm codding
 */
#[cfg(feature = "python")]
#[pg_extern(immutable, parallel_safe, name = "filtering_phase")]
#[allow(unused_variables)]
pub fn filtering_phase(mini_batch: String, n: i32, k: i32, config_file: String) -> String {
    let mut task_map = HashMap::new();
    task_map.insert("mini_batch", mini_batch);
    task_map.insert("n", n.to_string());
    task_map.insert("k", k.to_string());
    task_map.insert("config_file", config_file);
    let task_json = json!(task_map).to_string();
    crate::bindings::model_selection::filtering_phase(&task_json).to_string()
}


#[cfg(feature = "python")]
#[pg_extern(immutable, parallel_safe, name = "refinement_phase")]
#[allow(unused_variables)]
pub fn refinement_phase(config_file: String) -> String {
    let mut task_map = HashMap::new();
    task_map.insert("config_file", config_file);
    let task_json = json!(task_map).to_string();
    crate::bindings::model_selection::refinement_phase().to_string()
}



