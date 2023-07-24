use pgrx::prelude::*;
pgrx::pg_module_magic!();
use serde_json::json;
use std::collections::HashMap;

pub mod bindings;


/*
 * @param dataset: relation name.
 * @param columns: queried columns.
 * @param time_budget: user pre-defined time budget.
 */
// #[cfg(feature = "python")]
// #[pg_extern(immutable, parallel_safe, name = "model_selection")]
// pub fn model_selection(dataset: String, columns: Vec<String>, time_budget: i32) -> String {
//     let mut task_map = HashMap::new();
//     task_map.insert("dataset", dataset);
//     task_map.insert("columns", columns.join(","));
//     task_map.insert("time_budget", time_budget.to_string());
//     let task_json = json!(task_map);
//     crate::bindings::model_selection::model_selection(&task_json).to_string()
// }


/*
 * @param mini_batch: mini_batch of data. Assume all columns are string type in
 * libsvm codding
 */
#[cfg(feature = "python")]
#[pg_extern(immutable, parallel_safe, name = "filtering_phase")]
#[allow(unused_variables)]
pub fn filtering_phase(mini_batch: String) -> String {
    crate::bindings::model_selection::filtering_phase(&mini_batch).to_string()
}


/*
 * @param mini_batch: mini_batch of data. Assume all columns are string type in
 * libsvm codding
 */
#[cfg(feature = "python")]
#[pg_extern(immutable, parallel_safe, name = "profiling_filtering_phase")]
#[allow(unused_variables)]
pub fn profiling_filtering_phase(mini_batch: String) -> String {
    crate::bindings::model_selection::profiling_filtering_phase(&mini_batch).to_string()
}

/*
 * @param mini_batch: mini_batch of data. Assume all columns are string type in
 * libsvm codding
 */
#[cfg(feature = "python")]
#[pg_extern(immutable, parallel_safe, name = "profiling_refinement_phase")]
#[allow(unused_variables)]
pub fn profiling_refinement_phase(mini_batch: String) -> String {
    crate::bindings::model_selection::profiling_refinement_phase(&mini_batch).to_string()
}

//
// #[cfg(feature = "python")]
// #[pg_extern(immutable, parallel_safe, name = "profiling")]
// #[allow(unused_variables)]
// pub fn profiling(rows: Array<Array<String>>) -> i32 {
//     // Implement your logic here.
//     // You can access individual strings using rows[i][j]
//     0
// }
//
// #[cfg(feature = "python")]
// #[pg_extern(immutable, parallel_safe, name = "refinement_phase")]
// #[allow(unused_variables)]
// pub fn refinement_phase(rows: Array<Array<String>>) -> i32 {
//     // Implement your logic here.
//     // You can access individual strings using rows[i][j]
//     0
// }
//
//
// #[cfg(feature = "python")]
// #[pg_extern(immutable, parallel_safe, name = "coordinator")]
// #[allow(unused_variables)]
// pub fn coordinator(rows: Array<Array<String>>) -> i32 {
//     // Implement your logic here.
//     // You can access individual strings using rows[i][j]
//     0
// }
//
