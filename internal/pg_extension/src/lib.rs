
use pgrx::prelude::*;

pgrx::pg_module_magic!();

use serde_json::json;
use std::collections::HashMap;

pub mod bindings;


#[cfg(feature = "python")]
#[pg_extern(immutable, parallel_safe, name = "hello_pg")]
#[allow(unused_variables)]
pub fn hello_pgrxdemo(task: String) -> String {

    println!("Done");

    let mut task_map = HashMap::new();
    task_map.insert("task", task);
    let task_json = json!(task_map);
    crate::bindings::bdfile::print_hell(&task_json).to_string()

}


