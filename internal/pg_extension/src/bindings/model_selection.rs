use log::error;
use once_cell::sync::Lazy;
use pyo3::prelude::*;
use pyo3::types::PyTuple;


/*
 Python Module Path
 */
static PY_MODULE: Lazy<Py<PyModule>> = Lazy::new(|| {
    Python::with_gil(|py| -> Py<PyModule> {
        let src = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../ml/model_selection/pg_interface.py"
        ));
        PyModule::from_code(py, src, "", "").unwrap().into()
    })
});


/*
 * Rust Call Python,
 * @param parameters: the task parameter
 * @param function_name: the name of the python function
 */
pub fn run_python_function(
    parameters: &String,
    function_name: &str,
) -> serde_json::Value {
    let parameters_str = parameters.to_string();
    let results = Python::with_gil(|py| -> String {
        let run_script: Py<PyAny> = PY_MODULE.getattr(py, function_name).unwrap().into();
        let result = run_script.call1(
            py,
            PyTuple::new(
                py,
                &[parameters_str.into_py(py)],
            ),
        );
        let result = match result {
            Err(e) => {
                let traceback = e.traceback(py).unwrap().format().unwrap();
                error!("{traceback} {e}");
                format!("{traceback} {e}")
            }
            Ok(o) => o.extract(py).unwrap(),
        };
        result
    });

    serde_json::from_str(&results).unwrap()
}


pub fn profiling_filtering_phase(
    task: &String
) -> serde_json::Value {
    run_python_function(task, "profiling_filtering_phase")
}


pub fn profiling_refinement_phase(
    task: &String
) -> serde_json::Value {
    run_python_function(task, "profiling_refinement_phase")
}


pub fn coordinator(
    task: &String
) -> serde_json::Value {
    run_python_function(task, "coordinator")
}


pub fn filtering_phase(
    task: &String
) -> serde_json::Value {
    run_python_function(task, "filtering_phase")
}


pub fn refinement_phase() -> serde_json::Value {
    let task = "refinement_phase".to_string();
    run_python_function(&task, "refinement_phase")
}






