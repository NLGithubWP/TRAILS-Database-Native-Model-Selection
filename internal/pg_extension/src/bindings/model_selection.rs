
use log::error;
use once_cell::sync::Lazy;
use pyo3::prelude::*;
use pyo3::types::PyTuple;


static PY_MODULE: Lazy<Py<PyModule>> = Lazy::new(|| {
    Python::with_gil(|py| -> Py<PyModule> {
        let src = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../ml/model_selection/pg_interface.py"
        ));
        PyModule::from_code(py, src, "", "").unwrap().into()
    })
});


pub fn filtering_phase(
    task: &serde_json::Value,
) -> serde_json::Value {
    let task = serde_json::to_string(task).unwrap();
    println!("{}", task);
    let results = Python::with_gil(|py| -> String {
        let print_hell: Py<PyAny> = PY_MODULE.getattr(py, "filtering_phase").unwrap().into();
        let result = print_hell.call1(
            py,
            PyTuple::new(
                py,
                &[task.into_py(py)],
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

