
use serde::{Serialize, Deserialize};


#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Frappe {
    id: i32,
    label: i32,
    col1: String,
    col2: String,
    col3: String,
    col4: String,
    col5: String,
    col6: String,
    col7: String,
    col8: String,
    col9: String,
    col10: String,
}
