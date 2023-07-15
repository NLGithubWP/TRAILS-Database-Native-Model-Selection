

#[cfg(test)]
#[cfg(feature = "python")]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_hello_pgrxdemo() {
        let task = "test task".to_string();
        let result = hello_pgrxdemo(task);
        let expected_result = JsonB(
            crate::bindings::bdfile::print_hell(
                &json!({ "task": "test task" })
            )
        );
        assert_eq!(result, expected_result);
    }
}
