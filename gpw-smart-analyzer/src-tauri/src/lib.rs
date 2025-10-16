// src-tauri/src/lib.rs

// Re-export commonly used items from external crates
pub use sys_info;
pub use open;

// Add any shared utilities or helper functions here
pub mod utils {
    use serde_json::Value;

    pub fn format_bytes(bytes: u64) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        let mut size = bytes as f64;
        let mut unit_index = 0;

        while size >= 1024.0 && unit_index < UNITS.len() - 1 {
            size /= 1024.0;
            unit_index += 1;
        }

        format!("{:.2} {}", size, UNITS[unit_index])
    }

    pub fn validate_url(url: &str) -> bool {
        url.starts_with("http://") || url.starts_with("https://")
    }

    pub fn get_app_version() -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }
}