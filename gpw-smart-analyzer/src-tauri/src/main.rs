// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use tauri::Manager;

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[tauri::command]
async fn get_system_info() -> Result<serde_json::Value, String> {
    let mut info = serde_json::json!({
        "platform": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "version": env!("CARGO_PKG_VERSION")
    });

    // Add system memory info
    if let Ok(memory) = sys_info::mem_info() {
        info["memory"] = serde_json::json!({
            "total": memory.total,
            "free": memory.free,
            "used": memory.total - memory.free
        });
    }

    Ok(info)
}

#[tauri::command]
fn open_external_url(url: String) -> Result<(), String> {
    open::that(&url).map_err(|e| e.to_string())
}

#[tauri::command]
fn save_settings(settings: String) -> Result<(), String> {
    use std::fs;
    use std::path::PathBuf;

    let app_data_dir = tauri::api::path::app_data_dir(&tauri::Config::default())
        .ok_or("Failed to get app data directory")?;

    let settings_file = app_data_dir.join("settings.json");

    // Ensure directory exists
    if let Some(parent) = settings_file.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }

    fs::write(settings_file, settings).map_err(|e| e.to_string())
}

#[tauri::command]
fn load_settings() -> Result<String, String> {
    use std::fs;

    let app_data_dir = tauri::api::path::app_data_dir(&tauri::Config::default())
        .ok_or("Failed to get app data directory")?;

    let settings_file = app_data_dir.join("settings.json");

    if settings_file.exists() {
        fs::read_to_string(settings_file).map_err(|e| e.to_string())
    } else {
        Ok("{}".to_string()) // Return empty JSON if no settings file
    }
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_window::init())
        .invoke_handler(tauri::generate_handler![
            greet,
            get_system_info,
            open_external_url,
            save_settings,
            load_settings
        ])
        .setup(|app| {
            #[cfg(debug_assertions)]
            {
                let window = app.get_webview_window("main").unwrap();
                window.open_devtools();
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}