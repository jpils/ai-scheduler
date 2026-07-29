use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

const ALCHEMIST_HOME_ENV: &str = "ALCHEMIST_HOME";

fn repository_root() -> Result<PathBuf, Box<dyn Error>> {
    let mut dir = std::env::current_exe()?;

    // target/debug/alchemist -> target/debug
    dir.pop();

    loop {
        if dir.join("Cargo.toml").exists()
            && dir.join("python").is_dir()
            && dir.join("pixi.toml").exists()
        {
            return Ok(dir);
        }

        if !dir.pop() {
            break;
        }
    }

    Err("Could not locate the ALCHEMIST repository.".into())
}

pub fn default_installed_home() -> Result<PathBuf, Box<dyn Error>> {
    if let Some(path) = std::env::var_os(ALCHEMIST_HOME_ENV) {
        return Ok(PathBuf::from(path));
    }

    let base = dirs::data_local_dir().ok_or("Could not determine local data directory")?;

    Ok(base.join("alchemist"))
}

fn installed_home() -> Result<PathBuf, Box<dyn Error>> {
    let scheduler = default_installed_home()?;

    if scheduler.exists() {
        Ok(scheduler)
    } else {
        Err("ALCHEMIST is not initialized.\nRun `cargo run -- init` first.".into())
    }
}

pub fn scheduler_home() -> Result<PathBuf, Box<dyn Error>> {
    let exe = std::env::current_exe()?;

    // Development build: target/debug or target/release
    if exe.components().any(|c| {
        let s = c.as_os_str().to_string_lossy();
        s == "target"
    }) {
        return repository_root();
    }

    // Installed binary
    installed_home()
}

/// Construct a command that executes Python inside the scheduler's
/// Pixi environment.
pub fn pixi_python(environment: &str) -> Result<Command, Box<dyn Error>> {
    let scheduler = scheduler_home()?;

    let mut cmd = Command::new("pixi");

    cmd.current_dir(&scheduler);
    set_default_cache_env(&mut cmd, &scheduler)?;

    cmd.arg("run")
        .arg("-e")
        .arg(environment)
        .arg("--manifest-path")
        .arg(scheduler.join("pixi.toml"))
        .arg("python");

    Ok(cmd)
}

fn set_default_cache_env(cmd: &mut Command, scheduler: &PathBuf) -> Result<(), Box<dyn Error>> {
    let cache_dir = scheduler.join("cache");

    set_default_env_path(cmd, "PIXI_CACHE_DIR", cache_dir.join("pixi"))?;
    set_default_env_path(cmd, "UV_CACHE_DIR", cache_dir.join("uv"))?;
    set_default_env_path(cmd, "XDG_CACHE_HOME", cache_dir.join("xdg"))?;

    Ok(())
}

fn set_default_env_path(cmd: &mut Command, key: &str, path: PathBuf) -> Result<(), Box<dyn Error>> {
    if std::env::var_os(key).is_some() {
        return Ok(());
    }

    fs::create_dir_all(&path)?;
    cmd.env(key, path);

    Ok(())
}
