use std::error::Error;
use std::path::PathBuf;
use std::process::Command;

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

fn installed_home() -> Result<PathBuf, Box<dyn Error>> {
    let base = dirs::data_local_dir().ok_or("Could not determine local data directory")?;

    let scheduler = base.join("alchemist");

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

    cmd.arg("run")
        .arg("-e")
        .arg(environment)
        .arg("--manifest-path")
        .arg(scheduler.join("pixi.toml"))
        .arg("python");

    Ok(cmd)
}
