use std::fs;
use std::io;
use std::path::PathBuf;

/// Returns the installation directory.
///
/// Linux:
/// ~/.local/share/alchemist
pub fn install_dir() -> io::Result<PathBuf> {
    let base = dirs::data_local_dir().ok_or_else(|| {
        io::Error::new(io::ErrorKind::Other, "Could not determine data directory")
    })?;

    Ok(base.join("alchemist"))
}

fn copy_file(from: &std::path::Path, to: &std::path::Path) -> io::Result<()> {
    if let Some(parent) = to.parent() {
        fs::create_dir_all(parent)?;
    }

    fs::copy(from, to)?;
    Ok(())
}

fn copy_directory(from: &std::path::Path, to: &std::path::Path) -> io::Result<()> {
    fs::create_dir_all(to)?;

    for entry in fs::read_dir(from)? {
        let entry = entry?;
        let source = entry.path();
        let destination = to.join(entry.file_name());

        // Skip Python cache directories
        if source.is_dir() && entry.file_name() == "__pycache__" {
            continue;
        }

        // Skip compiled Python bytecode
        if source.is_file() {
            if let Some(ext) = source.extension() {
                if ext == "pyc" {
                    continue;
                }
            }
        }

        if source.is_dir() {
            copy_directory(&source, &destination)?;
        } else {
            copy_file(&source, &destination)?;
        }
    }

    Ok(())
}

fn repository_root() -> io::Result<PathBuf> {
    let mut dir = std::env::current_exe()?;

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

    Err(io::Error::new(
        io::ErrorKind::NotFound,
        "Could not locate ALCHEMIST repository.",
    ))
}

/// Initializes the scheduler installation.
///
pub fn initialize() -> io::Result<()> {
    let install_dir = install_dir()?;
    let repository = repository_root()?;

    fs::create_dir_all(&install_dir)?;

    println!("Installing into");
    println!("  {}", install_dir.display());

    copy_directory(&repository.join("python"), &install_dir.join("python"))?;

    copy_file(
        &repository.join("pixi.toml"),
        &install_dir.join("pixi.toml"),
    )?;

    copy_file(
        &repository.join("pixi.lock"),
        &install_dir.join("pixi.lock"),
    )?;

    println!("✓ Python resources copied.");
    println!("✓ Initialization complete.");
    println!("You can now run `alchemist` from any project directory.");

    Ok(())
}
