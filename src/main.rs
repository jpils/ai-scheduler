#![allow(unused)]

mod lammps;
mod paths;
mod training;
mod vasp;
mod watcher;
mod install;
mod job_template;
mod disagreement;
mod types;
mod slurm_client;
mod pipeline;

use disagreement::DisagreementSettings;
use lammps::{LammpsManager, MdModelPackage};
use serde::Deserialize;
use std::fs;
use std::path::PathBuf;
use std::time::Duration;
use pipeline::{
    DftCode,
    DftStep,
    DisagreementMode as PipelineDisagreementMode,
    EnergyMode as PipelineEnergyMode,
    MdEngine,
    MdStep,
    ModelBackend as PipelineModelBackend,
    QbcMethod,
    Pipeline,
    PipelineCtx,
    QbcStep,
    StepCtx,
    TrainingStep,
};
use pipeline::runner::{Runner, SlurmRunner};

use crate::pipeline::runner::DryRunner;

#[derive(Debug, Deserialize)]
struct Config {
    project: ProjectConfig,
    training: TrainingConfig,
    committee: CommitteeConfig,
    disagreement: Option<DisagreementConfig>,
}

#[derive(Debug, Deserialize)]
struct ProjectConfig {
    generations: u32,
}

#[derive(Debug, Deserialize)]
struct TrainingConfig {
    backend: Backend,
    energy_mode: EnergyMode,
}

#[derive(Debug, Deserialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
enum Backend {
    Upet,
    N2p2,
}

#[derive(Debug, Deserialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
enum EnergyMode {
    Pet,
    Raw,
}

#[derive(Debug, Deserialize)]
struct CommitteeConfig {
    members: usize,
}

#[derive(Debug, Deserialize)]
struct DisagreementConfig {
    mode: Option<DisagreementMode>,
    max_selected: Option<usize>,
    min_rrmse: Option<f64>,
    max_rrmse: Option<f64>,
}

#[derive(Debug, Deserialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
enum DisagreementMode {
    Mock,
    Real,
}

impl DisagreementConfig {
    fn mode(&self) -> DisagreementMode {
        self.mode.unwrap_or(DisagreementMode::Mock)
    }

    fn settings(&self) -> DisagreementSettings {
        let defaults = DisagreementSettings::default();

        DisagreementSettings {
            max_selected: self.max_selected.unwrap_or(defaults.max_selected),
            min_rrmse: self.min_rrmse.unwrap_or(defaults.min_rrmse),
            max_rrmse: self.max_rrmse.unwrap_or(defaults.max_rrmse),
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() > 1 && args[1] == "init" {
        if let Err(e) = install::initialize() {
            eprintln!("❌ {}", e);
        }
        return;
    }

    let project_dir =
        std::env::current_dir().expect("Failed to determine current working directory");

    let setup_dir = project_dir.join("setup");

    if !setup_dir.exists() {
        eprintln!("❌ Could not find setup directory.");
        eprintln!("Expected:");
        eprintln!("    {}", setup_dir.display());
        return;
    }

    let config_path = setup_dir.join("config.toml");

    let config_text = match fs::read_to_string(&config_path) {
        Ok(text) => text,
        Err(e) => {
            eprintln!("❌ Failed to read {}", config_path.display());
            eprintln!("{e}");
            return;
        }
    };

    let config: Config = match toml::from_str(&config_text) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("❌ Invalid config.toml");
            eprintln!("{e}");
            return;
        }
    };

    let total_generations = config.project.generations;

    if config.committee.members == 0 {
        eprintln!("❌ Invalid committee configuration.");
        eprintln!("`committee.members` must be greater than zero.");
        return;
    }

    let required_job_templates = [
        setup_dir.join("jobscripts").join("md_array.sh.template"),
        setup_dir.join("jobscripts").join("vasp_array.sh.template"),
    ];

    for template_path in required_job_templates {
        if !template_path.is_file() {
            eprintln!("❌ PRE-FLIGHT VALIDATION FAILED!");
            eprintln!("Missing job template: {}", template_path.display());
            return;
        }
    }

    if matches!(config.training.backend, Backend::Upet) {
        let required_upet_templates = [
            setup_dir.join("training").join("upet.yaml.template"),
            setup_dir
                .join("jobscripts")
                .join("upet_training_array.sh.template"),
        ];

        for template_path in required_upet_templates {
            if !template_path.is_file() {
                eprintln!("❌ PRE-FLIGHT VALIDATION FAILED!");

                eprintln!("Missing UPET template: {}", template_path.display());

                return;
            }
        }

        if config
            .disagreement
            .as_ref()
            .map(DisagreementConfig::mode)
            .is_some_and(|mode| matches!(mode, DisagreementMode::Real))
        {
            let disagreement_template = setup_dir
                .join("jobscripts")
                .join("upet_disagreement.sh.template");

            if !disagreement_template.is_file() {
                eprintln!("❌ PRE-FLIGHT VALIDATION FAILED!");
                eprintln!(
                    "Missing UPET disagreement template: {}",
                    disagreement_template.display()
                );
                return;
            }
        }
    }

    if matches!(config.training.backend, Backend::N2p2) {
        let required_n2p2_files = [
            setup_dir.join("training").join("input.nn"),
            setup_dir
                .join("jobscripts")
                .join("n2p2_scaling_array.sh.template"),
            setup_dir
                .join("jobscripts")
                .join("n2p2_training_array.sh.template"),
        ];

        for path in required_n2p2_files {
            if !path.is_file() {
                eprintln!("❌ PRE-FLIGHT VALIDATION FAILED!");
                eprintln!("Missing n2p2 setup file: {}", path.display());
                return;
            }
        }

        if config
            .disagreement
            .as_ref()
            .map(DisagreementConfig::mode)
            .is_some_and(|mode| matches!(mode, DisagreementMode::Real))
        {
            let disagreement_template = setup_dir
                .join("jobscripts")
                .join("n2p2_disagreement.sh.template");

            if !disagreement_template.is_file() {
                eprintln!("❌ PRE-FLIGHT VALIDATION FAILED!");
                eprintln!(
                    "Missing n2p2 disagreement template: {}",
                    disagreement_template.display()
                );
                return;
            }
        }
    }

    // ==========================================================
    // 🔍 PRE-FLIGHT ASSET VALIDATION LOOP
    // ==========================================================
    println!("🔍 Performing pre-flight asset validation...");

    let seed_dataset = {
        let extxyz = setup_dir.join("training").join("seed_dataset.extxyz");
        let xyz = setup_dir.join("training").join("seed_dataset.xyz");

        if extxyz.is_file() {
            extxyz
        } else if xyz.is_file() {
            xyz
        } else {
            eprintln!("❌ PRE-FLIGHT VALIDATION FAILED!");
            eprintln!("Missing seed dataset. Expected one of:");
            eprintln!("    {}", extxyz.display());
            eprintln!("    {}", xyz.display());
            return;
        }
    };

    println!(" ✓ Found seed dataset: {}", seed_dataset.display());

    for vasp_input in ["INCAR", "KPOINTS", "POTCAR"] {
        let path = setup_dir.join("vasp").join(vasp_input);

        if !path.is_file() {
            eprintln!("❌ PRE-FLIGHT VALIDATION FAILED!");
            eprintln!("Missing VASP input: {}", path.display());
            return;
        }
    }

    let lammps_data = setup_dir.join("lammps").join("data").join("lmp.data");

    if !lammps_data.is_file() {
        eprintln!("❌ PRE-FLIGHT VALIDATION FAILED!");
        eprintln!("Missing LAMMPS data file: {}", lammps_data.display());
        return;
    }

    // Check LAMMPS generation files
    for gen_num in 1..=total_generations {
        if let Err(e) = LammpsManager::find_input_file(&setup_dir, gen_num) {
            println!(
                "❌ PRE-FLIGHT VALIDATION FAILED! Gen {} missing input. Details: {}",
                gen_num, e
            );
            return;
        }
    }

    let checkpoint_required = matches!(config.training.backend, Backend::Upet)
        || matches!(config.training.energy_mode, EnergyMode::Pet);

    let checkpoint_file: Option<PathBuf> = if checkpoint_required {
        let mut checkpoint_path = None;
        let training_setup_dir = setup_dir.join("training");

        if let Ok(entries) = fs::read_dir(&training_setup_dir) {
            for entry in entries.flatten() {
                let path = entry.path();

                if path.is_file()
                    && path
                        .extension()
                        .is_some_and(|extension| extension == "ckpt")
                {
                    checkpoint_path = Some(path);
                    break;
                }
            }
        }

        match checkpoint_path {
            Some(path) => {
                println!(
                    " ✓ Found foundation model checkpoint: {:?}",
                    path.file_name().unwrap_or_default()
                );

                Some(path)
            }

            None => {
                eprintln!("❌ A checkpoint (*.ckpt) is required.");
                eprintln!(
                    "UPET training and PET energy correction require a foundation checkpoint in:"
                );
                eprintln!("    {}", training_setup_dir.display());
                return;
            }
        }
    } else {
        println!(" ✓ No foundation checkpoint is required.");

        None
    };

    println!(" ✓ All required foundation files validated successfully.");

    // ==========================================================
    // 🔄 THE MASTER GENERATION LOOP
    // ==========================================================
    let runner = match DryRunner::new(&project_dir) {
        Ok(runner) => runner,
        Err(error) => {
            eprintln!(" ❌ Failed to initialize dry runner: {}", error);
            return;
        }
    };

    for gen_num in 1..=total_generations {
        println!(
            "\n🌀 Starting Generation {}/{}...",
            gen_num, total_generations
        );

        let pipeline_backend = match config.training.backend {
            Backend::Upet => PipelineModelBackend::Upet,
            Backend::N2p2 => PipelineModelBackend::N2p2,
        };

        let pipeline_energy_mode = match config.training.energy_mode {
            EnergyMode::Pet => PipelineEnergyMode::Pet,
            EnergyMode::Raw => PipelineEnergyMode::Raw,
        };

        let model_package = match config.training.backend {
            Backend::Upet => Some(MdModelPackage::UpetModel),
            Backend::N2p2 => Some(MdModelPackage::N2p2Inputs),
        };

        let disagreement_settings = config
            .disagreement
            .as_ref()
            .map(DisagreementConfig::settings)
            .unwrap_or_default();

        let disagreement_mode = match config
            .disagreement
            .as_ref()
            .map(DisagreementConfig::mode)
            .unwrap_or(DisagreementMode::Mock)
        {
            DisagreementMode::Mock => PipelineDisagreementMode::Mock,
            DisagreementMode::Real => PipelineDisagreementMode::Real,
        };

        let step_ctx = || {
            StepCtx::new(
                project_dir.clone(),
                setup_dir.clone(),
                setup_dir.join("jobscripts"),
            )
        };

        let pipeline = Pipeline::new(vec![
            Box::new(TrainingStep::new(
                pipeline_backend,
                step_ctx(),
                config.committee.members,
                checkpoint_file.clone(),
                pipeline_energy_mode,
            )),
            Box::new(MdStep::new(
                MdEngine::Lammps,
                step_ctx(),
                config.committee.members,
                model_package,
            )),
            Box::new(QbcStep::new(
                QbcMethod::Rrmsfd,
                step_ctx(),
                pipeline_backend,
                config.committee.members,
                disagreement_settings,
                disagreement_mode,
            )),
            Box::new(DftStep::new(
                DftCode::Vasp,
                step_ctx(),
            )),
        ]);

        let pipeline_ctx = PipelineCtx {
            project_dir: project_dir.clone(),
            generation: gen_num,
            poll_interval: Duration::from_secs(30),
            max_retries: 0,
            dry_run: false,
            dry_config_limit: None,
        };

        if let Err(error) = runner.run(&pipeline, &pipeline_ctx) {
            eprintln!(" ❌ Generation {} failed: {}", gen_num, error);
            return;
        }
    }
}
