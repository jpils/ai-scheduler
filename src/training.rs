use std::fs;
use std::path::{Path, PathBuf};

use crate::job_template::{render_job_template, render_template};

pub struct TrainingWorkspace;

impl TrainingWorkspace {
    pub fn create_upet_workspace(
        project_dir: &Path,
        setup_dir: &Path,
        generation: u32,
        committee_members: usize,
        checkpoint: &Path,
        energy_key: &str,
    ) -> Result<PathBuf, String> {
        if committee_members == 0 {
            return Err("Cannot prepare UPET training for zero committee members.".to_string());
        }

        if !checkpoint.is_file() {
            return Err(format!(
                "UPET checkpoint does not exist: {}",
                checkpoint.display()
            ));
        }

        let generation_dir = project_dir
            .join("training")
            .join(format!("generation_{generation}"));

        let dataset_dir = generation_dir.join("dataset");

        let models_dir = generation_dir.join("models");

        let train_set = dataset_dir.join("train.extxyz");

        let validation_set = dataset_dir.join("validation.extxyz");

        let test_set = dataset_dir.join("test.extxyz");

        for dataset_path in [&train_set, &validation_set, &test_set] {
            if !dataset_path.is_file() {
                return Err(format!(
                    "Required UPET dataset file is missing: {}",
                    dataset_path.display()
                ));
            }
        }

        let yaml_template = setup_dir.join("training").join("upet.yaml.template");

        if !yaml_template.is_file() {
            return Err(format!(
                "Missing UPET training template: {}",
                yaml_template.display()
            ));
        }

        let job_template = setup_dir
            .join("jobscripts")
            .join("upet_training_array.sh.template");

        if !job_template.is_file() {
            return Err(format!(
                "Missing UPET training job template: {}",
                job_template.display()
            ));
        }

        fs::create_dir_all(&models_dir).map_err(|error| {
            format!(
                "Failed to create UPET model directory {}: {}",
                models_dir.display(),
                error
            )
        })?;

        let checkpoint_name = checkpoint
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                format!(
                    "Could not determine checkpoint file name from {}",
                    checkpoint.display()
                )
            })?;

        let checkpoint_string = format!("../../../../setup/training/{checkpoint_name}");

        let train_string = "../../dataset/train.extxyz".to_string();

        let validation_string = "../../dataset/validation.extxyz".to_string();

        let test_string = "../../dataset/test.extxyz".to_string();

        for member_index in 0..committee_members {
            let member_dir = models_dir.join(format!("member_{member_index:03}"));

            fs::create_dir_all(&member_dir).map_err(|error| {
                format!(
                    "Failed to create committee member directory {}: {}",
                    member_dir.display(),
                    error
                )
            })?;

            let member_yaml = member_dir.join("train.yaml");

            render_template(
                &yaml_template,
                &member_yaml,
                &[
                    ("checkpoint", checkpoint_string.clone()),
                    ("train_set", train_string.clone()),
                    ("validation_set", validation_string.clone()),
                    ("test_set", test_string.clone()),
                    ("energy_key", energy_key.to_string()),
                ],
            )
            .map_err(|error| {
                format!(
                    "Failed to render UPET configuration for member_{member_index:03}: {}",
                    error
                )
            })?;
        }

        let max_index = committee_members - 1;

        let job_script = generation_dir.join("submit_training_array.sh");

        render_job_template(&job_template, &job_script, generation, max_index)
            .map_err(|error| format!("Failed to render UPET training job script: {}", error))?;

        Ok(job_script)
    }

    pub fn create_mock_upet_models(
        project_dir: &Path,
        generation: u32,
        committee_members: usize,
    ) -> Result<(), String> {
        if committee_members == 0 {
            return Err("Cannot create mock UPET models for zero committee members.".to_string());
        }

        let models_dir = project_dir
            .join("training")
            .join(format!("generation_{generation}"))
            .join("models");

        for member_index in 0..committee_members {
            let member_name = format!("member_{member_index:03}");
            let member_dir = models_dir.join(&member_name);

            if !member_dir.is_dir() {
                return Err(format!(
                    "UPET member directory does not exist: {}",
                    member_dir.display()
                ));
            }

            let mock_model = member_dir.join("mock_trained_model.pt");

            fs::write(
                &mock_model,
                format!("mock UPET model for generation {generation}, {member_name}\n"),
            )
            .map_err(|error| {
                format!(
                    "Failed to create mock UPET model {}: {}",
                    mock_model.display(),
                    error
                )
            })?;
        }

        Ok(())
    }

    pub fn create_n2p2_workspace(
        project_dir: &Path,
        setup_dir: &Path,
        generation: u32,
        committee_members: usize,
    ) -> Result<(PathBuf, PathBuf), String> {
        if committee_members == 0 {
            return Err("Cannot prepare n2p2 training for zero committee members.".to_string());
        }

        let generation_dir = project_dir
            .join("training")
            .join(format!("generation_{generation}"));
        let dataset = generation_dir.join("dataset").join("input.data");
        let models_dir = generation_dir.join("models");
        let input_nn = setup_dir.join("training").join("input.nn");

        if !dataset.is_file() {
            return Err(format!(
                "Required n2p2 dataset file is missing: {}",
                dataset.display()
            ));
        }

        if !input_nn.is_file() {
            return Err(format!(
                "Missing n2p2 settings file: {}",
                input_nn.display()
            ));
        }

        fs::create_dir_all(&models_dir).map_err(|error| {
            format!(
                "Failed to create n2p2 model directory {}: {}",
                models_dir.display(),
                error
            )
        })?;

        for member_index in 0..committee_members {
            let member_dir = models_dir.join(format!("member_{member_index:03}"));
            let scaling_dir = member_dir.join("scaling");
            let train_dir = member_dir.join("train");

            for directory in [&member_dir, &scaling_dir, &train_dir] {
                fs::create_dir_all(directory).map_err(|error| {
                    format!(
                        "Failed to create n2p2 directory {}: {}",
                        directory.display(),
                        error
                    )
                })?;
            }

            for target_dir in [&member_dir, &scaling_dir, &train_dir] {
                fs::copy(&dataset, target_dir.join("input.data")).map_err(|error| {
                    format!(
                        "Failed to copy {} into {}: {}",
                        dataset.display(),
                        target_dir.display(),
                        error
                    )
                })?;

                fs::copy(&input_nn, target_dir.join("input.nn")).map_err(|error| {
                    format!(
                        "Failed to copy {} into {}: {}",
                        input_nn.display(),
                        target_dir.display(),
                        error
                    )
                })?;
            }
        }

        let scaling_template = setup_dir
            .join("jobscripts")
            .join("n2p2_scaling_array.sh.template");
        let training_template = setup_dir
            .join("jobscripts")
            .join("n2p2_training_array.sh.template");

        for template in [&scaling_template, &training_template] {
            if !template.is_file() {
                return Err(format!("Missing n2p2 job template: {}", template.display()));
            }
        }

        let max_index = committee_members - 1;
        let scaling_script = generation_dir.join("submit_scaling_array.sh");
        let training_script = generation_dir.join("submit_training_array.sh");

        render_job_template(&scaling_template, &scaling_script, generation, max_index)
            .map_err(|error| format!("Failed to render n2p2 scaling job script: {}", error))?;

        render_job_template(&training_template, &training_script, generation, max_index)
            .map_err(|error| format!("Failed to render n2p2 training job script: {}", error))?;

        Ok((scaling_script, training_script))
    }

    pub fn create_mock_n2p2_scaling_outputs(
        project_dir: &Path,
        generation: u32,
        committee_members: usize,
    ) -> Result<(), String> {
        if committee_members == 0 {
            return Err(
                "Cannot create mock n2p2 scaling outputs for zero committee members.".to_string(),
            );
        }

        let models_dir = project_dir
            .join("training")
            .join(format!("generation_{generation}"))
            .join("models");

        for member_index in 0..committee_members {
            let member_name = format!("member_{member_index:03}");
            let scaling_dir = models_dir.join(&member_name).join("scaling");

            if !scaling_dir.is_dir() {
                return Err(format!(
                    "n2p2 scaling directory does not exist: {}",
                    scaling_dir.display()
                ));
            }

            fs::write(
                scaling_dir.join("scaling.data"),
                format!("# mock n2p2 scaling data for generation {generation}, {member_name}\n"),
            )
            .map_err(|error| {
                format!(
                    "Failed to create mock scaling.data in {}: {}",
                    scaling_dir.display(),
                    error
                )
            })?;

            let estimated_mib = 246.40 + member_index as f64;
            let estimated_bytes = (estimated_mib * 1024.0 * 1024.0).round() as u64;
            let log_text = format!(
                concat!(
                    "*** MEMORY USAGE ESTIMATION ***************************************************\n",
                    "\n",
                    "Estimated memory usage for training (keyword \"memorize_symfunc_results\":\n",
                    "Valid for training of energies and forces.\n",
                    "Memory for local structures  :        12471858 bytes (11.89 MiB = 0.01 GiB).\n",
                    "Memory for all structures    : {estimated_bytes:>15} bytes ({estimated_mib:.2} MiB = {estimated_gib:.2} GiB).\n",
                    "Average memory per structure :        12918531 bytes (12.32 MiB).\n",
                    "*******************************************************************************\n",
                ),
                estimated_bytes = estimated_bytes,
                estimated_mib = estimated_mib,
                estimated_gib = estimated_mib / 1024.0,
            );

            fs::write(scaling_dir.join("log.out"), log_text).map_err(|error| {
                format!(
                    "Failed to create mock scaling log in {}: {}",
                    scaling_dir.display(),
                    error
                )
            })?;
        }

        Ok(())
    }

    pub fn write_n2p2_memory_report(
        generation_dir: &Path,
        training_script: &Path,
        committee_members: usize,
    ) -> Result<(), String> {
        let training_script_text = fs::read_to_string(training_script).map_err(|error| {
            format!(
                "Failed to read n2p2 training script {}: {}",
                training_script.display(),
                error
            )
        })?;
        let requested_memory_mib = parse_slurm_memory_mib(&training_script_text);
        let mut lines = Vec::new();

        lines.push(format!(
            "n2p2 memory check for {}",
            generation_dir.display()
        ));

        match requested_memory_mib {
            Some(memory_mib) => lines.push(format!(
                "training script memory request: {:.2} MiB",
                memory_mib
            )),
            None => lines
                .push("training script memory request: not found (#SBATCH --mem=...)".to_string()),
        }

        let mut checked_logs = 0usize;

        for member_index in 0..committee_members {
            let member_name = format!("member_{member_index:03}");
            let scaling_log = generation_dir
                .join("models")
                .join(&member_name)
                .join("scaling")
                .join("log.out");

            if !scaling_log.is_file() {
                lines.push(format!(
                    "{member_name}: no scaling/log.out found; memory check pending"
                ));
                continue;
            }

            checked_logs += 1;
            let scaling_log_text = fs::read_to_string(&scaling_log).map_err(|error| {
                format!(
                    "Failed to read n2p2 scaling log {}: {}",
                    scaling_log.display(),
                    error
                )
            })?;

            match parse_n2p2_training_memory_mib(&scaling_log_text) {
                Some(estimated_mib) => {
                    let status = match requested_memory_mib {
                        Some(requested_mib) if requested_mib >= estimated_mib => "OK",
                        Some(_) => "INSUFFICIENT",
                        None => "UNKNOWN",
                    };

                    lines.push(format!(
                        "{member_name}: estimated training memory {:.2} MiB -> {status}",
                        estimated_mib
                    ));
                }
                None => lines.push(format!(
                    "{member_name}: scaling/log.out found but memory estimate could not be parsed"
                )),
            }
        }

        if checked_logs == 0 {
            lines.push("no scaling logs were available yet".to_string());
        }

        let report = generation_dir.join("n2p2_memory_check.txt");
        fs::write(&report, format!("{}\n", lines.join("\n"))).map_err(|error| {
            format!(
                "Failed to write n2p2 memory report {}: {}",
                report.display(),
                error
            )
        })?;

        Ok(())
    }

    pub fn create_mock_n2p2_training_outputs(
        project_dir: &Path,
        generation: u32,
        committee_members: usize,
    ) -> Result<(), String> {
        if committee_members == 0 {
            return Err(
                "Cannot create mock n2p2 training outputs for zero committee members.".to_string(),
            );
        }

        let models_dir = project_dir
            .join("training")
            .join(format!("generation_{generation}"))
            .join("models");

        for member_index in 0..committee_members {
            let member_name = format!("member_{member_index:03}");
            let train_dir = models_dir.join(&member_name).join("train");

            if !train_dir.is_dir() {
                return Err(format!(
                    "n2p2 train directory does not exist: {}",
                    train_dir.display()
                ));
            }

            let epoch_1_rmse = 0.180 + member_index as f64 * 0.010;
            let epoch_2_rmse = 0.090 + member_index as f64 * 0.010;

            fs::write(
                train_dir.join("log.out"),
                format!("mock nnp-train completed for generation {generation}, {member_name}\n"),
            )
            .map_err(|error| {
                format!(
                    "Failed to create mock n2p2 training log in {}: {}",
                    train_dir.display(),
                    error
                )
            })?;

            fs::write(
                train_dir.join("learning-curve.out"),
                mock_learning_curve(epoch_1_rmse, epoch_2_rmse),
            )
            .map_err(|error| {
                format!(
                    "Failed to create mock learning-curve.out in {}: {}",
                    train_dir.display(),
                    error
                )
            })?;

            for epoch in [1_u32, 2_u32] {
                for atomic_number in [16_u32, 29_u32] {
                    let weights_name = format!("weights.{atomic_number:03}.{epoch:06}.out");
                    fs::write(
                        train_dir.join(&weights_name),
                        format!(
                            "# mock n2p2 weights for generation {generation}, {member_name}, Z={atomic_number}, epoch={epoch}\n"
                        ),
                    )
                    .map_err(|error| {
                        format!(
                            "Failed to create mock weight file {} in {}: {}",
                            weights_name,
                            train_dir.display(),
                            error
                        )
                    })?;
                }
            }
        }

        Ok(())
    }

    pub fn select_n2p2_best_epoch(
        project_dir: &Path,
        generation: u32,
        committee_members: usize,
    ) -> Result<(), String> {
        if committee_members == 0 {
            return Err("Cannot select n2p2 weights for zero committee members.".to_string());
        }

        let models_dir = project_dir
            .join("training")
            .join(format!("generation_{generation}"))
            .join("models");

        for member_index in 0..committee_members {
            let member_name = format!("member_{member_index:03}");
            let member_dir = models_dir.join(&member_name);
            let train_dir = member_dir.join("train");
            let learning_curve = train_dir.join("learning-curve.out");

            let learning_curve_text = fs::read_to_string(&learning_curve).map_err(|error| {
                format!(
                    "Failed to read n2p2 learning curve {}: {}",
                    learning_curve.display(),
                    error
                )
            })?;

            let (best_epoch, best_rmse) =
                parse_best_n2p2_epoch(&learning_curve_text).ok_or_else(|| {
                    format!(
                        "Could not select best n2p2 epoch from {}",
                        learning_curve.display()
                    )
                })?;

            fs::copy(
                member_dir.join("scaling").join("scaling.data"),
                member_dir.join("scaling.data"),
            )
            .map_err(|error| {
                format!(
                    "Failed to copy selected scaling.data for {}: {}",
                    member_name, error
                )
            })?;

            let mut copied_weights = 0usize;
            let epoch_suffix = format!(".{best_epoch:06}.out");

            for entry in fs::read_dir(&train_dir).map_err(|error| {
                format!(
                    "Failed to read n2p2 train directory {}: {}",
                    train_dir.display(),
                    error
                )
            })? {
                let entry = entry.map_err(|error| {
                    format!(
                        "Failed to inspect n2p2 train directory {}: {}",
                        train_dir.display(),
                        error
                    )
                })?;
                let path = entry.path();

                if !path.is_file() {
                    continue;
                }

                let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
                    continue;
                };

                if !file_name.starts_with("weights.") || !file_name.ends_with(&epoch_suffix) {
                    continue;
                }

                let Some(atomic_number) = file_name.split('.').nth(1) else {
                    continue;
                };
                let target = member_dir.join(format!("weights.{atomic_number}.data"));

                fs::copy(&path, &target).map_err(|error| {
                    format!(
                        "Failed to copy selected n2p2 weights {} to {}: {}",
                        path.display(),
                        target.display(),
                        error
                    )
                })?;
                copied_weights += 1;
            }

            if copied_weights == 0 {
                return Err(format!(
                    "No n2p2 weight files found for selected epoch {best_epoch:06} in {}",
                    train_dir.display()
                ));
            }

            fs::write(
                member_dir.join("selected_epoch.txt"),
                format!("epoch {best_epoch}\nRMSE_Ftest_pu {best_rmse:.8}\n"),
            )
            .map_err(|error| {
                format!(
                    "Failed to write selected epoch summary for {}: {}",
                    member_name, error
                )
            })?;
        }

        Ok(())
    }
}

fn parse_n2p2_training_memory_mib(log_text: &str) -> Option<f64> {
    for line in log_text.lines() {
        if !line.contains("Memory for all structures") {
            continue;
        }

        if let Some(memory_mib) = parse_parenthesized_mib(line) {
            return Some(memory_mib);
        }

        if let Some(bytes) = parse_first_number_before(line, "bytes") {
            return Some(bytes / (1024.0 * 1024.0));
        }
    }

    None
}

fn parse_slurm_memory_mib(script_text: &str) -> Option<f64> {
    for line in script_text.lines() {
        let trimmed = line.trim();

        if !trimmed.starts_with("#SBATCH") {
            continue;
        }

        if let Some(value) = trimmed.strip_prefix("#SBATCH --mem=") {
            return parse_memory_value_mib(value.trim());
        }

        if let Some(value) = trimmed.strip_prefix("#SBATCH --mem ") {
            return parse_memory_value_mib(value.trim());
        }
    }

    None
}

fn parse_parenthesized_mib(line: &str) -> Option<f64> {
    let open = line.find('(')?;
    let close = line[open + 1..].find("MiB")?;
    let mib_text = &line[open + 1..open + 1 + close];

    mib_text
        .split_whitespace()
        .last()
        .and_then(|number| number.parse::<f64>().ok())
}

fn parse_first_number_before(line: &str, marker: &str) -> Option<f64> {
    let before_marker = line.split(marker).next()?;

    before_marker
        .split_whitespace()
        .rev()
        .find_map(|token| token.parse::<f64>().ok())
}

fn parse_memory_value_mib(value: &str) -> Option<f64> {
    let token = value.split_whitespace().next()?;
    let number_end = token
        .find(|character: char| !character.is_ascii_digit() && character != '.')
        .unwrap_or(token.len());

    let number = token[..number_end].parse::<f64>().ok()?;
    let unit = token[number_end..].to_ascii_lowercase();

    match unit.as_str() {
        "" | "m" | "mb" | "mib" => Some(number),
        "k" | "kb" | "kib" => Some(number / 1024.0),
        "g" | "gb" | "gib" => Some(number * 1024.0),
        "t" | "tb" | "tib" => Some(number * 1024.0 * 1024.0),
        _ => None,
    }
}

fn mock_learning_curve(epoch_1_rmse: f64, epoch_2_rmse: f64) -> String {
    format!(
        concat!(
            "################################################################################\n",
            "# Learning curves for energies and forces.\n",
            "################################################################################\n",
            "# Col  Name             Description\n",
            "# 1    epoch            Current epoch.\n",
            "# 11   RMSE_Ftest_pu    RMSE of test forces (physical units)\n",
            "################################################################################\n",
            "#    epoch RMSEpa_Etrain_pu  RMSEpa_Etest_pu   RMSE_Etrain_pu    RMSE_Etest_pu  MAEpa_Etrain_pu   MAEpa_Etest_pu    MAE_Etrain_pu     MAE_Etest_pu   RMSE_Ftrain_pu    RMSE_Ftest_pu    MAE_Ftrain_pu     MAE_Ftest_pu\n",
            "{epoch_1:>10}   1.00000000E-02   2.00000000E-02   3.00000000E-01   4.00000000E-01   1.00000000E-02   2.00000000E-02   3.00000000E-01   4.00000000E-01   1.50000000E-01   {epoch_1_rmse:>14.8E}   1.00000000E-01   1.00000000E-01\n",
            "{epoch_2:>10}   8.00000000E-03   1.00000000E-02   2.00000000E-01   3.00000000E-01   8.00000000E-03   1.00000000E-02   2.00000000E-01   3.00000000E-01   1.00000000E-01   {epoch_2_rmse:>14.8E}   8.00000000E-02   8.00000000E-02\n",
        ),
        epoch_1 = 1,
        epoch_2 = 2,
        epoch_1_rmse = epoch_1_rmse,
        epoch_2_rmse = epoch_2_rmse,
    )
}

fn parse_best_n2p2_epoch(learning_curve_text: &str) -> Option<(u32, f64)> {
    let mut best: Option<(u32, f64)> = None;

    for line in learning_curve_text.lines() {
        let trimmed = line.trim();

        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let columns: Vec<&str> = trimmed.split_whitespace().collect();

        if columns.len() < 11 {
            continue;
        }

        let epoch = columns[0].parse::<u32>().ok()?;
        let rmse_ftest = columns[10].parse::<f64>().ok()?;

        match best {
            Some((_, best_rmse)) if best_rmse <= rmse_ftest => {}
            _ => best = Some((epoch, rmse_ftest)),
        }
    }

    best
}
