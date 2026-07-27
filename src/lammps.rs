use crate::job_template::render_job_template;
use std::fs;
use std::path::{Path, PathBuf};

pub struct LammpsManager;

#[derive(Clone, Copy)]
pub enum MdModelPackage {
    UpetModel,
    N2p2Inputs,
}

impl LammpsManager {
    /// Find the LAMMPS input file for a generation.
    ///
    /// Rules:
    /// - If exactly one `.in` file exists, use it for every generation.
    /// - If multiple `.in` files exist, require `gen_<generation>.in`.
    pub fn find_input_file(setup_dir: &Path, gen_num: u32) -> Result<PathBuf, String> {
        let in_dir = setup_dir.join("lammps").join("in");

        if !in_dir.is_dir() {
            return Err(format!(
                "The LAMMPS input folder is missing: {}",
                in_dir.display()
            ));
        }

        let mut in_files = Vec::new();

        let entries = fs::read_dir(&in_dir).map_err(|e| {
            format!(
                "Failed to read LAMMPS input directory {}: {}",
                in_dir.display(),
                e
            )
        })?;

        for entry in entries {
            let entry = entry.map_err(|e| {
                format!("Failed to inspect an entry in {}: {}", in_dir.display(), e)
            })?;

            let path = entry.path();

            if path.is_file() && path.extension().is_some_and(|extension| extension == "in") {
                in_files.push(path);
            }
        }

        in_files.sort();

        if in_files.is_empty() {
            return Err(format!(
                "No files ending in '.in' were found inside {}",
                in_dir.display()
            ));
        }

        if in_files.len() == 1 {
            return Ok(in_files[0].clone());
        }

        let expected_filename = format!("gen_{}.in", gen_num);
        let specific_path = in_dir.join(&expected_filename);

        if specific_path.is_file() {
            Ok(specific_path)
        } else {
            Err(format!(
                "Multiple LAMMPS input files were found, but '{}' is missing",
                expected_filename
            ))
        }
    }

    /// Create the MD committee workspace for one active-learning generation.
    ///
    /// One trajectory-producing MD run is created per generation.
    pub fn create_generation_workspace(
        project_dir: &Path,
        setup_dir: &Path,
        gen_num: u32,
        committee_members: usize,
        model_package: Option<MdModelPackage>,
    ) -> Result<PathBuf, String> {
        if committee_members == 0 {
            return Err("The committee must contain at least one member.".to_string());
        }

        let input_file = Self::find_input_file(setup_dir, gen_num)?;
        let data_file = setup_dir.join("lammps").join("data").join("lmp.data");

        if !data_file.is_file() {
            return Err(format!(
                "The LAMMPS data file is missing: {}",
                data_file.display()
            ));
        }

        let generation_dir = project_dir
            .join("md_runs")
            .join(format!("generation_{}", gen_num));

        fs::create_dir_all(&generation_dir).map_err(|e| {
            format!(
                "Failed to create MD generation directory {}: {}",
                generation_dir.display(),
                e
            )
        })?;

        let run_dir = generation_dir.join("run_000");

        fs::create_dir_all(&run_dir).map_err(|e| {
            format!(
                "Failed to create MD run directory {}: {}",
                run_dir.display(),
                e
            )
        })?;

        fs::copy(&input_file, run_dir.join("input.lmp")).map_err(|e| {
            format!(
                "Failed to copy {} into {}: {}",
                input_file.display(),
                run_dir.display(),
                e
            )
        })?;

        fs::copy(&data_file, run_dir.join("lmp.data")).map_err(|e| {
            format!(
                "Failed to copy {} into {}: {}",
                data_file.display(),
                run_dir.display(),
                e
            )
        })?;

        let committee_models_dir = generation_dir.join("committee_models");

        fs::create_dir_all(&committee_models_dir).map_err(|e| {
            format!(
                "Failed to create committee model package directory {}: {}",
                committee_models_dir.display(),
                e
            )
        })?;

        for member_index in 0..committee_members {
            let member_name = format!("member_{:03}", member_index);
            let model_dir = project_dir
                .join("training")
                .join(format!("generation_{}", gen_num))
                .join("models")
                .join(&member_name);
            let member_package_dir = committee_models_dir.join(&member_name);

            match model_package {
                Some(MdModelPackage::UpetModel) => {
                    fs::create_dir_all(&member_package_dir).map_err(|e| {
                        format!(
                            "Failed to create UPET model package directory {}: {}",
                            member_package_dir.display(),
                            e
                        )
                    })?;

                    let real_model = model_dir.join("model.pt");
                    let mock_model = model_dir.join("mock_trained_model.pt");
                    let source_model = if real_model.is_file() {
                        real_model
                    } else if mock_model.is_file() {
                        mock_model
                    } else {
                        return Err(format!(
                            "Required UPET model is missing. Expected one of: {}, {}",
                            real_model.display(),
                            mock_model.display()
                        ));
                    };

                    let stale_mock_package = member_package_dir.join("mock_trained_model.pt");
                    if stale_mock_package.is_file() {
                        fs::remove_file(&stale_mock_package).map_err(|e| {
                            format!(
                                "Failed to remove stale UPET model package {}: {}",
                                stale_mock_package.display(),
                                e
                            )
                        })?;
                    }

                    fs::copy(&source_model, member_package_dir.join("model.pt")).map_err(|e| {
                        format!(
                            "Failed to copy {} into {}: {}",
                            source_model.display(),
                            member_package_dir.display(),
                            e
                        )
                    })?;
                }

                Some(MdModelPackage::N2p2Inputs) => {
                    fs::create_dir_all(&member_package_dir).map_err(|e| {
                        format!(
                            "Failed to create n2p2 model package directory {}: {}",
                            member_package_dir.display(),
                            e
                        )
                    })?;

                    for file_name in ["input.data", "input.nn", "scaling.data"] {
                        let source = model_dir.join(file_name);

                        if !source.is_file() {
                            return Err(format!(
                                "Required n2p2 MD package file is missing: {}",
                                source.display()
                            ));
                        }

                        fs::copy(&source, member_package_dir.join(file_name)).map_err(|e| {
                            format!(
                                "Failed to copy {} into {}: {}",
                                source.display(),
                                member_package_dir.display(),
                                e
                            )
                        })?;
                    }

                    let mut copied_weights = 0usize;

                    for entry in fs::read_dir(&model_dir).map_err(|e| {
                        format!(
                            "Failed to read n2p2 model directory {}: {}",
                            model_dir.display(),
                            e
                        )
                    })? {
                        let entry = entry.map_err(|e| {
                            format!(
                                "Failed to inspect n2p2 model directory {}: {}",
                                model_dir.display(),
                                e
                            )
                        })?;
                        let path = entry.path();

                        if !path.is_file() {
                            continue;
                        }

                        let Some(file_name) = path.file_name().and_then(|name| name.to_str())
                        else {
                            continue;
                        };

                        if file_name.starts_with("weights.") && file_name.ends_with(".data") {
                            fs::copy(&path, member_package_dir.join(file_name)).map_err(|e| {
                                format!(
                                    "Failed to copy {} into {}: {}",
                                    path.display(),
                                    member_package_dir.display(),
                                    e
                                )
                            })?;
                            copied_weights += 1;
                        }
                    }

                    if copied_weights == 0 {
                        return Err(format!(
                            "No selected n2p2 weight files found in {}",
                            model_dir.display()
                        ));
                    }

                    if member_index == 0 {
                        let driver_dir = run_dir.join("n2p2");

                        fs::create_dir_all(&driver_dir).map_err(|e| {
                            format!(
                                "Failed to create n2p2 MD driver directory {}: {}",
                                driver_dir.display(),
                                e
                            )
                        })?;

                        for file_name in ["input.data", "input.nn", "scaling.data"] {
                            fs::copy(
                                member_package_dir.join(file_name),
                                driver_dir.join(file_name),
                            )
                            .map_err(|e| {
                                format!(
                                    "Failed to copy {} into {}: {}",
                                    member_package_dir.join(file_name).display(),
                                    driver_dir.display(),
                                    e
                                )
                            })?;
                        }

                        for entry in fs::read_dir(&member_package_dir).map_err(|e| {
                            format!(
                                "Failed to read n2p2 member package directory {}: {}",
                                member_package_dir.display(),
                                e
                            )
                        })? {
                            let entry = entry.map_err(|e| {
                                format!(
                                    "Failed to inspect n2p2 member package directory {}: {}",
                                    member_package_dir.display(),
                                    e
                                )
                            })?;
                            let path = entry.path();

                            if !path.is_file() {
                                continue;
                            }

                            let Some(file_name) = path.file_name().and_then(|name| name.to_str())
                            else {
                                continue;
                            };

                            if file_name.starts_with("weights.") && file_name.ends_with(".data") {
                                fs::copy(&path, driver_dir.join(file_name)).map_err(|e| {
                                    format!(
                                        "Failed to copy {} into {}: {}",
                                        path.display(),
                                        driver_dir.display(),
                                        e
                                    )
                                })?;
                            }
                        }
                    }
                }

                None => {}
            }
        }

        Self::create_md_array_script(setup_dir, &generation_dir, gen_num, 1)?;

        Ok(generation_dir)
    }

    /// Create one Slurm array task per MD run.
    pub fn create_md_array_script(
        setup_dir: &Path,
        generation_dir: &Path,
        gen_num: u32,
        committee_members: usize,
    ) -> Result<PathBuf, String> {
        if committee_members == 0 {
            return Err("Cannot create an MD array for zero committee members.".to_string());
        }

        let template_path = setup_dir.join("jobscripts").join("md_array.sh.template");

        if !template_path.is_file() {
            return Err(format!(
                "Missing MD job template: {}",
                template_path.display()
            ));
        }

        let max_index = committee_members - 1;
        let script_path = generation_dir.join("submit_array.sh");

        render_job_template(&template_path, &script_path, gen_num, max_index).map_err(|error| {
            format!(
                "Failed to render MD job template {}: {}",
                template_path.display(),
                error
            )
        })?;

        Ok(script_path)
    }
}
