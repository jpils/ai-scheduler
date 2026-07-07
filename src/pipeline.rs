pub(crate) mod runner;

use crate::{
    disagreement::{DisagreementSettings, DisagreementWorkspace},
    lammps::{LammpsManager, MdModelPackage},
    paths::{pixi_python, scheduler_home},
    slurm_client,
    training::TrainingWorkspace,
    types::{FinalJobStatus, FinishedData, JobId, JobScript},
    vasp::VaspWorkspace,
};
use std::{path::{Path, PathBuf}, process::Command, time::Duration};
use anyhow::{Result, anyhow};

pub(crate) enum StepPlan {
    Slurm(JobScript),
    LocalComplete,
}

pub(crate) enum StepSubmission {
    Slurm(JobId),
    LocalComplete,
}

pub(crate) trait PipelineStep {
    fn name(&self) -> &str;
    fn validate_required_files(&self, pipeline_ctx: &PipelineCtx) -> Result<()> {
        let _ = pipeline_ctx;
        Ok(())
    }

    fn prepare(&self, pipeline_ctx: &PipelineCtx) -> Result<StepPlan>;
    fn submit(&self, pipeline_ctx: &PipelineCtx) -> Result<StepSubmission> {
        match self.prepare(pipeline_ctx)? {
            StepPlan::Slurm(job_script) => slurm_client::submit(&job_script).map(StepSubmission::Slurm),
            StepPlan::LocalComplete => Ok(StepSubmission::LocalComplete),
        }
    }
    fn resubmit_from_current_state(&self) -> Result<JobId> {
        Err(anyhow!("{} does not support resubmission", self.name()))
    }
    fn on_completion(&self, job_state: &FinishedData, pipeline_ctx: &PipelineCtx) -> Result<()>;
    fn describe(&self) -> String {
        self.name().to_owned()
    }
}

pub(crate) struct Pipeline {
    steps: Vec<Box<dyn PipelineStep>>
}

impl Pipeline {
    pub(crate) fn new(steps: Vec<Box<dyn PipelineStep>>) -> Self {
        Self { steps }
    }
}

pub(crate) struct PipelineCtx {
    pub(crate) project_dir: PathBuf,
    pub(crate) generation: u32,
    pub(crate) poll_interval: Duration,
    pub(crate) max_retries: u32,
    pub(crate) dry_run: bool,
    pub(crate) dry_config_limit: Option<usize>,
}

pub(crate) struct StepCtx {
    pub(crate) working_dir: PathBuf,
    pub(crate) setup_dir: PathBuf,
    pub(crate) template: PathBuf,
}

impl StepCtx {
    pub(crate) fn new(working_dir: PathBuf, setup_dir: PathBuf, template: PathBuf) -> Self {
        Self { working_dir, setup_dir, template }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum MdEngine {
    Lammps,
}

impl MdEngine {
    fn as_str(&self) -> &str {
        match self {
            Self::Lammps => "lammps"
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum DftCode {
    Vasp,
}

impl DftCode {
    fn as_str(&self) -> &str {
        match self {
            Self::Vasp => "vasp"
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum ModelBackend {
    Upet,
    N2p2
}

impl ModelBackend {
    fn as_str(&self) -> &str {
        match self {
            Self::Upet => "UPET",
            Self::N2p2 => "n2p2"
        }
    }

    fn backend_key(&self) -> &'static str {
        match self {
            Self::Upet => "upet",
            Self::N2p2 => "n2p2",
        }
    }

    fn pixi_env(&self) -> &'static str {
        self.backend_key()
    }

    fn python_script(&self) -> &'static str {
        match self {
            Self::Upet => "poscar_to_upet.py",
            Self::N2p2 => "poscar_to_n2p2.py",
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum EnergyMode {
    Pet,
    Raw,
}

impl EnergyMode {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Pet => "pet",
            Self::Raw => "raw",
        }
    }

    fn training_key(&self) -> &'static str {
        match self {
            Self::Pet => "energy-corrected",
            Self::Raw => "energy",
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum QbcMethod {
    Rrmsfd,
}

impl QbcMethod {
    fn as_str(&self) -> &str {
        match self {
            Self::Rrmsfd => "RRMSFD (relative root mean squared force deviation)",
        }
    }
}

pub(crate) struct MdStep {
    engine: MdEngine,
    ctx: StepCtx,
    committee_members: usize,
    model_package: Option<MdModelPackage>,
}

impl MdStep {
    pub(crate) fn new(
        engine: MdEngine,
        ctx: StepCtx,
        committee_members: usize,
        model_package: Option<MdModelPackage>,
    ) -> Self {
        Self { engine, ctx, committee_members, model_package }
    }
}

impl PipelineStep for MdStep {
    fn name(&self) -> &str {
        self.engine.as_str()
    }

    fn validate_required_files(&self, pipeline_ctx: &PipelineCtx) -> Result<()> {
        if self.committee_members == 0 {
            return Err(anyhow!("MD step requires at least one committee member"));
        }

        LammpsManager::find_input_file(&self.ctx.setup_dir, pipeline_ctx.generation)
            .map_err(|error| anyhow!(error))?;

        let data_file = self.ctx.setup_dir.join("lammps").join("data").join("lmp.data");
        if !data_file.is_file() {
            return Err(anyhow!("missing LAMMPS data file: {}", data_file.display()));
        }

        let template = self.ctx.setup_dir.join("jobscripts").join("md_array.sh.template");
        if !template.is_file() {
            return Err(anyhow!("missing MD job template: {}", template.display()));
        }

        Ok(())
    }

    fn prepare(&self, pipeline_ctx: &PipelineCtx) -> Result<StepPlan> {
        let generation_dir = LammpsManager::create_generation_workspace(
            &pipeline_ctx.project_dir,
            &self.ctx.setup_dir,
            pipeline_ctx.generation,
            self.committee_members,
            self.model_package,
        ).map_err(|error| anyhow!(error))?;

        Ok(StepPlan::Slurm(JobScript::new(generation_dir.join("submit_array.sh"))))
    }

    fn on_completion(&self, job_state: &FinishedData, _pipeline_ctx: &PipelineCtx) -> Result<()> {
        ensure_completed(job_state)
    }
}

pub(crate) struct DftStep {
    dft_code: DftCode,
    ctx: StepCtx
}

impl DftStep {
    pub(crate) fn new(dft_code: DftCode, ctx: StepCtx) -> Self {
        Self { dft_code, ctx }
    }
}

impl PipelineStep for DftStep {
    fn name(&self) -> &str {
        self.dft_code.as_str()
    }

    fn validate_required_files(&self, pipeline_ctx: &PipelineCtx) -> Result<()> {
        for input in ["INCAR", "KPOINTS", "POTCAR"] {
            let path = self.ctx.setup_dir.join("vasp").join(input);
            if !path.is_file() {
                return Err(anyhow!("missing VASP input: {}", path.display()));
            }
        }

        let template = self.ctx.setup_dir.join("jobscripts").join("vasp_array.sh.template");
        if !template.is_file() {
            return Err(anyhow!("missing VASP job template: {}", template.display()));
        }

        let selected = pipeline_ctx.project_dir
            .join("selected_structures")
            .join(format!("generation_{}.xyz", pipeline_ctx.generation));

        if !selected.is_file() && dry_seed_dataset(&pipeline_ctx.project_dir).is_err() {
            return Err(anyhow!(
                "missing selected structures and seed dataset for DFT dry-run: {}",
                selected.display()
            ));
        }

        Ok(())
    }

    fn prepare(&self, pipeline_ctx: &PipelineCtx) -> Result<StepPlan> {
        let selected_structures = pipeline_ctx.project_dir
            .join("selected_structures")
            .join(format!("generation_{}.xyz", pipeline_ctx.generation));

        let selected_structures = if selected_structures.is_file() {
            selected_structures
        } else if pipeline_ctx.dry_run {
            dry_seed_dataset(&pipeline_ctx.project_dir)?
        } else {
            return Err(anyhow!(
                "selected structures file not found: {}",
                selected_structures.display()
            ));
        };

        let generation_dir = pipeline_ctx.project_dir
            .join("vasp_runs")
            .join(format!("generation_{}", pipeline_ctx.generation));

        let count = VaspWorkspace::get_configuration_count(&selected_structures)?;
        let count = pipeline_ctx
            .dry_config_limit
            .map_or(count, |limit| count.min(limit));

        for config_index in 0..count {
            let run_name = format!("config_{config_index:03}");
            let run_dir = generation_dir.join(&run_name);

            VaspWorkspace::create_run_directory(
                &run_name,
                &selected_structures,
                &generation_dir,
                &self.ctx.setup_dir,
                config_index,
            )?;

            VaspWorkspace::create_mock_outcar(&run_dir, config_index)?;
        }

        let job_script = VaspWorkspace::create_array_script(
            &self.ctx.setup_dir,
            &generation_dir,
            pipeline_ctx.generation,
            count,
        )?;

        Ok(StepPlan::Slurm(JobScript::new(job_script)))
    }

    fn on_completion(&self, job_state: &FinishedData, _pipeline_ctx: &PipelineCtx) -> Result<()> {
        ensure_completed(job_state)
    }
}

pub(crate) struct TrainingStep {
    model_backend: ModelBackend,
    ctx: StepCtx,
    committee_members: usize,
    checkpoint: Option<PathBuf>,
    energy_mode: EnergyMode,
}

impl TrainingStep {
    pub(crate) fn new(
        model_backend: ModelBackend,
        ctx: StepCtx,
        committee_members: usize,
        checkpoint: Option<PathBuf>,
        energy_mode: EnergyMode,
    ) -> Self {
        Self { model_backend, ctx, committee_members, checkpoint, energy_mode }
    }
}

impl PipelineStep for TrainingStep {
    fn name(&self) -> &str {
        self.model_backend.as_str()
    }

    fn validate_required_files(&self, _pipeline_ctx: &PipelineCtx) -> Result<()> {
        if self.committee_members == 0 {
            return Err(anyhow!("training step requires at least one committee member"));
        }

        match self.model_backend {
            ModelBackend::Upet => {
                let checkpoint = self.checkpoint.as_deref()
                    .ok_or_else(|| anyhow!("UPET training requires a checkpoint"))?;
                if !checkpoint.is_file() {
                    return Err(anyhow!("checkpoint does not exist: {}", checkpoint.display()));
                }

                for path in [
                    self.ctx.setup_dir.join("training").join("upet.yaml.template"),
                    self.ctx.setup_dir.join("jobscripts").join("upet_training_array.sh.template"),
                ] {
                    if !path.is_file() {
                        return Err(anyhow!("missing UPET training config: {}", path.display()));
                    }
                }
            }
            ModelBackend::N2p2 => {
                for path in [
                    self.ctx.setup_dir.join("training").join("input.nn"),
                    self.ctx.setup_dir.join("jobscripts").join("n2p2_scaling_array.sh.template"),
                    self.ctx.setup_dir.join("jobscripts").join("n2p2_training_array.sh.template"),
                ] {
                    if !path.is_file() {
                        return Err(anyhow!("missing n2p2 training config: {}", path.display()));
                    }
                }
            }
        }

        Ok(())
    }

    fn prepare(&self, pipeline_ctx: &PipelineCtx) -> Result<StepPlan> {
        if pipeline_ctx.dry_run {
            prepare_dry_training_dataset(
                &pipeline_ctx.project_dir,
                pipeline_ctx.generation,
                &self.model_backend,
                self.checkpoint.as_deref(),
            )?;
        } else {
            prepare_training_dataset(
                &pipeline_ctx.project_dir,
                pipeline_ctx.generation,
                &self.model_backend,
                self.checkpoint.as_deref(),
                &self.energy_mode,
            ).map_err(|error| anyhow!(error))?;
        }

        let job_script = match self.model_backend {
            ModelBackend::Upet => {
                let checkpoint = self.checkpoint.as_deref().ok_or_else(|| {
                    anyhow!("UPET training requires a checkpoint")
                })?;

                TrainingWorkspace::create_upet_workspace(
                    &pipeline_ctx.project_dir,
                    &self.ctx.setup_dir,
                    pipeline_ctx.generation,
                    self.committee_members,
                    checkpoint,
                    self.energy_mode.training_key(),
                ).map_err(|error| anyhow!(error))?
            }

            ModelBackend::N2p2 => {
                let (scaling_script, training_script) = TrainingWorkspace::create_n2p2_workspace(
                    &pipeline_ctx.project_dir,
                    &self.ctx.setup_dir,
                    pipeline_ctx.generation,
                    self.committee_members,
                ).map_err(|error| anyhow!(error))?;

                TrainingWorkspace::create_mock_n2p2_scaling_outputs(
                    &pipeline_ctx.project_dir,
                    pipeline_ctx.generation,
                    self.committee_members,
                ).map_err(|error| anyhow!(error))?;

                let generation_dir = pipeline_ctx.project_dir
                    .join("training")
                    .join(format!("generation_{}", pipeline_ctx.generation));

                TrainingWorkspace::write_n2p2_memory_report(
                    &generation_dir,
                    &training_script,
                    self.committee_members,
                ).map_err(|error| anyhow!(error))?;

                training_script
            }
        };

        Ok(StepPlan::Slurm(JobScript::new(job_script)))
    }

    fn on_completion(&self, job_state: &FinishedData, pipeline_ctx: &PipelineCtx) -> Result<()> {
        ensure_completed(job_state)?;

        match self.model_backend {
            ModelBackend::Upet => TrainingWorkspace::create_mock_upet_models(
                &pipeline_ctx.project_dir,
                pipeline_ctx.generation,
                self.committee_members,
            ).map_err(|error| anyhow!(error)),

            ModelBackend::N2p2 => {
                TrainingWorkspace::create_mock_n2p2_training_outputs(
                    &pipeline_ctx.project_dir,
                    pipeline_ctx.generation,
                    self.committee_members,
                ).map_err(|error| anyhow!(error))?;

                TrainingWorkspace::select_n2p2_best_epoch(
                    &pipeline_ctx.project_dir,
                    pipeline_ctx.generation,
                    self.committee_members,
                ).map_err(|error| anyhow!(error))
            }
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum DisagreementMode {
    Mock,
    Real,
}

pub(crate) struct QbcStep {
    method: QbcMethod,
    ctx: StepCtx,
    backend: ModelBackend,
    committee_members: usize,
    settings: DisagreementSettings,
    mode: DisagreementMode,
}

impl QbcStep {
    pub(crate) fn new(
        method: QbcMethod,
        ctx: StepCtx,
        backend: ModelBackend,
        committee_members: usize,
        settings: DisagreementSettings,
        mode: DisagreementMode,
    ) -> Self {
        Self { method, ctx, backend, committee_members, settings, mode }
    }
}

impl PipelineStep for QbcStep {
    fn name(&self) -> &str {
        self.method.as_str()
    }

    fn validate_required_files(&self, _pipeline_ctx: &PipelineCtx) -> Result<()> {
        if self.committee_members < 2 {
            return Err(anyhow!("QBC step requires at least two committee members"));
        }

        if matches!(self.mode, DisagreementMode::Real) {
            let template = self.ctx.setup_dir
                .join("jobscripts")
                .join(format!("{}_disagreement.sh.template", self.backend.backend_key()));

            if !template.is_file() {
                return Err(anyhow!("missing disagreement job template: {}", template.display()));
            }
        }

        Ok(())
    }

    fn prepare(&self, pipeline_ctx: &PipelineCtx) -> Result<StepPlan> {
        match self.mode {
            DisagreementMode::Mock => {
                DisagreementWorkspace::evaluate_mock(
                    &pipeline_ctx.project_dir,
                    pipeline_ctx.generation,
                    self.backend.backend_key(),
                    self.committee_members,
                    self.settings,
                ).map_err(|error| anyhow!(error))?;

                Ok(StepPlan::LocalComplete)
            }

            DisagreementMode::Real => {
                let job_script = DisagreementWorkspace::create_real_job_script(
                    &pipeline_ctx.project_dir,
                    &self.ctx.setup_dir,
                    pipeline_ctx.generation,
                    self.backend.backend_key(),
                    self.settings,
                ).map_err(|error| anyhow!(error))?;

                Ok(StepPlan::Slurm(JobScript::new(job_script)))
            }
        }
    }

    fn on_completion(&self, job_state: &FinishedData, _pipeline_ctx: &PipelineCtx) -> Result<()> {
        ensure_completed(job_state)
    }
}

fn prepare_dry_training_dataset(
    project_dir: &Path,
    generation: u32,
    backend: &ModelBackend,
    checkpoint_file: Option<&Path>,
) -> Result<()> {
    if matches!(backend, ModelBackend::Upet) {
        let checkpoint = checkpoint_file
            .ok_or_else(|| anyhow!("UPET dry-run requires a checkpoint"))?;

        if !checkpoint.is_file() {
            return Err(anyhow!("checkpoint does not exist: {}", checkpoint.display()));
        }
    }

    let dataset_dir = project_dir
        .join("training")
        .join(format!("generation_{generation}"))
        .join("dataset");

    std::fs::create_dir_all(&dataset_dir)?;

    match backend {
        ModelBackend::Upet => {
            for file_name in ["train.extxyz", "validation.extxyz", "test.extxyz"] {
                let path = dataset_dir.join(file_name);
                if !path.is_file() {
                    std::fs::write(&path, "# dry-run placeholder dataset\n")?;
                }
            }
        }
        ModelBackend::N2p2 => {
            let path = dataset_dir.join("input.data");
            if !path.is_file() {
                std::fs::write(&path, "# dry-run placeholder n2p2 dataset\n")?;
            }
        }
    }

    Ok(())
}

fn dry_seed_dataset(project_dir: &Path) -> Result<PathBuf> {
    let extxyz = project_dir.join("setup").join("training").join("seed_dataset.extxyz");
    let xyz = project_dir.join("setup").join("training").join("seed_dataset.xyz");

    if extxyz.is_file() {
        Ok(extxyz)
    } else if xyz.is_file() {
        Ok(xyz)
    } else {
        Err(anyhow!(
            "dry-run selected structures missing and no seed dataset found; expected {} or {}",
            extxyz.display(),
            xyz.display()
        ))
    }
}

fn prepare_training_dataset(
    project_dir: &Path,
    generation: u32,
    backend: &ModelBackend,
    checkpoint_file: Option<&Path>,
    energy_mode: &EnergyMode,
) -> std::result::Result<(), String> {
    let scheduler_dir = scheduler_home().map_err(|error| error.to_string())?;
    let python_script_path = scheduler_dir.join("python").join(backend.python_script());
    let checkpoint_arg = checkpoint_file.unwrap_or_else(|| Path::new(""));

    let status = pixi_python(backend.pixi_env())
        .map_err(|error| format!("Failed to configure Pixi: {}", error))?
        .arg(&python_script_path)
        .arg(project_dir)
        .arg(generation.to_string())
        .arg(checkpoint_arg)
        .arg(energy_mode.as_str())
        .status()
        .map_err(|error| format!("Failed to spawn companion dataset engine: {}", error))?;

    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "Python dataset pipeline returned non-zero exit status: {}",
            status
        ))
    }
}

fn ensure_completed(job_state: &FinishedData) -> Result<()> {
    match job_state.final_status {
        FinalJobStatus::Completed => Ok(()),
        ref status => Err(anyhow!("job finished with non-completed status: {status:?}")),
    }
}

fn parse_command() -> Result<Command> {
    todo!()
}
