use anyhow::{anyhow, Result};
use std::{
    fs,
    path::{Path, PathBuf},
};

use crate::{
    pipeline::{PipelineStep, StepPlan},
    slurm_client,
    types::{FinalJobStatus, FinishedData, JobId, JobScript},
    watcher::wait_for_job,
};

use super::{Pipeline, PipelineCtx};

pub(crate) trait Runner {
    async fn run(&self, pipeline: &Pipeline, pipeline_ctx: &PipelineCtx) -> Result<()> {
        for step in pipeline.steps.iter() {
            self.run_step(step.as_ref(), pipeline_ctx).await?;
        }
        Ok(())
    }

    async fn run_step(&self, pipeline_step: &dyn PipelineStep, pipeline_ctx: &PipelineCtx) -> Result<()>;
}

pub(crate) enum RunnerKind {
    Slurm(SlurmRunner),
    Local(LocalRunner),
    Dry(DryRunner)
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SlurmRunner;

impl Runner for SlurmRunner {
    async fn run_step(&self, step: &dyn PipelineStep, pipeline_ctx: &PipelineCtx) -> Result<()> {
        step.validate_required_files(pipeline_ctx)?;

        let StepPlan::Slurm(job_script) = step.prepare(pipeline_ctx)? else {
            return Err(anyhow!(
                "{} cannot run with SlurmRunner: step does not produce a Slurm job",
                step.name()
            ));
        };

        let mut job_id = slurm_client::submit(&job_script).await?;
        let mut retry_count = 0;

        loop {
            let finished_data = wait_for_job(&job_id, pipeline_ctx.poll_interval).await?;

            match finished_data.final_status {
                FinalJobStatus::Completed => {
                    step.on_completion(&finished_data, pipeline_ctx)?;
                    return Ok(());
                }
                FinalJobStatus::Timeout if retry_count < pipeline_ctx.max_retries => {
                    job_id = step.resubmit_from_current_state()?;
                    retry_count += 1;
                }
                FinalJobStatus::Timeout => {
                    return Err(anyhow!("{} failed: Max retries reached", step.name()));
                }
                FinalJobStatus::OutOfMemory => {
                    return Err(anyhow!("{} failed: Out of memory", step.name()));
                }
                FinalJobStatus::Cancelled => {
                    return Err(anyhow!("{} failed: Job cancelled", step.name()));
                }
                FinalJobStatus::Failed => {
                    return Err(anyhow!("{} failed: Job failed", step.name()));
                }
                FinalJobStatus::Other(e) => {
                    return Err(anyhow!("{} failed: Unknown status: {e}", step.name()));
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct LocalRunner;

impl Runner for LocalRunner {
    async fn run_step(&self, step: &dyn PipelineStep, pipeline_ctx: &PipelineCtx) -> Result<()> {
        //step.validate_required_files(pipeline_ctx)?;

        //match step.prepare(pipeline_ctx)? {
        //    StepPlan::LocalComplete => {
        //        let finished_data =
        //            synthetic_finished_data(JobScript::new(PathBuf::from("<local>")))?;
        //        step.on_completion(&finished_data, pipeline_ctx)
        //    }
        //    StepPlan::Slurm(_) => Err(anyhow!(
        //        "{} cannot run with LocalRunner: step requires Slurm",
        //        step.name()
        //    )),
        //}
        unimplemented!()
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct TestRunner;

impl Runner for TestRunner {
    async fn run_step(
        &self,
        _pipeline_step: &dyn PipelineStep,
        _pipeline_ctx: &PipelineCtx,
    ) -> Result<()> {
        unimplemented!()
    }
}

pub(crate) struct DryRunner {
    _temp_dir: tempfile::TempDir,
    temp_project_dir: PathBuf,
}

impl DryRunner {
    pub(crate) fn new(project_dir: &Path) -> Result<Self> {
        let temp_dir = tempfile::tempdir()?;
        let temp_project_dir = temp_dir.path().join("project");
        copy_project_for_dry_run(project_dir, &temp_project_dir)?;

        println!(
            "[DryRun] using temporary project copy: {}",
            temp_project_dir.display()
        );

        Ok(Self {
            _temp_dir: temp_dir,
            temp_project_dir,
        })
    }
}

impl Runner for DryRunner {
    async fn run(&self, pipeline: &Pipeline, pipeline_ctx: &PipelineCtx) -> Result<()> {
        let dry_ctx = PipelineCtx {
            project_dir: self.temp_project_dir.clone(),
            generation: pipeline_ctx.generation,
            poll_interval: pipeline_ctx.poll_interval,
            max_retries: pipeline_ctx.max_retries,
            dry_run: true,
            dry_config_limit: Some(1),
        };

        for step in pipeline.steps.iter() {
            self.run_step(step.as_ref(), &dry_ctx).await?;
        }

        Ok(())
    }

    async fn run_step(&self, step: &dyn PipelineStep, pipeline_ctx: &PipelineCtx) -> Result<()> {
        println!("[DryRun] preparing {}", step.describe());

        step.validate_required_files(pipeline_ctx)?;
        println!("[DryRun] {} required files ok", step.describe());

        match step.prepare(pipeline_ctx)? {
            StepPlan::LocalComplete => {
                println!("[DryRun] {} completed locally", step.describe());
                let finished_data = synthetic_finished_data(JobScript::new(PathBuf::from("<local>")))?;
                step.on_completion(&finished_data, pipeline_ctx)?;
            }
            StepPlan::Slurm(job_script) => {
                println!("[DryRun] would submit: sbatch {}", job_script.as_path().display());

                let finished_data = synthetic_finished_data(job_script)?;
                step.on_completion(&finished_data, pipeline_ctx)?;
            }
        }

        Ok(())
    }
}

fn synthetic_finished_data(jobscript: JobScript) -> Result<FinishedData> {
    Ok(FinishedData {
        jobscript,
        job_id: JobId::new("0".to_owned())?,
        start_time: String::new(),
        end_time: String::new(),
        runtime: String::new(),
        final_status: FinalJobStatus::Completed,
    })
}

fn copy_project_for_dry_run(source: &Path, target: &Path) -> Result<()> {
    if !source.is_dir() {
        return Err(anyhow!("dry-run project dir is not a directory: {}", source.display()));
    }

    copy_dir_filtered(source, target)
}

fn copy_dir_filtered(source: &Path, target: &Path) -> Result<()> {
    fs::create_dir_all(target)?;

    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let file_name = entry.file_name();
        let file_name_str = file_name.to_string_lossy();

        if matches!(file_name_str.as_ref(), "target" | ".git" | ".jj" | ".direnv") {
            continue;
        }

        let source_path = entry.path();
        let target_path = target.join(&file_name);

        if source_path.is_dir() {
            copy_dir_filtered(&source_path, &target_path)?;
        } else if source_path.is_file() {
            if let Some(parent) = target_path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(&source_path, &target_path)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    struct FakeStep {
        plan: FakePlan,
    }

    enum FakePlan {
        Slurm,
        LocalComplete,
    }

    impl PipelineStep for FakeStep {
        fn name(&self) -> &str {
            "fake"
        }

        fn prepare(&self, pipeline_ctx: &PipelineCtx) -> Result<StepPlan> {
            match self.plan {
                FakePlan::Slurm => Ok(StepPlan::Slurm(JobScript::new(
                    pipeline_ctx.project_dir.join("fake_job.sh"),
                ))),
                FakePlan::LocalComplete => Ok(StepPlan::LocalComplete),
            }
        }

        fn on_completion(
            &self,
            _job_state: &FinishedData,
            _pipeline_ctx: &PipelineCtx,
        ) -> Result<()> {
            Ok(())
        }
    }

    fn test_ctx(project_dir: PathBuf) -> PipelineCtx {
        PipelineCtx {
            project_dir,
            generation: 1,
            poll_interval: Duration::from_secs(1),
            max_retries: 0,
            dry_run: false,
            dry_config_limit: None,
        }
    }

    #[tokio::test]
    async fn slurm_runner_rejects_local_complete_steps() {
        let temp_dir = tempfile::tempdir().unwrap();
        let ctx = test_ctx(temp_dir.path().to_path_buf());
        let step = FakeStep {
            plan: FakePlan::LocalComplete,
        };

        let error = SlurmRunner.run_step(&step, &ctx).await.unwrap_err().to_string();

        assert!(error.contains("cannot run with SlurmRunner"));
    }

    #[tokio::test]
    async fn local_runner_rejects_slurm_steps() {
        let temp_dir = tempfile::tempdir().unwrap();
        let ctx = test_ctx(temp_dir.path().to_path_buf());
        let step = FakeStep {
            plan: FakePlan::Slurm,
        };

        let error = LocalRunner.run_step(&step, &ctx).await.unwrap_err().to_string();

        assert!(error.contains("cannot run with LocalRunner"));
    }
}
