use std::path::{PathBuf, Path};
use anyhow::{Context, Ok, Result, anyhow};
use strum_macros::EnumString;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct JobId(String);

impl JobId {
    pub(crate) fn new(string: String) -> Result<Self> {
        let jobid = string
            .trim()
            .split(';')
            .next()
            .ok_or_else(|| anyhow!("Invalid job id"))?
            .trim();

        if jobid.is_empty() { 
            return Err(anyhow!("job id is empty")) 
        }

        jobid.parse::<u64>()
            .context("Parse error: job id invalid format")?;

        Ok(Self(jobid.into()))
    }

    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PendingData {
    pub(crate) jobscript: JobScript,
    pub(crate) job_id: JobId,
    pub(crate) submit_time: String,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RunningData {
    pub(crate) jobscript: JobScript,
    pub(crate) job_id: JobId,
    pub(crate) nodes: Vec<String>,
    pub(crate) uptime: String,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FinishedData {
    pub(crate) jobscript: JobScript,
    pub(crate) job_id: JobId,
    pub(crate) start_time: String,
    pub(crate) end_time: String,
    pub(crate) runtime: String,
    pub(crate) final_status: FinalJobStatus,
}


#[derive(Debug, Clone, PartialEq)]
pub(crate) enum JobState {
    Pending(PendingData),
    Running(RunningData),
    Finished(FinishedData),
    Other(String)
}

#[derive(Debug, EnumString, Clone, PartialEq)]
#[strum(serialize_all = "SCREAMING_SNAKE_CASE")]
pub(crate) enum FinalJobStatus {
    Completed,
    Cancelled,
    Timeout,
    OutOfMemory,
    Failed,
    #[strum(default)]
    Other(String)
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct JobScript(PathBuf);

impl JobScript {
    pub(crate) fn new(path: PathBuf) -> Self {
        JobScript(path)
    }

    pub(crate) fn as_path(&self) -> &Path {
        &self.0
    }
}
