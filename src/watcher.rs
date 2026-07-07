use std::{thread::sleep, time::Duration};

use crate::types::{JobId, FinishedData, JobState};
use anyhow::{Result};
use crate::slurm_client;

pub(crate) fn wait_for_job(job_id: &JobId, poll_interval: Duration) -> Result<FinishedData> {
    loop {
        let job_state = slurm_client::query_state(job_id)?;

        match job_state {
            JobState::Finished(data) => return Ok(data),
            _ => sleep(poll_interval)
        }
    } 
}

pub fn is_terminal(job_state: &JobState) -> bool {
    matches!(job_state, JobState::Finished(_))
}
