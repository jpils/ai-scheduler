use crate::types::{FinalJobStatus, JobId, JobScript, JobState, PendingData, RunningData, FinishedData};

use std::default;
use std::process::{Command, Output};
use std::path::PathBuf;
use anyhow::{Context, Ok, Result, anyhow};

pub(crate) fn submit(job_script: &JobScript) -> Result<JobId> {
    let sbatch_output = Command::new("sbatch")
        .arg("--parsable")
        .arg(job_script.as_path())
        .output()?;

    if !sbatch_output.status.success() {
        let err = String::from_utf8_lossy(&sbatch_output.stderr).to_string();
        return Err(anyhow!("Failed to submit jobscript with err: {err}"));
    }

    let output = String::from_utf8_lossy(&sbatch_output.stdout).to_string();

    JobId::new(output)
}

pub(crate) fn query_state(job_id: &JobId) -> Result<JobState> {
    let queue_state = get_queue_state(job_id)?;

    match queue_state {
        QueueState::Pending => query_pending(job_id),
        QueueState::Running => query_running(job_id),
        QueueState::NotInQueue => query_finished(job_id),
        QueueState::Other(s) => { return Err(anyhow!("unsupported queue state {s}")) },
        QueueState::Unknown => { return Err(anyhow!("queue state was not polled")) },
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
enum QueueState {
    Pending,
    Running, 
    NotInQueue,
    Other(String),
    #[default]
    Unknown,
}

impl From<&str> for QueueState {
    fn from(value: &str) -> Self {
        match value.trim() {
            "" => QueueState::NotInQueue,
            "PENDING" => QueueState::Pending,
            "RUNNING" => QueueState::Running,
            s => QueueState::Other(s.to_owned())
        }
    }
}

fn query_pending(job_id: &JobId) -> Result<JobState> {
    let query_out = Command::new("squeue")
        .arg("-h")
        .arg("-j")
        .arg(job_id.as_str())
        .arg("-o")
        .arg("%V|%o")
        .output()?;

    if !query_out.status.success() {
        return Err(anyhow!("pending query failed"));
    }

    let query_out = String::from_utf8_lossy(&query_out.stdout);
    let (submit_time, job_script) = query_out
        .trim()
        .split_once('|')
        .ok_or_else(|| anyhow!("could not parse query in new_pending"))?;

    let pending_data = PendingData { 
        jobscript: JobScript::new(job_script.into()), 
        job_id: job_id.to_owned(), 
        submit_time: submit_time.to_owned() 
    };
    
    Ok(JobState::Pending(pending_data))
}

fn query_running(job_id: &JobId) -> Result<JobState> {
    let query_out = Command::new("squeue")
        .arg("-h")
        .arg("-j")
        .arg(job_id.as_str())
        .arg("-o")
        .arg("%V|%o|%N|%M")
        .output()?;

    if !query_out.status.success() {
        return Err(anyhow!("running query failed"));
    }

    let query_out = String::from_utf8_lossy(&query_out.stdout);
    let fields: Vec<_> = query_out
        .trim()
        .split('|')
        .collect();

    let &[submit_time, job_script, node_list, uptime] = fields.as_slice() else {
        return Err(anyhow!("Expected fields: 4, got {}", fields.len()));
    };

    let node_list = node_list.split(',').map(|s| s.to_owned()).collect();
    
    let running_data = RunningData { 
        jobscript: JobScript::new(job_script.into()), 
        job_id: job_id.to_owned(),
        nodes: node_list,
        uptime: uptime.to_owned() 
    };

    Ok(JobState::Running(running_data))
}

fn query_finished(job_id: &JobId) -> Result<JobState> {
    let query_out = Command::new("sacct")
        .arg("-n")
        .arg("-P")
        .arg("-X")
        .arg("-j")
        .arg(job_id.as_str())
        .arg("--format=Start,End,Elapsed,State,SubmitLine")
        .output()?;

    if !query_out.status.success() {
        return Err(anyhow!("finished query failed"));
    }

    let query_out = String::from_utf8_lossy(&query_out.stdout);
    let fields: Vec<_> = query_out
        .trim()
        .splitn(5, '|')
        .collect();

    let &[start, end, elapsed, status, jobscript] = fields.as_slice() else {
        return Err(anyhow!("Expected fields: 5, got {}", fields.len()));
    };

    let final_status: FinalJobStatus = status
        .split_whitespace()
        .next()
        .ok_or_else(|| anyhow!("missing final status"))?
        .parse()?;

    let jobscript = if let Some(jobscript) = jobscript.split_whitespace().last() {
        jobscript   
    } else {
        return Err(anyhow!("Could not parse jobscript from submit line"));
    };

    let finished_data = FinishedData { 
        jobscript: JobScript::new(jobscript.into()), 
        job_id: job_id.to_owned(), 
        start_time: start.to_owned(), 
        end_time: end.to_owned(),
        runtime: elapsed.to_owned(),
        final_status: final_status.to_owned()
    };

    Ok(JobState::Finished(finished_data))
}

fn get_queue_state(job_id: &JobId) -> Result<QueueState> {
    let query_out = Command::new("squeue")
        .arg("-h")
        .arg("-j")
        .arg(job_id.as_str())
        .arg("-o")
        .arg("%T")
        .output()?;

    let query_out_str = String::from_utf8_lossy(&query_out.stdout);

    if query_out_str.trim().is_empty() && query_out.status.success() {
        return Ok(QueueState::NotInQueue);
    } else if !query_out.status.success() {
        return Err(anyhow!("query failed"));
    }

    let queue_state: QueueState = query_out_str.trim().into();

    Ok(queue_state)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_job_id() -> JobId {
        JobId::new("12345".to_owned()).unwrap()
    }

    fn slurm_test_job_id() -> JobId {
        let job_id = std::env::var("SLURM_TEST_JOB_ID")
            .expect("set SLURM_TEST_JOB_ID to run this test");
        JobId::new(job_id).unwrap()
    }

    #[test]
    fn parses_pending_queue_state() {
        assert_eq!(QueueState::from("PENDING"), QueueState::Pending);
        assert_eq!(QueueState::from(" PENDING\n"), QueueState::Pending);
    }

    #[test]
    fn parses_running_queue_state() {
        assert_eq!(QueueState::from("RUNNING"), QueueState::Running);
        assert_eq!(QueueState::from(" RUNNING\n"), QueueState::Running);
    }

    #[test]
    fn empty_queue_state_means_finished() {
        assert_eq!(QueueState::from(""), QueueState::NotInQueue);
        assert_eq!(QueueState::from("\n"), QueueState::NotInQueue);
    }

    #[test]
    fn unsupported_queue_state_is_other() {
        assert_eq!(
            QueueState::from("CONFIGURING"),
            QueueState::Other("CONFIGURING".to_owned())
        );
    }

    #[test]
    fn test_job_id_helper_is_valid() {
        assert_eq!(test_job_id().as_str(), "12345");
    }

    #[test]
    #[ignore = "requires a real pending Slurm job id in SLURM_TEST_JOB_ID"]
    fn queries_pending_job() {
        let job_id = slurm_test_job_id();
        let state = query_pending(&job_id).unwrap();

        assert!(matches!(state, JobState::Pending { .. }));
    }

    #[test]
    #[ignore = "requires a real running Slurm job id in SLURM_TEST_JOB_ID"]
    fn queries_running_job() {
        let job_id = slurm_test_job_id();
        let state = query_running(&job_id).unwrap();

        assert!(matches!(state, JobState::Running { .. }));
    }

    #[test]
    #[ignore = "requires a real finished Slurm job id in SLURM_TEST_JOB_ID"]
    fn queries_finished_job() {
        let job_id = slurm_test_job_id();
        let state = query_finished(&job_id).unwrap();

        assert!(matches!(state, JobState::Finished { .. }));
    }

    #[test]
    #[ignore = "requires a real Slurm job id in SLURM_TEST_JOB_ID"]
    fn queries_job_state() {
        let job_id = slurm_test_job_id();
        let state = query_state(&job_id).unwrap();

        assert!(matches!(
            state,
            JobState::Pending { .. } | JobState::Running { .. } | JobState::Finished { .. }
        ));
    }
}
