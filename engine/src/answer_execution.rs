//! One request clock shared by asynchronous answer stages and model attempts.
use crate::api_types::{
    AnswerRequest, AnswerResponse, ExecutionBudget, RetrievedItem, SemanticAnswerNote,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};
use tokio::time::{Duration, Instant};

fn authority_fingerprint(engine: &crate::MemoryEngine) -> String {
    use std::collections::{BTreeMap, BTreeSet};
    let principals=engine.admin_principals.iter().map(|(id,p)| (id,(&p.principal_id,&p.key_fingerprint,&p.status,&p.principal_type,p.updated_at)))
        .collect::<BTreeMap<_,_>>();
    let roles=engine.admin_role_bindings.iter().map(|(id,r)| (id,(&r.principal_id,&r.role,&r.status,
        r.tenant_ids.iter().collect::<BTreeSet<_>>(),r.workspace_ids.iter().collect::<BTreeSet<_>>(),r.updated_at)))
        .collect::<BTreeMap<_,_>>();
    let policies=engine.control_plane_policy_assignments.iter().map(|(id,p)| (id,
        (&p.tenant_id,&p.workspace_id,&p.policy_key,&p.policy_value,&p.status,p.updated_at))).collect::<BTreeMap<_,_>>();
    let workspaces=engine.workspace_registry.iter().map(|(id,w)| (id,
        (&w.tenant_id,&w.isolation_policy,&w.compliance_profile,&w.lifecycle_status,w.updated_at))).collect::<BTreeMap<_,_>>();
    let conversations=engine.conversation_workspaces.iter().collect::<BTreeMap<_,_>>();
    let visibility=engine.shared_memory_visibility.iter().map(|(id,v)|(id,(&v.agent_id,&v.shared_block))).collect::<BTreeMap<_,_>>();
    crate::shared_object::stable_digest(&serde_json::to_vec(&(principals,roles,policies,workspaces,conversations,visibility))
        .expect("authority snapshot is serializable"))
}

#[derive(Clone)]
struct WorkerBudget {
    cancelled: std::sync::Arc<std::sync::atomic::AtomicBool>,
    deadline: Instant,
}

thread_local! {
    static WORKER_BUDGET: std::cell::RefCell<Option<WorkerBudget>> = const { std::cell::RefCell::new(None) };
}

/// Deadline of the answer worker running on this thread, if any.
pub(crate) fn answer_worker_deadline() -> Option<Instant> {
    WORKER_BUDGET.with(|budget| budget.borrow().as_ref().map(|budget| budget.deadline))
}

pub(crate) fn answer_worker_cancelled() -> bool {
    WORKER_BUDGET.with(|budget| budget.borrow().as_ref().is_some_and(|budget|
        budget.cancelled.load(Ordering::Acquire) || Instant::now() >= budget.deadline))
}

struct WorkerBudgetGuard(Option<WorkerBudget>);
impl Drop for WorkerBudgetGuard {
    fn drop(&mut self) { WORKER_BUDGET.with(|budget| *budget.borrow_mut() = self.0.take()); }
}

/// Cancellation is cooperative inside synchronous work. A partial internal
/// result is discarded at this boundary, never promoted to an answer.
pub(crate) struct BlockingAnswerWorker<T>(
    pub tokio::task::JoinHandle<Result<T, &'static str>>,
    std::sync::Arc<std::sync::atomic::AtomicBool>,
);
impl<T: Send + 'static> BlockingAnswerWorker<T> {
    pub fn spawn(deadline: Instant, work: impl FnOnce() -> T + Send + 'static) -> Self {
        let cancelled = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let budget = WorkerBudget { cancelled: cancelled.clone(), deadline };
        let handle = tokio::task::spawn_blocking(move || {
            let previous = WORKER_BUDGET.with(|slot| slot.replace(Some(budget)));
            let _guard = WorkerBudgetGuard(previous);
            if answer_worker_cancelled() { return Err("answer_worker_cancelled"); }
            let result = work();
            if answer_worker_cancelled() { Err("answer_worker_cancelled") } else { Ok(result) }
        });
        Self(handle, cancelled)
    }
}
impl<T> Drop for BlockingAnswerWorker<T> {
    fn drop(&mut self) {
        self.1.store(true, Ordering::Release);
        self.0.abort();
    }
}

/// Dropping an answer aborts queued blocking work. An already running blocking
/// closure must still cooperate or finish on its isolated engine snapshot.



pub(crate) struct AnswerExecution {
    pub started: Instant,
    pub deadline: Instant,
    pub mode: crate::support::AnswerReasoningMode,
    pub calls: AtomicUsize,
    cancelled: std::sync::atomic::AtomicBool,
    phase: Mutex<&'static str>,
    progress: Mutex<Option<tokio::sync::mpsc::Sender<serde_json::Value>>>,
    checkpoint: Mutex<Option<(AnswerResponse, Vec<RetrievedItem>)>>,
    authority: Mutex<Option<String>>,
}

/// Answer deadline for a reasoning mode. `MEMORY_ANSWER_DEADLINE_MS` is an
/// evaluation-only override (unset in production): it lets a slow provider be
/// measured without changing the product's 20 s / 60 s budgets.
pub(crate) fn answer_deadline_ms(mode: crate::support::AnswerReasoningMode) -> u64 {
    std::env::var("MEMORY_ANSWER_DEADLINE_MS")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .map(|value| value.clamp(1_000, 600_000))
        .unwrap_or_else(|| mode.timeout_ms())
}

impl AnswerExecution {
    pub fn new(mode: crate::support::AnswerReasoningMode, budget: &ExecutionBudget) -> Self {
        let started = Instant::now();
        let milliseconds = answer_deadline_ms(mode)
            .min(budget.timeout_ms.unwrap_or(u64::MAX))
            .min(budget.deadline_ms.unwrap_or(u64::MAX))
            .min(budget.max_compute_ms.unwrap_or(u64::MAX));
        Self {
            started,
            deadline: started + Duration::from_millis(milliseconds),
            mode,
            calls: AtomicUsize::new(0),
            cancelled: std::sync::atomic::AtomicBool::new(false),
            phase: Mutex::new("request_admission"),
            progress: Mutex::new(None),
            checkpoint: Mutex::new(None),
            authority: Mutex::new(None),
        }
    }

    pub fn stage(&self, name: &'static str) {
        if self.is_cancelled() {
            return;
        }
        *self.phase.lock().unwrap() = name;
        if let Some(sender) = self.progress.lock().unwrap().as_ref() {
            let _ = sender.try_send(serde_json::json!({
                "stage":name, "elapsed_ms":self.started.elapsed().as_millis(),
                "mode":self.mode, "calls":self.calls.load(Ordering::Relaxed),
                "deadline_ms":self.deadline.duration_since(self.started).as_millis()
            }));
        }
    }

    pub fn freeze_authority(&self, engine: &crate::MemoryEngine) {
        if self.is_cancelled() {
            return;
        }
        self.authority.lock().unwrap().get_or_insert_with(|| authority_fingerprint(engine));
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    pub fn authority_is_current(&self, engine: &crate::MemoryEngine) -> bool {
        self.authority.lock().unwrap().as_ref().is_none_or(|expected| expected == &authority_fingerprint(engine))
    }

    pub fn subscribe(&self) -> tokio::sync::mpsc::Receiver<serde_json::Value> {
        let (sender, receiver) = tokio::sync::mpsc::channel(32);
        *self.progress.lock().unwrap() = Some(sender);
        self.stage("request_admission");
        receiver
    }

    pub fn save_local(&self, response: &AnswerResponse, sources: Vec<RetrievedItem>) {
        // Never use a model result or an inferred conclusion as deterministic
        // fallback. This checkpoint is written only by the local worker.
        if response
            .claims
            .iter()
            .any(|claim| claim.support_status == "inferred")
        {
            return;
        }
        *self.checkpoint.lock().unwrap() = Some((response.clone(), sources));
    }

    pub fn take_local(&self) -> Option<(AnswerResponse, Vec<RetrievedItem>)> {
        self.checkpoint.lock().unwrap().take()
    }

    pub fn timeout_response(
        &self,
        req: &AnswerRequest,
        local: Option<AnswerResponse>,
    ) -> AnswerResponse {
        let had_local = local.is_some();
        let mut response = local.unwrap_or_else(|| AnswerResponse {
            answer: "The answer deadline expired before a current supported answer was available."
                .into(),
            direct_answer:
                "The answer deadline expired before a current supported answer was available."
                    .into(),
            answer_mode: "unavailable".into(),
            answer_status: "uncertain".into(),
            claims: vec![],
            confidence: "unverified".into(),
            evidence: vec![],
            citations: vec![],
            memories: vec![],
            memory_count: 0,
            coverage: vec![],
            semantic_notes: vec![],
            memory_used: 0,
            memory_ignored: 0,
            conflict_count: 0,
            trace_id: String::new(),
            runtime_enforcement: serde_json::json!({"entrypoint":"answer_execution_deadline"}),
            selection_receipt: None,
            answer_shape: None,
            answer_pipeline: None,
            performance: None,
        });
        if response.answer_status == "answered" {
            response.answer_status = "partial".into();
        }
        response.memory_used = response
            .coverage
            .iter()
            .filter(|claim| claim.covered)
            .flat_map(|claim| claim.evidence_ids.iter().copied())
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        if let Some(receipt) = response.answer_pipeline.as_mut() {
            receipt.failure_class = "answer_deadline_exceeded".into();
            for stage in &mut receipt.stage_trace {
                if stage.stage == "answer" {
                    stage.status = "partial".into();
                    stage.reason = Some(
                        "Request deadline exceeded; current local proof checkpoint retained."
                            .into(),
                    );
                }
            }
        }
        response.semantic_notes.push(SemanticAnswerNote {
            note_type: "reasoning_execution".into(),
            message: serde_json::json!({
                "mode":self.mode, "calls":self.calls.load(Ordering::Relaxed),
                "elapsed_ms":self.started.elapsed().as_millis(),
                "deadline_ms":self.deadline.duration_since(self.started).as_millis(),
                "stage":*self.phase.lock().unwrap(),
                "verification_method":if had_local { "revalidated_local_checkpoint" } else { "none" },
                "degradation_reason":"answer_deadline_exceeded",
                "coverage_gap":"remaining_answer_work_cancelled"
            }).to_string(),
            entity_id:None, attribute_key:None, scope:req.conversation_id.clone(), candidate_values:vec![],
        });
        response
    }
}

impl Drop for AnswerExecution {
    fn drop(&mut self) {
        // Dropping an SSE stream cancels the shared answer future.
        self.cancel();
    }
}




#[cfg(test)]
mod tests {
    use super::*;



    #[tokio::test]
    async fn blocking_deadline_discards_partial_output_and_scope_restores() {
        let mut expired=BlockingAnswerWorker::spawn(Instant::now(),|| panic!("expired work must not start"));
        assert_eq!((&mut expired.0).await.unwrap().unwrap_err(),"answer_worker_cancelled");
        let cancelled=std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let previous=WORKER_BUDGET.with(|slot|slot.replace(Some(WorkerBudget {
            cancelled:cancelled.clone(),deadline:Instant::now()+Duration::from_secs(1),
        })));
        let guard=WorkerBudgetGuard(previous);
        assert!(!answer_worker_cancelled());
        cancelled.store(true,Ordering::Release);
        assert!(answer_worker_cancelled());
        drop(guard);
        assert!(!answer_worker_cancelled());
        let mut worker=BlockingAnswerWorker::spawn(Instant::now()+Duration::from_secs(1),|| {
            WORKER_BUDGET.with(|slot|slot.borrow().as_ref().unwrap().cancelled.store(true,Ordering::Release));
            42
        });
        assert_eq!((&mut worker.0).await.unwrap(),Err("answer_worker_cancelled"));
    }

}
