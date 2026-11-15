//! Handlers for the prompt hub, feedback, annotation queues, run search,
//! metrics, pricing, settings, rules and alerts.

use axum::{
    body::Bytes,
    extract::{Path, Query, State},
    http::Uri,
    Json,
};
use serde::{de::DeserializeOwned, Deserialize};
use serde_json::{json, Value};
use sinew_eval::{Example, PairwiseReport};

use crate::{
    pricing::Pricing,
    server::{parse_json, ApiError, ApiResult, AppState},
    store::{
        AlertInput, FeedbackInput, MetricsQuery, PromptPushInput, RuleInput, RunQuery, Settings,
    },
};

/// Parses the query string of `uri` into `T` (unknown keys are ignored).
pub(crate) fn query_from<T: DeserializeOwned>(uri: &Uri) -> Result<T, ApiError> {
    Query::<T>::try_from_uri(uri)
        .map(|q| q.0)
        .map_err(|e| ApiError::bad_request(format!("invalid query: {e}")))
}

/// `meta.<key>=<value>` query parameters.
pub(crate) fn meta_filters(uri: &Uri) -> Vec<(String, String)> {
    Query::<Vec<(String, String)>>::try_from_uri(uri)
        .map(|q| q.0)
        .unwrap_or_default()
        .into_iter()
        .filter_map(|(k, v)| {
            k.strip_prefix("meta.")
                .filter(|k| !k.is_empty())
                .map(|k| (k.to_string(), v))
        })
        .collect()
}

fn body<T: DeserializeOwned>(b: &Bytes, what: &str) -> Result<T, ApiError> {
    serde_json::from_value(parse_json(b)?)
        .map_err(|e| ApiError::bad_request(format!("invalid {what}: {e}")))
}

fn found<T>(v: Option<T>, what: &str) -> Result<T, ApiError> {
    v.ok_or_else(|| ApiError::not_found(what))
}

fn deleted(ok: bool, what: &str) -> ApiResult<Value> {
    if ok {
        Ok(Json(json!({ "deleted": true })))
    } else {
        Err(ApiError::not_found(what))
    }
}

fn engine(state: &AppState) -> Result<&crate::engine::Engine, ApiError> {
    state.engine.as_ref().ok_or_else(|| {
        ApiError(
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "the background engine is not running".into(),
        )
    })
}

// ---------- runs, metrics ----------

pub(crate) async fn search_runs(State(state): State<AppState>, uri: Uri) -> ApiResult<Value> {
    let mut q: RunQuery = query_from(&uri)?;
    q.metadata = meta_filters(&uri);
    Ok(Json(json!(state.store.search_runs(&q).await?)))
}

pub(crate) async fn metrics(State(state): State<AppState>, uri: Uri) -> ApiResult<Value> {
    let mut q: MetricsQuery = query_from(&uri)?;
    q.metadata = meta_filters(&uri);
    Ok(Json(json!(state.store.metrics(&q).await?)))
}

// ---------- feedback ----------

#[derive(Deserialize, Default)]
#[serde(default)]
pub(crate) struct FeedbackFilter {
    run_id: Option<String>,
    trace_id: Option<String>,
    key: Option<String>,
    project: Option<String>,
    from_ms: Option<i64>,
    to_ms: Option<i64>,
}

pub(crate) async fn create_feedback(State(state): State<AppState>, b: Bytes) -> ApiResult<Value> {
    let f: FeedbackInput = body(&b, "feedback")?;
    Ok(Json(json!(state.store.add_feedback(&f).await?)))
}

pub(crate) async fn list_feedback(
    State(state): State<AppState>,
    Query(f): Query<FeedbackFilter>,
) -> ApiResult<Value> {
    Ok(Json(json!(
        state
            .store
            .feedback(f.run_id.as_deref(), f.trace_id.as_deref(), f.key.as_deref())
            .await?
    )))
}

pub(crate) async fn run_feedback(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Value> {
    Ok(Json(json!(
        state.store.feedback(Some(&id), None, None).await?
    )))
}

pub(crate) async fn feedback_stats(
    State(state): State<AppState>,
    Query(f): Query<FeedbackFilter>,
) -> ApiResult<Value> {
    Ok(Json(json!(
        state
            .store
            .feedback_stats(f.project.as_deref(), f.from_ms, f.to_ms)
            .await?
    )))
}

pub(crate) async fn delete_feedback(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Value> {
    deleted(state.store.delete_feedback(&id).await?, "feedback")
}

// ---------- prompt hub ----------

pub(crate) async fn list_prompts(State(state): State<AppState>) -> ApiResult<Value> {
    Ok(Json(json!(state.store.prompts().await?)))
}

pub(crate) async fn push_prompt(State(state): State<AppState>, b: Bytes) -> ApiResult<Value> {
    let p: PromptPushInput = body(&b, "prompt")?;
    let (v, created) = state.store.push_prompt(&p).await?;
    Ok(Json(json!({
        "name": v.name,
        "commit": v.commit,
        "version": v.version,
        "created": created,
        "prompt": v,
    })))
}

/// `GET /api/prompts/{name}`; `name@ref` resolves a version instead.
pub(crate) async fn get_prompt(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> ApiResult<Value> {
    if let Some((n, r)) = name.split_once('@') {
        let v = state.store.resolve_prompt(n, r).await?;
        return Ok(Json(json!(found(v, "prompt version")?)));
    }
    Ok(Json(json!(found(
        state.store.prompt(&name).await?,
        "prompt"
    )?)))
}

pub(crate) async fn resolve_prompt(
    State(state): State<AppState>,
    Path((name, reference)): Path<(String, String)>,
) -> ApiResult<Value> {
    let v = state.store.resolve_prompt(&name, &reference).await?;
    Ok(Json(json!(found(v, "prompt version")?)))
}

#[derive(Deserialize)]
pub(crate) struct DiffQuery {
    from: String,
    #[serde(default)]
    to: Option<String>,
}

pub(crate) async fn diff_prompt(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Query(q): Query<DiffQuery>,
) -> ApiResult<Value> {
    let to = q.to.unwrap_or_else(|| "latest".into());
    let d = state.store.diff_prompt(&name, &q.from, &to).await?;
    Ok(Json(json!(found(d, "prompt version")?)))
}

pub(crate) async fn set_prompt_tag(
    State(state): State<AppState>,
    Path((name, tag)): Path<(String, String)>,
    b: Bytes,
) -> ApiResult<Value> {
    let v = parse_json(&b)?;
    let reference = v["commit"]
        .as_str()
        .or_else(|| v["ref"].as_str())
        .ok_or_else(|| ApiError::bad_request("expected {\"commit\": \"<ref>\"}"))?;
    let r = state.store.set_prompt_tag(&name, &tag, reference).await?;
    Ok(Json(json!(found(r, "prompt version")?)))
}

pub(crate) async fn delete_prompt_tag(
    State(state): State<AppState>,
    Path((name, tag)): Path<(String, String)>,
) -> ApiResult<Value> {
    deleted(state.store.delete_prompt_tag(&name, &tag).await?, "tag")
}

pub(crate) async fn delete_prompt(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> ApiResult<Value> {
    deleted(state.store.delete_prompt(&name).await?, "prompt")
}

// ---------- annotation queues ----------

#[derive(Deserialize, Default)]
#[serde(default)]
pub(crate) struct QueueInput {
    name: String,
    description: String,
    keys: Vec<String>,
}

pub(crate) async fn list_queues(State(state): State<AppState>) -> ApiResult<Value> {
    Ok(Json(json!(state.store.queues().await?)))
}

pub(crate) async fn create_queue(State(state): State<AppState>, b: Bytes) -> ApiResult<Value> {
    let q: QueueInput = body(&b, "queue")?;
    Ok(Json(json!(
        state
            .store
            .create_queue(&q.name, &q.description, &q.keys)
            .await?
    )))
}

#[derive(Deserialize, Default)]
#[serde(default)]
pub(crate) struct ItemsQuery {
    status: Option<String>,
    limit: Option<i64>,
}

pub(crate) async fn get_queue(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(q): Query<ItemsQuery>,
) -> ApiResult<Value> {
    let queue = found(state.store.queue(&id).await?, "queue")?;
    let items = state
        .store
        .queue_items(&id, q.status.as_deref(), q.limit.unwrap_or(200))
        .await?;
    let summary = state
        .store
        .queues()
        .await?
        .into_iter()
        .find(|s| s.queue.id == id);
    Ok(Json(json!({
        "queue": queue,
        "pending": summary.as_ref().map(|s| s.pending).unwrap_or(0),
        "done": summary.as_ref().map(|s| s.done).unwrap_or(0),
        "items": items,
    })))
}

pub(crate) async fn delete_queue(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Value> {
    deleted(state.store.delete_queue(&id).await?, "queue")
}

pub(crate) async fn enqueue(
    State(state): State<AppState>,
    Path(id): Path<String>,
    b: Bytes,
) -> ApiResult<Value> {
    let v = parse_json(&b)?;
    let ids: Vec<String> = match v.get("run_ids").or_else(|| v.get("runs")) {
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(|x| x.as_str().map(str::to_string))
            .collect(),
        _ => match v["run_id"].as_str() {
            Some(r) => vec![r.to_string()],
            None => return Err(ApiError::bad_request("expected {\"run_ids\": [...]}")),
        },
    };
    let added = state.store.enqueue(&id, &ids).await?;
    Ok(Json(json!({ "added": added })))
}

/// The oldest pending item (with its run, trace and feedback), or `null`.
pub(crate) async fn queue_next(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Value> {
    let queue = found(state.store.queue(&id).await?, "queue")?;
    let item = state
        .store
        .queue_items(&id, Some("pending"), 1)
        .await?
        .into_iter()
        .next();
    let Some(item) = item else {
        return Ok(Json(json!({ "queue": queue, "item": null })));
    };
    let feedback = state.store.feedback(Some(&item.run_id), None, None).await?;
    Ok(Json(
        json!({ "queue": queue, "item": item, "feedback": feedback }),
    ))
}

#[derive(Deserialize, Default)]
#[serde(default)]
pub(crate) struct DoneInput {
    feedback: Vec<FeedbackInput>,
}

pub(crate) async fn complete_queue_item(
    State(state): State<AppState>,
    Path((id, run_id)): Path<(String, String)>,
    b: Bytes,
) -> ApiResult<Value> {
    let d: DoneInput = if b.is_empty() {
        DoneInput::default()
    } else {
        body(&b, "review")?
    };
    let ok = state
        .store
        .complete_queue_item(&id, &run_id, &d.feedback)
        .await?;
    if ok {
        Ok(Json(json!({ "done": true })))
    } else {
        Err(ApiError::not_found("queue item"))
    }
}

pub(crate) async fn remove_queue_item(
    State(state): State<AppState>,
    Path((id, run_id)): Path<(String, String)>,
) -> ApiResult<Value> {
    deleted(
        state.store.remove_queue_item(&id, &run_id).await?,
        "queue item",
    )
}

// ---------- datasets / pairwise ----------

/// `POST /api/datasets/{name}/examples`: an `Example`, or
/// `{"run_id": "...", "reference": <optional>}` to copy a run (its inputs,
/// and its outputs as the reference unless one is given).
pub(crate) async fn add_example(
    State(state): State<AppState>,
    Path(name): Path<String>,
    b: Bytes,
) -> ApiResult<Value> {
    let v = parse_json(&b)?;
    let example = if let Some(run_id) = v["run_id"].as_str() {
        let run = found(state.store.run(run_id).await?, "run")?;
        let reference = match v.get("reference") {
            Some(r) if !r.is_null() => Some(r.clone()),
            _ => run.run.outputs.clone(),
        };
        let mut ex = Example::new(run.run.inputs.clone());
        ex.reference = reference;
        ex.metadata = json!({ "source_run_id": run_id, "source_trace_id": run.run.trace_id });
        ex
    } else {
        serde_json::from_value::<Example>(v)
            .map_err(|e| ApiError::bad_request(format!("invalid example: {e}")))?
    };
    let count = state.store.add_example(&name, &example).await?;
    Ok(Json(
        json!({ "dataset": name, "id": example.id, "examples": count }),
    ))
}

#[derive(Deserialize, Default)]
#[serde(default)]
pub(crate) struct DatasetFilter {
    dataset: Option<String>,
}

pub(crate) async fn list_pairwise(
    State(state): State<AppState>,
    Query(f): Query<DatasetFilter>,
) -> ApiResult<Value> {
    Ok(Json(json!(
        state.store.pairwise_list(f.dataset.as_deref()).await?
    )))
}

pub(crate) async fn put_pairwise(State(state): State<AppState>, b: Bytes) -> ApiResult<Value> {
    let r: PairwiseReport = body(&b, "pairwise report")?;
    state.store.put_pairwise(&r).await?;
    Ok(Json(json!({ "id": r.id })))
}

pub(crate) async fn get_pairwise(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Value> {
    Ok(Json(json!(found(
        state.store.pairwise(&id).await?,
        "pairwise report"
    )?)))
}

// ---------- pricing, settings, retention ----------

pub(crate) async fn get_pricing(State(state): State<AppState>) -> ApiResult<Value> {
    Ok(Json(json!(state.store.pricing())))
}

pub(crate) async fn put_pricing(State(state): State<AppState>, b: Bytes) -> ApiResult<Value> {
    let p: Pricing = body(&b, "pricing table")?;
    let priced = state.store.set_pricing(p).await?;
    Ok(Json(json!({ "ok": true, "priced_runs": priced })))
}

pub(crate) async fn reset_pricing(State(state): State<AppState>) -> ApiResult<Value> {
    let priced = state.store.reset_pricing().await?;
    Ok(Json(json!({ "ok": true, "priced_runs": priced })))
}

pub(crate) async fn recompute_costs(State(state): State<AppState>) -> ApiResult<Value> {
    let priced = state.store.recompute_costs().await?;
    Ok(Json(json!({ "ok": true, "priced_runs": priced })))
}

pub(crate) async fn get_settings(State(state): State<AppState>) -> ApiResult<Value> {
    Ok(Json(json!(state.store.settings().await?)))
}

pub(crate) async fn put_settings(State(state): State<AppState>, b: Bytes) -> ApiResult<Value> {
    let s: Settings = body(&b, "settings")?;
    state.store.set_settings(&s).await?;
    Ok(Json(json!(s)))
}

pub(crate) async fn run_retention(State(state): State<AppState>) -> ApiResult<Value> {
    Ok(Json(json!(state.store.apply_retention().await?)))
}

// ---------- rules ----------

pub(crate) async fn list_rules(State(state): State<AppState>) -> ApiResult<Value> {
    Ok(Json(json!(state.store.rules().await?)))
}

pub(crate) async fn create_rule(State(state): State<AppState>, b: Bytes) -> ApiResult<Value> {
    let r: RuleInput = body(&b, "rule")?;
    Ok(Json(json!(state.store.put_rule(None, &r).await?)))
}

pub(crate) async fn get_rule(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Value> {
    Ok(Json(json!(found(state.store.rule(&id).await?, "rule")?)))
}

pub(crate) async fn update_rule(
    State(state): State<AppState>,
    Path(id): Path<String>,
    b: Bytes,
) -> ApiResult<Value> {
    found(state.store.rule(&id).await?, "rule")?;
    let r: RuleInput = body(&b, "rule")?;
    Ok(Json(json!(state.store.put_rule(Some(&id), &r).await?)))
}

pub(crate) async fn delete_rule(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Value> {
    deleted(state.store.delete_rule(&id).await?, "rule")
}

#[derive(Deserialize, Default)]
#[serde(default)]
pub(crate) struct LimitQuery {
    limit: Option<i64>,
}

/// Runs a rule over the latest matching runs now.
pub(crate) async fn run_rule(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(q): Query<LimitQuery>,
) -> ApiResult<Value> {
    let rule = found(state.store.rule(&id).await?, "rule")?;
    let n = engine(&state)?
        .backfill_rule(&rule, q.limit.unwrap_or(100))
        .await?;
    Ok(Json(json!({ "feedback": n })))
}

// ---------- alerts ----------

pub(crate) async fn list_alerts(State(state): State<AppState>) -> ApiResult<Value> {
    Ok(Json(json!(state.store.alerts().await?)))
}

pub(crate) async fn create_alert(State(state): State<AppState>, b: Bytes) -> ApiResult<Value> {
    let a: AlertInput = body(&b, "alert")?;
    Ok(Json(json!(state.store.put_alert(None, &a).await?)))
}

pub(crate) async fn get_alert(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Value> {
    Ok(Json(json!(found(state.store.alert(&id).await?, "alert")?)))
}

pub(crate) async fn update_alert(
    State(state): State<AppState>,
    Path(id): Path<String>,
    b: Bytes,
) -> ApiResult<Value> {
    found(state.store.alert(&id).await?, "alert")?;
    let a: AlertInput = body(&b, "alert")?;
    Ok(Json(json!(state.store.put_alert(Some(&id), &a).await?)))
}

pub(crate) async fn delete_alert(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Value> {
    deleted(state.store.delete_alert(&id).await?, "alert")
}

/// Evaluates an alert now (fires the webhook if it crosses the threshold).
pub(crate) async fn check_alert(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Value> {
    let a = found(state.store.alert(&id).await?, "alert")?;
    Ok(Json(engine(&state)?.check_alert(&a).await?))
}
