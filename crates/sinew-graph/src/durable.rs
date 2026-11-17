//! Durable tool calls: a tool that finished before a crash is not run again
//! on resume, and a side-effecting tool whose outcome is unknown is never
//! repeated silently.

use serde::{Deserialize, Serialize};
use serde_json::json;
use sinew_core::{tool::with_idempotency_key, Message, ToolBox, ToolCall};
use sinew_trace::traced_tool;

use crate::{checkpoint::now_millis, BoxError, NodeContext, ToolEntry};

/// What to do with a side-effecting tool call (`Tool::side_effects`) that
/// started before the run stopped and has no recorded result: it may or may
/// not have happened.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnknownOutcome {
    /// Answer the model with an error explaining the outcome is unknown, so
    /// it (or the user) checks before trying again (default).
    #[default]
    Fail,
    /// Run the call again. Safe when the tool passes
    /// [`idempotency_key`](sinew_core::tool::idempotency_key) to a system
    /// that drops duplicates.
    Retry,
    /// Pause with an interrupt carrying `unknown_tool_outcome`; resume with
    /// `true` to run the call again or `false` to answer as in `Fail`.
    Interrupt,
}

/// The reply sent for a call whose outcome is unknown.
pub fn unknown_outcome_reply(name: &str) -> String {
    format!(
        "error: the outcome of `{name}` is unknown: the call started before the run was \
         interrupted and may have completed. Check its effect before calling it again."
    )
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// Ledger key of a tool call: stable across retries and resumes of the same
/// call, different for every call of a thread. `position` identifies where
/// the call was requested (e.g. the index of the assistant message), since
/// some servers reuse call ids across turns.
pub fn tool_call_key(ctx: &NodeContext, position: &str, call: &ToolCall) -> String {
    let args = serde_json::to_string(&call.arguments).unwrap_or_default();
    let digest = fnv1a(format!("{}\u{1f}{args}", call.name).as_bytes());
    format!(
        "{}/{}/{position}/{}/{digest:016x}",
        ctx.namespace.join("|"),
        ctx.node,
        call.id
    )
}

/// Runs one tool call through the checkpointer's tool ledger when it has
/// one (otherwise exactly like [`traced_tool`]):
///
/// - a call recorded as done returns its recorded result without running;
/// - a call recorded as started but not finished runs again if the tool has
///   no side effects, and follows `unknown` otherwise;
/// - a new call is recorded as pending, runs with
///   [`idempotency_key`](sinew_core::tool::idempotency_key) set to its key,
///   and is recorded as done.
///
/// Ledger write failures never fail the call; the run is then just not
/// durable for that call.
pub async fn durable_tool_call(
    tools: &ToolBox,
    call: &ToolCall,
    ctx: &NodeContext,
    position: &str,
    unknown: UnknownOutcome,
) -> Result<Message, BoxError> {
    let side_effects = tools.get(&call.name).is_some_and(|t| t.side_effects());
    let content = durable_run(ctx, position, call, side_effects, unknown, || async {
        traced_tool(tools, call, ctx.tracer.as_ref(), ctx.run.as_ref())
            .await
            .content
    })
    .await?;
    Ok(Message::tool_result(call.id.clone(), content))
}

/// The ledger logic of [`durable_tool_call`] around any way of running a
/// call: `run` produces the tool message content (errors as `"error: ..."`).
/// For runtimes that keep tools outside a [`ToolBox`] (like the harness).
pub async fn durable_run<F, Fut>(
    ctx: &NodeContext,
    position: &str,
    call: &ToolCall,
    side_effects: bool,
    unknown: UnknownOutcome,
    run: F,
) -> Result<String, BoxError>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = String>,
{
    let Some(cp) = ctx
        .checkpointer
        .as_ref()
        .filter(|c| c.supports_tool_ledger())
    else {
        return Ok(run().await);
    };
    let key = tool_call_key(ctx, position, call);
    let thread = ctx.thread_id.as_str();
    match cp.get_tool_entry(thread, &key).await {
        Ok(Some(ToolEntry::Done { content })) => {
            ctx.emit(json!({ "tool_replayed": { "name": call.name, "id": call.id } }));
            return Ok(content);
        }
        Ok(Some(ToolEntry::Pending { .. })) if side_effects => {
            let rerun = match unknown {
                UnknownOutcome::Retry => true,
                UnknownOutcome::Fail => false,
                UnknownOutcome::Interrupt => ctx.interrupt::<bool>(json!({
                    "unknown_tool_outcome": {
                        "name": call.name,
                        "id": call.id,
                        "arguments": call.arguments,
                    }
                }))?,
            };
            if !rerun {
                let content = unknown_outcome_reply(&call.name);
                let _ = cp
                    .put_tool_entry(
                        thread,
                        &key,
                        &ToolEntry::Done {
                            content: content.clone(),
                        },
                    )
                    .await;
                return Ok(content);
            }
        }
        _ => {}
    }
    let _ = cp
        .put_tool_entry(
            thread,
            &key,
            &ToolEntry::Pending {
                name: call.name.clone(),
                arguments: call.arguments.clone(),
                started_ms: now_millis(),
            },
        )
        .await;
    let content = with_idempotency_key(key.clone(), run()).await;
    let _ = cp
        .put_tool_entry(
            thread,
            &key,
            &ToolEntry::Done {
                content: content.clone(),
            },
        )
        .await;
    Ok(content)
}
