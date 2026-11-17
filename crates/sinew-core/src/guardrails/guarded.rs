//! The [`Guarded`] chat model and the [`GuardStep`] runnable.

use std::collections::HashMap;

use async_trait::async_trait;
use futures_util::StreamExt;

use super::{GuardDecision, GuardEvent, GuardObserver, GuardrailStage, Guardrails};
use crate::{
    runnable::{Runnable, RunnableStream},
    stream::{events_from_response, ChatStream},
    Capabilities, ChatModel, ChatRequest, ChatResponse, Error, Message, Result, Role, StreamEvent,
    Usage,
};

/// How a [`Guarded`] model applies output guardrails to
/// [`chat_stream`](ChatModel::chat_stream).
///
/// Input guardrails always run before the stream starts, in both modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StreamGuardMode {
    /// Run the whole request as [`Guarded::chat_with_report`] (the inner
    /// model is called with `chat`, retries included), then replay the
    /// checked response as stream events. Nothing reaches the caller before
    /// every output guardrail has passed, at the cost of losing incremental
    /// output.
    #[default]
    Buffer,
    /// Forward the inner model's events as they arrive and run the output
    /// guardrails on the complete text when the inner stream ends. Text
    /// already delivered cannot be taken back, so a guardrail that would
    /// block, refuse, retry or rewrite (redact) ends the stream with an
    /// `Err(`[`Error::Guardrail`]`)` item in place of
    /// [`StreamEvent::Done`]; consumers must discard what they showed.
    /// [`Warn`](super::GuardAction::Warn) rules record their event and the
    /// stream ends normally.
    Final,
}

/// What a [`Guarded`] call recorded: every guardrail that did not pass and
/// the number of retries.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GuardReport {
    /// Events in the order they happened (input checks first).
    pub events: Vec<GuardEvent>,
    /// Extra model calls made by [`Retry`](super::GuardAction::Retry) rules.
    pub retries: usize,
}

impl GuardReport {
    /// Events from [`Warn`](super::GuardAction::Warn) rules.
    pub fn warnings(&self) -> impl Iterator<Item = &GuardEvent> {
        self.events
            .iter()
            .filter(|e| e.action == super::GuardAction::Warn)
    }
}

/// A [`ChatModel`] with input and output guardrails; built with
/// [`ChatModelExt::with_guardrails`](crate::ChatModelExt::with_guardrails).
///
/// For each request:
///
/// 1. **Input.** The messages after the last assistant message (the new
///    user messages and tool results) are checked with the input
///    guardrails, user messages at [`GuardrailStage::Input`] and tool
///    results at [`GuardrailStage::ToolOutput`]. Rewrites replace the
///    message content; a block fails with [`Error::Guardrail`]; a refusal
///    returns the refusal message as the answer without calling the model.
///    Earlier messages were checked when they were new and are not checked
///    again.
/// 2. **Model.** The inner model is called.
/// 3. **Output.** If the response has no tool calls, its text is checked
///    with the output guardrails at [`GuardrailStage::Output`]. A rewrite
///    or refusal replaces the text, a block fails, and a
///    [`Retry`](super::GuardAction::Retry) rule appends the rejected answer
///    and a user message with the reason, then goes back to step 2, within
///    the rule's budget. Responses with tool calls are intermediate steps of
///    an agent loop and pass unchecked; their tool results are checked as
///    input on the next call.
///
/// Token usage of retried calls is added to the returned response. Only
/// `content` is checked; multimodal parts pass through.
///
/// Events go to the observer set with [`on_event`](Self::on_event) and are
/// returned by [`chat_with_report`](Self::chat_with_report). Streaming is
/// described in [`StreamGuardMode`].
pub struct Guarded<M> {
    inner: M,
    input: Guardrails,
    output: Guardrails,
    stream_mode: StreamGuardMode,
}

impl<M> Guarded<M> {
    /// Wraps `inner`. Pass [`Guardrails::new()`] for an empty side.
    pub fn new(inner: M, input: impl Into<Guardrails>, output: impl Into<Guardrails>) -> Self {
        Self {
            inner,
            input: input.into(),
            output: output.into(),
            stream_mode: StreamGuardMode::default(),
        }
    }

    /// Calls `f` for every [`GuardEvent`] on both sides. Replaces observers
    /// set on the [`Guardrails`] passed in.
    pub fn on_event(self, f: impl Fn(&GuardEvent) + Send + Sync + 'static) -> Self {
        self.with_observer(std::sync::Arc::new(f))
    }

    /// Sets a shared observer on both sides (see [`on_event`](Self::on_event)).
    pub fn with_observer(mut self, observer: GuardObserver) -> Self {
        self.input = self.input.with_observer(observer.clone());
        self.output = self.output.with_observer(observer);
        self
    }

    /// Sets how output guardrails apply to streams (default
    /// [`StreamGuardMode::Buffer`]).
    pub fn stream_mode(mut self, mode: StreamGuardMode) -> Self {
        self.stream_mode = mode;
        self
    }

    /// The wrapped model.
    pub fn inner(&self) -> &M {
        &self.inner
    }

    /// The input guardrails.
    pub fn input_guardrails(&self) -> &Guardrails {
        &self.input
    }

    /// The output guardrails.
    pub fn output_guardrails(&self) -> &Guardrails {
        &self.output
    }
}

enum InputCheck {
    Continue(ChatRequest),
    Refused(ChatResponse),
}

fn add_usage(total: &mut Usage, u: &Usage) {
    total.input_tokens = total.input_tokens.saturating_add(u.input_tokens);
    total.output_tokens = total.output_tokens.saturating_add(u.output_tokens);
    total.cached_input_tokens = total
        .cached_input_tokens
        .saturating_add(u.cached_input_tokens);
    total.cache_creation_input_tokens = total
        .cache_creation_input_tokens
        .saturating_add(u.cache_creation_input_tokens);
    total.reasoning_tokens = total.reasoning_tokens.saturating_add(u.reasoning_tokens);
}

/// The user message sent after an answer rejected by a retry rule.
pub(crate) fn retry_feedback(guard: &str, reason: &str) -> String {
    format!(
        "Your previous answer was rejected by the `{guard}` guardrail: {reason}. \
         Answer again and fix this."
    )
}

impl<M: ChatModel> Guarded<M> {
    fn refusal(&self, message: String) -> ChatResponse {
        ChatResponse {
            message: Message::assistant(message),
            usage: Usage::default(),
            model: self.inner.name().to_string(),
        }
    }

    async fn check_input(
        &self,
        mut request: ChatRequest,
        report: &mut GuardReport,
    ) -> Result<InputCheck> {
        if self.input.is_empty() {
            return Ok(InputCheck::Continue(request));
        }
        let start = request
            .messages
            .iter()
            .rposition(|m| m.role == Role::Assistant)
            .map_or(0, |i| i + 1);
        for msg in request.messages.iter_mut().skip(start) {
            let stage = match msg.role {
                Role::User => GuardrailStage::Input,
                Role::Tool => GuardrailStage::ToolOutput,
                _ => continue,
            };
            let outcome = self.input.check(stage, &msg.content).await?;
            report.events.extend(outcome.events);
            match outcome.decision {
                GuardDecision::Allow => msg.content = outcome.text,
                GuardDecision::Refuse { message, .. } => {
                    return Ok(InputCheck::Refused(self.refusal(message)))
                }
                GuardDecision::Block { guard, reason }
                | GuardDecision::Retry { guard, reason, .. } => {
                    return Err(Error::Guardrail {
                        guard,
                        stage,
                        reason,
                    })
                }
            }
        }
        Ok(InputCheck::Continue(request))
    }

    /// Like [`chat`](ChatModel::chat), and also returns the
    /// [`GuardReport`] (warnings, rewrites, retries).
    ///
    /// ```
    /// use sinew_core::guardrails::{guard, Guardrails, MaxLength};
    /// use sinew_core::{mock::ScriptedModel, ChatModelExt, ChatRequest, Message};
    ///
    /// # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
    /// let model = ScriptedModel::new([Message::assistant("a rather long answer")])
    ///     .with_guardrails(Guardrails::new(), [guard(MaxLength::chars(10)).warn()]);
    /// let (resp, report) = model
    ///     .chat_with_report(ChatRequest::new(vec![Message::user("hi")]))
    ///     .await
    ///     .unwrap();
    /// assert_eq!(resp.message.content, "a rather long answer");
    /// assert_eq!(report.warnings().count(), 1);
    /// # });
    /// ```
    pub async fn chat_with_report(
        &self,
        request: ChatRequest,
    ) -> Result<(ChatResponse, GuardReport)> {
        let mut report = GuardReport::default();
        let mut request = match self.check_input(request, &mut report).await? {
            InputCheck::Continue(r) => r,
            InputCheck::Refused(resp) => return Ok((resp, report)),
        };
        let mut usage = Usage::default();
        let mut retries: HashMap<usize, usize> = HashMap::new();
        loop {
            let mut resp = self.inner.chat(request.clone()).await?;
            add_usage(&mut usage, &resp.usage);
            if resp.message.has_tool_calls() || self.output.is_empty() {
                resp.usage = usage;
                return Ok((resp, report));
            }
            let outcome = self
                .output
                .check(GuardrailStage::Output, &resp.message.content)
                .await?;
            report.events.extend(outcome.events);
            match outcome.decision {
                GuardDecision::Allow => resp.message.content = outcome.text,
                GuardDecision::Refuse { message, .. } => resp.message.content = message,
                GuardDecision::Block { guard, reason } => {
                    return Err(Error::Guardrail {
                        guard,
                        stage: GuardrailStage::Output,
                        reason,
                    })
                }
                GuardDecision::Retry {
                    guard,
                    reason,
                    rule,
                    max_retries,
                } => {
                    let used = retries.entry(rule).or_insert(0);
                    if *used >= max_retries {
                        return Err(Error::Guardrail {
                            guard,
                            stage: GuardrailStage::Output,
                            reason: format!("{reason} (still failing after {max_retries} retries)"),
                        });
                    }
                    *used += 1;
                    report.retries += 1;
                    request.messages.push(resp.message);
                    request
                        .messages
                        .push(Message::user(retry_feedback(&guard, &reason)));
                    continue;
                }
            }
            resp.usage = usage;
            return Ok((resp, report));
        }
    }
}

#[async_trait]
impl<M: ChatModel> ChatModel for Guarded<M> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn capabilities(&self) -> Capabilities {
        self.inner.capabilities()
    }

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        self.chat_with_report(request).await.map(|(r, _)| r)
    }

    async fn chat_stream(&self, request: ChatRequest) -> Result<ChatStream> {
        let replay = |resp: ChatResponse| -> ChatStream {
            Box::pin(futures_util::stream::iter(
                events_from_response(resp).into_iter().map(Ok),
            ))
        };
        if self.stream_mode == StreamGuardMode::Buffer {
            let (resp, _) = self.chat_with_report(request).await?;
            return Ok(replay(resp));
        }
        let mut report = GuardReport::default();
        let request = match self.check_input(request, &mut report).await? {
            InputCheck::Continue(r) => r,
            InputCheck::Refused(resp) => return Ok(replay(resp)),
        };
        let inner = self.inner.chat_stream(request).await?;
        if self.output.is_empty() {
            return Ok(inner);
        }
        let state = FinalState {
            inner,
            output: self.output.clone(),
            text: String::new(),
            tool_calls: false,
            finished: false,
        };
        Ok(Box::pin(futures_util::stream::unfold(
            state,
            |mut st| async move {
                if st.finished {
                    return None;
                }
                let item = st.inner.next().await?;
                match &item {
                    Ok(StreamEvent::TextDelta { text }) => st.text.push_str(text),
                    Ok(StreamEvent::ToolCall { .. } | StreamEvent::ToolCallDelta { .. }) => {
                        st.tool_calls = true
                    }
                    Ok(StreamEvent::Done { .. }) if !st.tool_calls => {
                        st.finished = true;
                        let checked = check_final(&st.output, &st.text).await;
                        return Some((checked.map(|()| item.expect("matched Ok")), st));
                    }
                    _ => {}
                }
                Some((item, st))
            },
        )))
    }
}

struct FinalState {
    inner: ChatStream,
    output: Guardrails,
    text: String,
    tool_calls: bool,
    finished: bool,
}

/// Output check at the end of a [`StreamGuardMode::Final`] stream.
async fn check_final(output: &Guardrails, text: &str) -> Result<()> {
    let stage = GuardrailStage::Output;
    let outcome = output.check(stage, text).await?;
    let (guard, reason) = match outcome.decision {
        GuardDecision::Allow if outcome.text == text => return Ok(()),
        GuardDecision::Allow => {
            let guard = outcome
                .events
                .iter()
                .rev()
                .find(|e| e.rewrote && e.action == super::GuardAction::Redact)
                .map_or_else(String::new, |e| e.guard.clone());
            (
                guard,
                "the streamed text was rewritten after it was sent; discard it".to_string(),
            )
        }
        GuardDecision::Refuse { guard, .. } => (
            guard,
            "the streamed answer was refused after it was sent; discard it".to_string(),
        ),
        GuardDecision::Block { guard, reason } | GuardDecision::Retry { guard, reason, .. } => {
            (guard, reason)
        }
    };
    Err(Error::Guardrail {
        guard,
        stage,
        reason,
    })
}

/// A [`Runnable`] that checks its input with [`Guardrails`] at one stage;
/// built with [`Guardrails::step`].
///
/// - `Runnable<String, String>`: returns the (possibly rewritten) text, the
///   refusal message for [`Refuse`](super::GuardAction::Refuse) rules, and
///   [`Error::Guardrail`] for blocks. [`Retry`](super::GuardAction::Retry)
///   rules block: there is no model to ask again (wrap the model with
///   [`Guarded`] for that).
/// - `Runnable<ChatResponse, ChatResponse>`: the same on the message text,
///   so it can follow a model in a chain; responses with tool calls pass
///   unchecked.
///
/// Streaming input is merged first (the default
/// [`Runnable::transform`]), so the step buffers.
///
/// ```
/// use sinew_core::guardrails::{guard, Guardrails, GuardrailStage, SecretRedactor};
/// use sinew_core::{mock::ScriptedModel, ChatResponse, Message, Runnable, RunnableExt};
/// use std::sync::Arc;
///
/// # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
/// let model = Arc::new(ScriptedModel::new([Message::assistant(
///     "use the key sk-proj-abcdefghijklmnopqrstuvwx",
/// )]));
/// let chain = model
///     .pipe(Guardrails::new().with(guard(SecretRedactor::new())).step(GuardrailStage::Output))
///     .map(|r: ChatResponse| r.message.content);
/// let out: String = chain.invoke("hi").await.unwrap();
/// assert_eq!(out, "use the key [REDACTED_SECRET]");
/// # });
/// ```
#[derive(Debug, Clone)]
pub struct GuardStep {
    guardrails: Guardrails,
    stage: GuardrailStage,
}

impl GuardStep {
    /// Checks with `guardrails` at `stage`.
    pub fn new(guardrails: Guardrails, stage: GuardrailStage) -> Self {
        Self { guardrails, stage }
    }
}

#[async_trait]
impl Runnable<String, String> for GuardStep {
    async fn invoke(&self, input: String) -> Result<String> {
        self.guardrails
            .check(self.stage, &input)
            .await?
            .into_result(self.stage)
    }
}

#[async_trait]
impl Runnable<ChatResponse, ChatResponse> for GuardStep {
    async fn invoke(&self, mut input: ChatResponse) -> Result<ChatResponse> {
        if input.message.has_tool_calls() {
            return Ok(input);
        }
        input.message.content = self
            .guardrails
            .check(self.stage, &input.message.content)
            .await?
            .into_result(self.stage)?;
        Ok(input)
    }

    async fn transform(
        &self,
        input: RunnableStream<ChatResponse>,
    ) -> Result<RunnableStream<ChatResponse>> {
        let whole = crate::runnable::merge_chunks(input).await?;
        let out = Runnable::<ChatResponse, ChatResponse>::invoke(self, whole).await?;
        Ok(Box::pin(futures_util::stream::once(async move { Ok(out) })))
    }
}
