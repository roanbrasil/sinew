//! Composable pipeline steps, the Rust take on LangChain's LCEL.
//!
//! A [`Runnable<I, O>`] turns an `I` into an `O` asynchronously. Prompt
//! templates, chat models, [`StructuredOutput`], output parsers and plain
//! closures are all runnables, and [`RunnableExt`] chains them:
//!
//! ```no_run
//! # use sinew_core::*;
//! # use sinew_core::runnable::*;
//! # async fn demo(model: std::sync::Arc<dyn ChatModel>) -> Result<()> {
//! let chain = PromptTemplate::new("Tell me a joke about {topic}")?
//!     .pipe(model)
//!     .pipe(StrOutputParser);
//! let joke: String = chain.invoke(vars([("topic", "Rust")])).await?;
//! # Ok(()) }
//! ```
//!
//! Combinators are plain generic structs, so a chain compiles to nested
//! static calls with no boxing between steps (each `invoke` costs one boxed
//! future from `async_trait`, which is noise next to a network call). Use
//! [`RunnableExt::boxed`] / [`BoxRunnable`] when you need type erasure, for
//! example to store heterogeneous steps in a `Vec` or a [`RunnableBranch`].
//!
//! ## Streaming
//!
//! [`Runnable::stream`] yields output *chunks*. Chat models stream
//! [`ChatResponse`] chunks (each holding a text delta, a complete tool call,
//! usage or the final model name), and [`StrOutputParser`](crate::StrOutputParser)
//! turns them into text deltas as they arrive, so
//! `prompt.pipe(model).pipe(StrOutputParser).stream(vars)` streams tokens.
//! Steps that cannot work on partial input (JSON parsing, closures, ...)
//! first merge the incoming chunks: `String` chunks are concatenated,
//! `ChatResponse` and `Message` chunks are folded into one value, and for
//! any other type the last chunk wins.

use std::{
    any::Any, borrow::Cow, collections::HashMap, future::Future, marker::PhantomData, sync::Arc,
};

use async_trait::async_trait;
use futures_util::{StreamExt, TryFutureExt};
use serde::{de::DeserializeOwned, Serialize};
use serde_json::Value;

use crate::{
    resilience::RetryPolicy,
    stream::{BoxStream, ChatStream},
    ChatModel, ChatRequest, ChatResponse, Error, Message, PromptTemplate, Result, Role,
    StreamEvent, StructuredOutput, Usage,
};

/// Default `max_concurrency` for [`Runnable::batch`].
pub const DEFAULT_BATCH_CONCURRENCY: usize = 8;

/// Template variables: what [`PromptTemplate`] and [`ChatPromptTemplate`]
/// take as input (they also accept a JSON object).
pub type Vars = HashMap<String, String>;

/// Builds [`Vars`] from pairs: `vars([("topic", "Rust")])`.
pub fn vars<K: Into<String>, V: Into<String>>(pairs: impl IntoIterator<Item = (K, V)>) -> Vars {
    pairs
        .into_iter()
        .map(|(k, v)| (k.into(), v.into()))
        .collect()
}

/// A stream of output chunks.
pub type RunnableStream<O> = BoxStream<'static, Result<O>>;

/// An async step from `I` to `O`. See the [module docs](self).
#[async_trait]
pub trait Runnable<I, O>: Send + Sync
where
    I: Send + 'static,
    O: Send + 'static,
{
    /// Runs the step once.
    async fn invoke(&self, input: I) -> Result<O>;

    /// Runs every input, at most [`DEFAULT_BATCH_CONCURRENCY`] at a time,
    /// and returns the results in input order.
    async fn batch(&self, inputs: Vec<I>) -> Vec<Result<O>> {
        self.batch_with_concurrency(inputs, DEFAULT_BATCH_CONCURRENCY)
            .await
    }

    /// [`batch`](Self::batch) with an explicit concurrency bound.
    async fn batch_with_concurrency(
        &self,
        inputs: Vec<I>,
        max_concurrency: usize,
    ) -> Vec<Result<O>> {
        futures_util::stream::iter(inputs)
            .map(|i| self.invoke(i))
            .buffered(max_concurrency.max(1))
            .collect()
            .await
    }

    /// Streams output chunks. The default runs [`invoke`](Self::invoke)
    /// and yields its result as a single chunk.
    async fn stream(&self, input: I) -> Result<RunnableStream<O>> {
        let out = self.invoke(input).await?;
        Ok(Box::pin(futures_util::stream::once(async move { Ok(out) })))
    }

    /// Consumes a stream of input chunks and streams output chunks. Used when
    /// this step follows a streaming step in a [`Pipe`]. The default merges
    /// the input chunks (see the [module docs](self)) and calls
    /// [`stream`](Self::stream).
    async fn transform(&self, input: RunnableStream<I>) -> Result<RunnableStream<O>> {
        let whole = merge_chunks(input).await?;
        self.stream(whole).await
    }
}

/// Collects a chunk stream into one value (see the [module docs](self)).
pub async fn merge_chunks<T: Send + 'static>(mut s: RunnableStream<T>) -> Result<T> {
    let mut chunks = Vec::new();
    while let Some(c) = s.next().await {
        chunks.push(c?);
    }
    match chunks.len() {
        0 => Err(Error::Stream("empty stream".into())),
        1 => Ok(chunks.pop().unwrap()),
        _ => Ok(merge_vec(chunks)),
    }
}

fn cast<T: 'static>(v: impl Any) -> T {
    *(Box::new(v) as Box<dyn Any>)
        .downcast::<T>()
        .expect("type checked by caller")
}

fn merge_vec<T: 'static>(mut chunks: Vec<T>) -> T {
    let any: &mut dyn Any = &mut chunks;
    if let Some(v) = any.downcast_mut::<Vec<String>>() {
        return cast(v.concat());
    }
    if let Some(v) = any.downcast_mut::<Vec<ChatResponse>>() {
        return cast(merge_responses(std::mem::take(v)));
    }
    if let Some(v) = any.downcast_mut::<Vec<Message>>() {
        let mut it = std::mem::take(v).into_iter();
        let mut first = it.next().expect("non-empty");
        for m in it {
            merge_message(&mut first, m);
        }
        return cast(first);
    }
    chunks.pop().expect("non-empty")
}

fn merge_message(into: &mut Message, m: Message) {
    into.content.push_str(&m.content);
    into.tool_calls.extend(m.tool_calls);
    into.parts.extend(m.parts);
    if into.tool_call_id.is_none() {
        into.tool_call_id = m.tool_call_id;
    }
}

fn merge_usage(into: &mut Usage, u: &Usage) {
    let set = |d: &mut u32, s: u32| {
        if s > 0 {
            *d = s
        }
    };
    set(&mut into.input_tokens, u.input_tokens);
    set(&mut into.output_tokens, u.output_tokens);
    set(&mut into.cached_input_tokens, u.cached_input_tokens);
    set(
        &mut into.cache_creation_input_tokens,
        u.cache_creation_input_tokens,
    );
    set(&mut into.reasoning_tokens, u.reasoning_tokens);
}

/// Folds [`ChatResponse`] chunks from a model stream into one response.
pub fn merge_responses(chunks: Vec<ChatResponse>) -> ChatResponse {
    let mut it = chunks.into_iter();
    let mut out = it.next().unwrap_or_else(|| ChatResponse {
        message: Message::assistant(""),
        usage: Usage::default(),
        model: String::new(),
    });
    for c in it {
        merge_message(&mut out.message, c.message);
        merge_usage(&mut out.usage, &c.usage);
        if !c.model.is_empty() {
            out.model = c.model;
        }
    }
    out
}

fn chunk(message: Message, usage: Usage, model: String) -> ChatResponse {
    ChatResponse {
        message,
        usage,
        model,
    }
}

/// Turns a model's event stream into [`ChatResponse`] chunks.
pub fn response_chunks(stream: ChatStream) -> RunnableStream<ChatResponse> {
    stream
        .filter_map(|ev| async move {
            match ev {
                Err(e) => Some(Err(e)),
                Ok(StreamEvent::TextDelta { text }) => Some(Ok(chunk(
                    Message::assistant(text),
                    Usage::default(),
                    String::new(),
                ))),
                Ok(StreamEvent::ToolCall { call, .. }) => Some(Ok(chunk(
                    Message::assistant_with_tools("", vec![call]),
                    Usage::default(),
                    String::new(),
                ))),
                Ok(StreamEvent::Usage { usage }) => {
                    Some(Ok(chunk(Message::assistant(""), usage, String::new())))
                }
                Ok(StreamEvent::Done { model, .. }) => {
                    Some(Ok(chunk(Message::assistant(""), Usage::default(), model)))
                }
                Ok(StreamEvent::ToolCallDelta { .. }) => None,
            }
        })
        .boxed()
}

// ---------------------------------------------------------------------------
// Chat models

/// Every chat model is a runnable from anything convertible into a
/// [`ChatRequest`]: a request, a `Vec<Message>`, a single [`Message`], or a
/// `String`/`&'static str` (sent as one user message, so
/// `prompt.pipe(model)` works).
#[async_trait]
impl<M, In> Runnable<In, ChatResponse> for M
where
    M: ChatModel,
    In: Into<ChatRequest> + Send + 'static,
{
    async fn invoke(&self, input: In) -> Result<ChatResponse> {
        self.chat(input.into()).await
    }
    async fn stream(&self, input: In) -> Result<RunnableStream<ChatResponse>> {
        Ok(response_chunks(self.chat_stream(input.into()).await?))
    }
}

// ---------------------------------------------------------------------------
// Prompts

/// Something template variables can be looked up in: [`Vars`],
/// `HashMap<&'static str, String>`, or a JSON object (strings are inserted
/// as-is, other values as JSON text, `null` as an empty string).
pub trait TemplateVars: Send + 'static {
    fn var(&self, name: &str) -> Option<Cow<'_, str>>;
}

impl TemplateVars for Vars {
    fn var(&self, name: &str) -> Option<Cow<'_, str>> {
        self.get(name).map(|s| Cow::Borrowed(s.as_str()))
    }
}

impl TemplateVars for HashMap<&'static str, String> {
    fn var(&self, name: &str) -> Option<Cow<'_, str>> {
        self.get(name).map(|s| Cow::Borrowed(s.as_str()))
    }
}

fn json_var(v: &Value) -> Cow<'_, str> {
    match v {
        Value::String(s) => Cow::Borrowed(s.as_str()),
        Value::Null => Cow::Borrowed(""),
        other => Cow::Owned(other.to_string()),
    }
}

impl TemplateVars for Value {
    fn var(&self, name: &str) -> Option<Cow<'_, str>> {
        self.get(name).map(json_var)
    }
}

impl TemplateVars for serde_json::Map<String, Value> {
    fn var(&self, name: &str) -> Option<Cow<'_, str>> {
        self.get(name).map(json_var)
    }
}

#[async_trait]
impl<V: TemplateVars> Runnable<V, String> for PromptTemplate {
    async fn invoke(&self, input: V) -> Result<String> {
        self.render_lookup(|name| input.var(name))
    }
}

/// A list of role + template pairs rendered into a [`ChatRequest`].
#[derive(Debug, Clone)]
pub struct ChatPromptTemplate {
    messages: Vec<(Role, PromptTemplate)>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
}

impl ChatPromptTemplate {
    /// `ChatPromptTemplate::from_messages([(Role::System, "You are {persona}"), (Role::User, "{question}")])`
    pub fn from_messages<S: Into<String>>(
        messages: impl IntoIterator<Item = (Role, S)>,
    ) -> Result<Self> {
        let messages = messages
            .into_iter()
            .map(|(r, t)| Ok((r, PromptTemplate::new(t)?)))
            .collect::<Result<_>>()?;
        Ok(Self {
            messages,
            temperature: None,
            max_tokens: None,
        })
    }

    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }

    pub fn with_max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = Some(n);
        self
    }

    /// Variable names across all messages, in order of first appearance.
    pub fn variables(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for (_, t) in &self.messages {
            for v in t.variables() {
                if !out.contains(&v) {
                    out.push(v);
                }
            }
        }
        out
    }

    fn build(&self, render: impl Fn(&PromptTemplate) -> Result<String>) -> Result<ChatRequest> {
        let mut msgs = Vec::with_capacity(self.messages.len());
        for (role, t) in &self.messages {
            let text = render(t)?;
            msgs.push(match role {
                Role::System => Message::system(text),
                Role::User => Message::user(text),
                Role::Assistant => Message::assistant(text),
                Role::Tool => Message::tool_result("", text),
            });
        }
        let mut req = ChatRequest::new(msgs);
        req.temperature = self.temperature;
        req.max_tokens = self.max_tokens;
        Ok(req)
    }

    /// Renders every message with `vars`.
    pub fn format(&self, vars: &impl TemplateVars) -> Result<ChatRequest> {
        self.build(|t| t.render_lookup(|name| vars.var(name)))
    }
}

#[async_trait]
impl<V: TemplateVars + Sync> Runnable<V, ChatRequest> for ChatPromptTemplate {
    async fn invoke(&self, input: V) -> Result<ChatRequest> {
        self.format(&input)
    }
}

// ---------------------------------------------------------------------------
// Structured output

/// [`StructuredOutput`] bound to a model; produced by
/// [`StructuredOutput::runnable`].
pub struct StructuredRunnable<T, M> {
    spec: StructuredOutput,
    model: M,
    _t: PhantomData<fn() -> T>,
}

impl StructuredOutput {
    /// Binds this spec to `model`, giving a runnable from anything
    /// `Into<ChatRequest>` (a request, messages, or a user-message string)
    /// to `T`.
    pub fn runnable<T, M>(self, model: M) -> StructuredRunnable<T, M>
    where
        T: DeserializeOwned + Send + 'static,
        M: ChatModel,
    {
        StructuredRunnable {
            spec: self,
            model,
            _t: PhantomData,
        }
    }
}

#[async_trait]
impl<T, M, In> Runnable<In, T> for StructuredRunnable<T, M>
where
    T: DeserializeOwned + Send + 'static,
    M: ChatModel,
    In: Into<ChatRequest> + Send + 'static,
{
    async fn invoke(&self, input: In) -> Result<T> {
        self.spec.generate(&self.model, input.into()).await
    }
}

// ---------------------------------------------------------------------------
// Combinators

/// `first` then `second`; built by [`RunnableExt::pipe`]. `I` is the
/// chain's input type and `M` the type passed between the two steps; both
/// are inferred.
pub struct Pipe<A, B, I, M> {
    first: A,
    second: B,
    _m: PhantomData<fn(I) -> M>,
}

#[async_trait]
impl<I, M, O, A, B> Runnable<I, O> for Pipe<A, B, I, M>
where
    I: Send + 'static,
    M: Send + 'static,
    O: Send + 'static,
    A: Runnable<I, M>,
    B: Runnable<M, O>,
{
    async fn invoke(&self, input: I) -> Result<O> {
        let mid = self.first.invoke(input).await?;
        self.second.invoke(mid).await
    }
    async fn stream(&self, input: I) -> Result<RunnableStream<O>> {
        let s = self.first.stream(input).await?;
        self.second.transform(s).await
    }
    async fn transform(&self, input: RunnableStream<I>) -> Result<RunnableStream<O>> {
        let s = self.first.transform(input).await?;
        self.second.transform(s).await
    }
}

/// Applies a sync function to the output; built by [`RunnableExt::map`].
pub struct Map<R, F, I, M> {
    inner: R,
    f: F,
    _m: PhantomData<fn(I) -> M>,
}

#[async_trait]
impl<I, M, O, R, F> Runnable<I, O> for Map<R, F, I, M>
where
    I: Send + 'static,
    M: Send + 'static,
    O: Send + 'static,
    R: Runnable<I, M>,
    F: Fn(M) -> O + Send + Sync,
{
    async fn invoke(&self, input: I) -> Result<O> {
        self.inner.invoke(input).await.map(&self.f)
    }
}

/// An async closure as a runnable; see [`lambda`].
pub struct Lambda<F>(F);

/// Wraps `async |input| -> Result<O>` as a runnable.
pub fn lambda<F>(f: F) -> Lambda<F> {
    Lambda(f)
}

#[async_trait]
impl<I, O, F, Fut> Runnable<I, O> for Lambda<F>
where
    I: Send + 'static,
    O: Send + 'static,
    F: Fn(I) -> Fut + Send + Sync,
    Fut: Future<Output = Result<O>> + Send,
{
    async fn invoke(&self, input: I) -> Result<O> {
        (self.0)(input).await
    }
}

/// A sync closure as a runnable; see [`lambda_sync`].
pub struct SyncLambda<F>(F);

/// Wraps `|input| -> Result<O>` as a runnable.
pub fn lambda_sync<F>(f: F) -> SyncLambda<F> {
    SyncLambda(f)
}

#[async_trait]
impl<I, O, F> Runnable<I, O> for SyncLambda<F>
where
    I: Send + 'static,
    O: Send + 'static,
    F: Fn(I) -> Result<O> + Send + Sync,
{
    async fn invoke(&self, input: I) -> Result<O> {
        (self.0)(input)
    }
}

/// Returns its input unchanged (LangChain `RunnablePassthrough`). Streams
/// pass straight through.
#[derive(Debug, Clone, Copy, Default)]
pub struct RunnablePassthrough;

/// [`RunnablePassthrough`].
pub fn passthrough() -> RunnablePassthrough {
    RunnablePassthrough
}

#[async_trait]
impl<I: Send + 'static> Runnable<I, I> for RunnablePassthrough {
    async fn invoke(&self, input: I) -> Result<I> {
        Ok(input)
    }
    async fn transform(&self, input: RunnableStream<I>) -> Result<RunnableStream<I>> {
        Ok(input)
    }
}

impl RunnablePassthrough {
    /// Starts a [`RunnableAssign`]: keeps the input object and adds keys.
    pub fn assign() -> RunnableAssign {
        RunnableAssign::default()
    }
}

/// A type-erased, cheaply cloneable runnable.
pub struct BoxRunnable<I, O>(Arc<dyn Runnable<I, O>>);

impl<I, O> Clone for BoxRunnable<I, O> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<I: Send + 'static, O: Send + 'static> BoxRunnable<I, O> {
    pub fn new(r: impl Runnable<I, O> + 'static) -> Self {
        Self(Arc::new(r))
    }
}

#[async_trait]
impl<I: Send + 'static, O: Send + 'static> Runnable<I, O> for BoxRunnable<I, O> {
    async fn invoke(&self, input: I) -> Result<O> {
        self.0.invoke(input).await
    }
    async fn batch_with_concurrency(&self, inputs: Vec<I>, n: usize) -> Vec<Result<O>> {
        self.0.batch_with_concurrency(inputs, n).await
    }
    async fn stream(&self, input: I) -> Result<RunnableStream<O>> {
        self.0.stream(input).await
    }
    async fn transform(&self, input: RunnableStream<I>) -> Result<RunnableStream<O>> {
        self.0.transform(input).await
    }
}

/// Serializes a runnable's output to JSON (used by [`RunnableMap`]).
struct ToJson<R, O>(R, PhantomData<fn() -> O>);

#[async_trait]
impl<I, O, R> Runnable<I, Value> for ToJson<R, O>
where
    I: Send + 'static,
    O: Serialize + Send + 'static,
    R: Runnable<I, O>,
{
    async fn invoke(&self, input: I) -> Result<Value> {
        Ok(serde_json::to_value(self.0.invoke(input).await?)?)
    }
}

/// Runs named runnables concurrently on (clones of) the same input and
/// collects their outputs into a JSON object (LangChain `RunnableParallel`
/// with a dict). For a fixed set of typed outputs use [`parallel`] instead.
pub struct RunnableMap<I> {
    entries: Vec<(String, BoxRunnable<I, Value>)>,
}

impl<I> Default for RunnableMap<I> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
        }
    }
}

impl<I: Clone + Send + Sync + 'static> RunnableMap<I> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a branch whose output (anything `Serialize`) lands under `key`.
    pub fn with<O, R>(mut self, key: impl Into<String>, r: R) -> Self
    where
        O: Serialize + Send + 'static,
        R: Runnable<I, O> + 'static,
    {
        self.entries
            .push((key.into(), BoxRunnable::new(ToJson(r, PhantomData))));
        self
    }

    async fn run(&self, input: &I) -> Result<serde_json::Map<String, Value>> {
        let futs = self
            .entries
            .iter()
            .map(|(k, r)| r.invoke(input.clone()).map_ok(move |v| (k.clone(), v)));
        Ok(futures_util::future::try_join_all(futs)
            .await?
            .into_iter()
            .collect())
    }
}

#[async_trait]
impl<I: Clone + Send + Sync + 'static> Runnable<I, Value> for RunnableMap<I> {
    async fn invoke(&self, input: I) -> Result<Value> {
        Ok(Value::Object(self.run(&input).await?))
    }
}

/// Keeps the input JSON object and adds keys computed concurrently from it
/// (LangChain `RunnablePassthrough.assign`). Non-object inputs are stored
/// under `"input"`.
#[derive(Default)]
pub struct RunnableAssign {
    map: RunnableMap<Value>,
}

impl RunnableAssign {
    pub fn with<O, R>(mut self, key: impl Into<String>, r: R) -> Self
    where
        O: Serialize + Send + 'static,
        R: Runnable<Value, O> + 'static,
    {
        self.map = self.map.with(key, r);
        self
    }
}

#[async_trait]
impl Runnable<Value, Value> for RunnableAssign {
    async fn invoke(&self, input: Value) -> Result<Value> {
        let added = self.map.run(&input).await?;
        let mut obj = match input {
            Value::Object(m) => m,
            other => {
                let mut m = serde_json::Map::new();
                m.insert("input".into(), other);
                m
            }
        };
        obj.extend(added);
        Ok(Value::Object(obj))
    }
}

/// Runs a tuple of runnables concurrently on clones of the input and returns
/// a tuple of their outputs; see [`parallel`].
pub struct Parallel<T>(T);

/// `parallel((a, b))` runs `a` and `b` concurrently on the same input and
/// yields `(a_out, b_out)`. Tuples of 2 to 5 runnables are supported; the
/// first error cancels the rest.
pub fn parallel<T>(runnables: T) -> Parallel<T> {
    Parallel(runnables)
}

macro_rules! parallel_impl {
    ($join:ident; $($r:ident $o:ident),+; $last_r:ident $last_o:ident) => {
        #[async_trait]
        #[allow(non_snake_case)]
        impl<I, $($r, $o,)+ $last_r, $last_o> Runnable<I, ($($o,)+ $last_o)> for Parallel<($($r,)+ $last_r)>
        where
            I: Clone + Send + Sync + 'static,
            $($r: Runnable<I, $o>, $o: Send + 'static,)+
            $last_r: Runnable<I, $last_o>,
            $last_o: Send + 'static,
        {
            async fn invoke(&self, input: I) -> Result<($($o,)+ $last_o)> {
                let ($($r,)+ $last_r) = &self.0;
                futures_util::future::$join($($r.invoke(input.clone()),)+ $last_r.invoke(input)).await
            }
        }
    };
}

parallel_impl!(try_join; A OA; B OB);
parallel_impl!(try_join3; A OA, B OB; C OC);
parallel_impl!(try_join4; A OA, B OB, C OC; D OD);
parallel_impl!(try_join5; A OA, B OB, C OC, D OD; E OE);

type Predicate<I> = Box<dyn Fn(&I) -> bool + Send + Sync>;

/// Routes the input to the first branch whose predicate matches, or to the
/// default (LangChain `RunnableBranch`).
pub struct RunnableBranch<I, O> {
    branches: Vec<(Predicate<I>, BoxRunnable<I, O>)>,
    default: BoxRunnable<I, O>,
}

impl<I: Send + 'static, O: Send + 'static> RunnableBranch<I, O> {
    pub fn new(default: impl Runnable<I, O> + 'static) -> Self {
        Self {
            branches: Vec::new(),
            default: BoxRunnable::new(default),
        }
    }

    /// Adds a branch, checked in insertion order.
    pub fn when(
        mut self,
        predicate: impl Fn(&I) -> bool + Send + Sync + 'static,
        r: impl Runnable<I, O> + 'static,
    ) -> Self {
        self.branches
            .push((Box::new(predicate), BoxRunnable::new(r)));
        self
    }

    fn pick(&self, input: &I) -> &BoxRunnable<I, O> {
        self.branches
            .iter()
            .find(|(p, _)| p(input))
            .map_or(&self.default, |(_, r)| r)
    }
}

#[async_trait]
impl<I: Send + Sync + 'static, O: Send + 'static> Runnable<I, O> for RunnableBranch<I, O> {
    async fn invoke(&self, input: I) -> Result<O> {
        self.pick(&input).invoke(input).await
    }
    async fn stream(&self, input: I) -> Result<RunnableStream<O>> {
        self.pick(&input).stream(input).await
    }
}

/// Retries the inner runnable per a [`RetryPolicy`]; built by
/// [`RunnableExt::retry`].
pub struct RetryRunnable<R, I, O> {
    inner: R,
    policy: RetryPolicy,
    _io: PhantomData<fn(I) -> O>,
}

#[async_trait]
impl<I, O, R> Runnable<I, O> for RetryRunnable<R, I, O>
where
    I: Clone + Send + Sync + 'static,
    O: Send + 'static,
    R: Runnable<I, O>,
{
    async fn invoke(&self, input: I) -> Result<O> {
        self.policy.run(|| self.inner.invoke(input.clone())).await
    }
}

/// Tries each runnable in turn until one succeeds; built by
/// [`RunnableExt::fallbacks`].
pub struct Fallbacks<I, O> {
    chain: Vec<BoxRunnable<I, O>>,
}

#[async_trait]
impl<I, O> Runnable<I, O> for Fallbacks<I, O>
where
    I: Clone + Send + Sync + 'static,
    O: Send + 'static,
{
    async fn invoke(&self, input: I) -> Result<O> {
        let mut last = None;
        for r in &self.chain {
            match r.invoke(input.clone()).await {
                Ok(o) => return Ok(o),
                Err(e) => last = Some(e),
            }
        }
        Err(last.unwrap_or_else(|| Error::Config("no runnables".into())))
    }
}

/// Combinators for every [`Runnable`].
pub trait RunnableExt<I, O>: Runnable<I, O> + Sized
where
    I: Send + 'static,
    O: Send + 'static,
{
    /// Feeds this step's output into `next` (LCEL `|`).
    fn pipe<B, O2>(self, next: B) -> Pipe<Self, B, I, O>
    where
        B: Runnable<O, O2>,
        O2: Send + 'static,
    {
        Pipe {
            first: self,
            second: next,
            _m: PhantomData,
        }
    }

    /// Transforms the output with a sync function.
    fn map<F, O2>(self, f: F) -> Map<Self, F, I, O>
    where
        F: Fn(O) -> O2 + Send + Sync,
        O2: Send + 'static,
    {
        Map {
            inner: self,
            f,
            _m: PhantomData,
        }
    }

    /// Feeds the output into an async closure.
    fn then<F, Fut, O2>(self, f: F) -> Pipe<Self, Lambda<F>, I, O>
    where
        F: Fn(O) -> Fut + Send + Sync,
        Fut: Future<Output = Result<O2>> + Send,
        O2: Send + 'static,
    {
        self.pipe(Lambda(f))
    }

    /// Retries failed invocations (input must be `Clone`).
    fn retry(self, policy: RetryPolicy) -> RetryRunnable<Self, I, O>
    where
        I: Clone + Sync,
    {
        RetryRunnable {
            inner: self,
            policy,
            _io: PhantomData,
        }
    }

    /// Falls back to `others`, in order, when this step fails.
    fn fallbacks(self, others: impl IntoIterator<Item = BoxRunnable<I, O>>) -> Fallbacks<I, O>
    where
        Self: 'static,
        I: Clone + Sync,
    {
        let mut chain = vec![BoxRunnable::new(self)];
        chain.extend(others);
        Fallbacks { chain }
    }

    /// Erases the type.
    fn boxed(self) -> BoxRunnable<I, O>
    where
        Self: 'static,
    {
        BoxRunnable::new(self)
    }
}

impl<I, O, R> RunnableExt<I, O> for R
where
    I: Send + 'static,
    O: Send + 'static,
    R: Runnable<I, O>,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{mock::ScriptedModel, StrOutputParser};
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    #[tokio::test]
    async fn prompt_model_parser_chain() {
        let model = Arc::new(ScriptedModel::new([Message::assistant("Ferris says hi")]));
        let chain = PromptTemplate::new("Greet {name}")
            .unwrap()
            .pipe(model.clone())
            .pipe(StrOutputParser);
        let out = chain.invoke(vars([("name", "Ferris")])).await.unwrap();
        assert_eq!(out, "Ferris says hi");
        assert_eq!(model.requests()[0].messages[0].content, "Greet Ferris");
    }

    #[tokio::test]
    async fn streaming_through_parser_yields_text_deltas() {
        let model = ScriptedModel::new([Message::assistant("one two three")]);
        let chain = ChatPromptTemplate::from_messages([(Role::User, "{q}")])
            .unwrap()
            .pipe(model)
            .pipe(StrOutputParser);
        let s = chain.stream(vars([("q", "count")])).await.unwrap();
        let chunks: Vec<String> = s.map(|c| c.unwrap()).collect().await;
        assert_eq!(chunks, vec!["one", " two", " three"]);
    }

    #[tokio::test]
    async fn stream_then_non_streaming_step_merges_chunks() {
        let model = ScriptedModel::new([Message::assistant("a b c")]);
        let chain = lambda_sync(|s: String| Ok(ChatRequest::new(vec![Message::user(s)])))
            .pipe(model)
            .pipe(lambda_sync(|r: ChatResponse| {
                Ok(format!("{}|{}", r.message.content.to_uppercase(), r.model))
            }));
        let s = chain.stream("x".to_string()).await.unwrap();
        let out: Vec<String> = s.map(|c| c.unwrap()).collect().await;
        assert_eq!(out, vec!["A B C|scripted"]);
    }

    #[tokio::test]
    async fn parallel_tuple_and_map() {
        let double = lambda_sync(|x: i64| Ok(x * 2));
        let square = lambda(|x: i64| async move { Ok(x * x) });
        let (d, s) = parallel((double, square)).invoke(7).await.unwrap();
        assert_eq!((d, s), (14, 49));

        let m = RunnableMap::new()
            .with("len", lambda_sync(|s: String| Ok(s.len())))
            .with("upper", lambda_sync(|s: String| Ok(s.to_uppercase())));
        let v = m.invoke("abc".to_string()).await.unwrap();
        assert_eq!(v, json!({"len": 3, "upper": "ABC"}));
    }

    #[tokio::test]
    async fn parallel_branches_run_concurrently() {
        let slow = |n: u64| {
            lambda(move |_: ()| async move {
                tokio::time::sleep(Duration::from_millis(100)).await;
                Ok(n)
            })
        };
        let t = std::time::Instant::now();
        let out = parallel((slow(1), slow(2), slow(3)))
            .invoke(())
            .await
            .unwrap();
        assert_eq!(out, (1, 2, 3));
        assert!(t.elapsed() < Duration::from_millis(250));
    }

    #[tokio::test]
    async fn assign_and_passthrough() {
        let chain = RunnablePassthrough::assign().with(
            "n",
            lambda_sync(|v: Value| Ok(v["text"].as_str().unwrap_or("").len())),
        );
        let v = chain.invoke(json!({"text": "hello"})).await.unwrap();
        assert_eq!(v, json!({"text": "hello", "n": 5}));
        let p: i32 = Runnable::<i32, i32>::invoke(&passthrough(), 3)
            .await
            .unwrap();
        assert_eq!(p, 3);
    }

    #[tokio::test]
    async fn branch_routes_by_predicate() {
        let b = RunnableBranch::new(lambda_sync(|s: String| Ok(format!("other:{s}")))).when(
            |s: &String| s.starts_with("math"),
            lambda_sync(|_s: String| Ok("math".to_string())),
        );
        assert_eq!(b.invoke("math 1+1".into()).await.unwrap(), "math");
        assert_eq!(b.invoke("poem".into()).await.unwrap(), "other:poem");
    }

    #[tokio::test]
    async fn batch_preserves_order_and_bounds_concurrency() {
        let live = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let (l, p) = (live.clone(), peak.clone());
        let r = lambda(move |x: u32| {
            let (l, p) = (l.clone(), p.clone());
            async move {
                let now = l.fetch_add(1, Ordering::SeqCst) + 1;
                p.fetch_max(now, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(10)).await;
                l.fetch_sub(1, Ordering::SeqCst);
                Ok(x * 10)
            }
        });
        let out = r.batch_with_concurrency((0..20).collect(), 3).await;
        let out: Vec<u32> = out.into_iter().map(|r| r.unwrap()).collect();
        assert_eq!(out, (0..20).map(|x| x * 10).collect::<Vec<_>>());
        assert!(peak.load(Ordering::SeqCst) <= 3);
    }

    #[tokio::test]
    async fn retry_and_fallbacks_on_runnables() {
        let calls = Arc::new(AtomicUsize::new(0));
        let c = calls.clone();
        let flaky = lambda_sync(move |x: u32| {
            if c.fetch_add(1, Ordering::SeqCst) < 2 {
                Err(Error::Timeout(Duration::from_millis(1)))
            } else {
                Ok(x)
            }
        });
        let policy = RetryPolicy::new(3).with_initial_backoff(Duration::from_millis(1));
        assert_eq!(flaky.retry(policy).invoke(5).await.unwrap(), 5);
        assert_eq!(calls.load(Ordering::SeqCst), 3);

        let broken = lambda_sync(|_: u32| -> Result<u32> { Err(Error::Other("nope".into())) });
        let chain = broken.fallbacks([lambda_sync(|x: u32| Ok(x + 1)).boxed()]);
        assert_eq!(chain.invoke(1).await.unwrap(), 2);
    }

    #[tokio::test]
    async fn structured_runnable_in_a_chain() {
        #[derive(serde::Deserialize, Debug, PartialEq)]
        struct City {
            name: String,
        }
        let model = ScriptedModel::new([Message::assistant_with_tools(
            "",
            vec![crate::ToolCall {
                id: "1".into(),
                name: "city".into(),
                arguments: json!({"name": "Rio"}),
            }],
        )]);
        let schema = json!({"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]});
        let chain = PromptTemplate::new("Capital-ish city of {country}?")
            .unwrap()
            .pipe(StructuredOutput::new("city", schema).runnable::<City, _>(model));
        let c = chain.invoke(json!({"country": "Brazil"})).await.unwrap();
        assert_eq!(c, City { name: "Rio".into() });
    }

    #[test]
    fn merging_chunks_by_type() {
        assert_eq!(merge_vec(vec!["a".to_string(), "b".into()]), "ab");
        assert_eq!(merge_vec(vec![1, 2, 3]), 3);
        let r = merge_responses(vec![
            chunk(Message::assistant("he"), Usage::default(), String::new()),
            chunk(Message::assistant("y"), Usage::new(3, 2), String::new()),
            chunk(Message::assistant(""), Usage::default(), "m".into()),
        ]);
        assert_eq!(r.message.content, "hey");
        assert_eq!(r.usage, Usage::new(3, 2));
        assert_eq!(r.model, "m");
    }
}
