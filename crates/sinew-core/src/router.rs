//! Per-request model routing: send each request to a cheap, fast or strong
//! model by rules, by a classifier model, or by trying the cheap model first
//! and escalating when its answer fails a check.
//!
//! A [`Router`] is itself a [`ChatModel`], so it works anywhere a model does
//! (agents, graphs, structured output, runnables, the resilience wrappers).
//!
//! ```
//! # use std::sync::Arc;
//! # use sinew_core::{mock::ScriptedModel, router::{Router, Validator}, *};
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn main() -> Result<()> {
//! let cheap = Arc::new(ScriptedModel::new([Message::assistant("I'm sorry, I can't help with that.")]));
//! let strong = Arc::new(ScriptedModel::new([Message::assistant("Here is the proof.")]));
//!
//! let router = Router::builder()
//!     .route("cheap", cheap)
//!     .route("strong", strong)
//!     // Long prompts go straight to the strong model.
//!     .rule("long-input", |r| r.estimated_tokens() > 8_000, "strong")
//!     .default_route("cheap")
//!     // Otherwise try cheap first and escalate on an empty answer or a refusal.
//!     .cascade(["cheap", "strong"], Validator::no_refusal())
//!     .build()?;
//!
//! let resp = router.chat("Prove that sqrt(2) is irrational.".into()).await?;
//! assert_eq!(resp.message.content, "Here is the proof.");
//! assert_eq!(router.escalations(), 1);
//! # Ok(()) }
//! ```
//!
//! How a route is chosen for one request:
//!
//! 1. The first [rule](RouterBuilder::rule) whose predicate matches names the
//!    preferred routes.
//! 2. Otherwise the [classifier](Classifier), if any, labels the request and
//!    the label names the preferred routes. A classifier error falls back to
//!    the default routes.
//! 3. Otherwise the [default routes](RouterBuilder::default_route) are
//!    preferred (the first registered route when none is set).
//! 4. Preferred routes that cannot serve the request (images, PDFs, audio or
//!    tools the model does not support, see [`Capabilities`]) or whose circuit
//!    breaker is open are skipped. Among the rest, the route with the lowest
//!    recent latency wins. When none is left, the default routes and then all
//!    routes are tried the same way, and the decision is marked
//!    [`fallback`](RouteDecision::fallback).
//!
//! See `docs/routing.md` for the full guide.

use std::{
    collections::BTreeMap,
    fmt,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::time::Instant;

use crate::{
    message::ContentPart, pricing::estimate_request_tokens, stream::ChatStream, Capabilities,
    ChatModel, ChatRequest, ChatResponse, Error, Message, Result, Role, StructuredMode,
    StructuredOutput,
};

type Predicate = Arc<dyn Fn(&RouteInput<'_>) -> bool + Send + Sync>;
type ErrorPredicate = Arc<dyn Fn(&Error) -> bool + Send + Sync>;
type RouteCallback = Arc<dyn Fn(&RouteDecision) + Send + Sync>;

/// Caller-supplied facts about a request that are not part of the
/// [`ChatRequest`], such as the tenant's plan. Rules read them through
/// [`RouteInput::hint`]; pass them with [`Router::chat_with_hints`].
///
/// ```
/// use sinew_core::router::RouteHints;
///
/// let hints = RouteHints::new().with("tier", "enterprise");
/// assert_eq!(hints.get("tier"), Some("enterprise"));
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RouteHints {
    values: BTreeMap<String, String>,
}

impl RouteHints {
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets `key` to `value`, replacing any previous value.
    pub fn with(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.values.insert(key.into(), value.into());
        self
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(String::as_str)
    }
}

/// What a request needs from the model that serves it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Needs {
    tools: bool,
    vision: bool,
    pdf: bool,
    audio: bool,
}

impl Needs {
    fn of(req: &ChatRequest) -> Self {
        let mut n = Needs {
            tools: !req.tools.is_empty(),
            ..Default::default()
        };
        for p in req.messages.iter().flat_map(|m| &m.parts) {
            match p {
                ContentPart::Image { .. } => n.vision = true,
                ContentPart::Document { .. } => n.pdf = true,
                ContentPart::Audio { .. } => n.audio = true,
                ContentPart::Text { .. } => {}
            }
        }
        n
    }

    fn met_by(&self, c: &Capabilities) -> bool {
        (!self.tools || c.tools)
            && (!self.vision || c.vision)
            && (!self.pdf || c.pdf)
            && (!self.audio || c.audio)
    }
}

impl fmt::Display for Needs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        if self.tools {
            parts.push("tools");
        }
        if self.vision {
            parts.push("images");
        }
        if self.pdf {
            parts.push("documents");
        }
        if self.audio {
            parts.push("audio");
        }
        if parts.is_empty() {
            f.write_str("plain text")
        } else {
            f.write_str(&parts.join(", "))
        }
    }
}

/// What a routing rule sees: the request, the caller's [`RouteHints`] and a
/// few precomputed facts.
pub struct RouteInput<'a> {
    request: &'a ChatRequest,
    hints: &'a RouteHints,
    estimated_tokens: u32,
    needs: Needs,
}

impl RouteInput<'_> {
    pub fn request(&self) -> &ChatRequest {
        self.request
    }
    pub fn hints(&self) -> &RouteHints {
        self.hints
    }
    /// Shortcut for `hints().get(key)`.
    pub fn hint(&self, key: &str) -> Option<&str> {
        self.hints.get(key)
    }
    /// Prompt size from [`estimate_request_tokens`] (no tokenizer).
    pub fn estimated_tokens(&self) -> u32 {
        self.estimated_tokens
    }
    /// The request offers tools.
    pub fn has_tools(&self) -> bool {
        self.needs.tools
    }
    /// Some message carries an image.
    pub fn has_images(&self) -> bool {
        self.needs.vision
    }
    /// Some message carries a document (PDF).
    pub fn has_documents(&self) -> bool {
        self.needs.pdf
    }
    /// Some message carries audio.
    pub fn has_audio(&self) -> bool {
        self.needs.audio
    }
}

/// Why a route was chosen.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RouteReason {
    /// The named rule matched.
    Rule { name: String },
    /// The classifier returned this label.
    Classifier { label: String },
    /// The classifier failed; the default routes were used.
    ClassifierError { message: String },
    /// No rule matched and there is no classifier.
    Default,
    /// The cascade escalated from route `from` after its answer failed the
    /// validator or its call failed.
    Escalation { from: String },
}

/// One routing decision, passed to [`RouterBuilder::on_route`] for every
/// model call the router makes (the first call and each escalation).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteDecision {
    /// The route name.
    pub route: String,
    /// The [`ChatModel::name`] of the route's model.
    pub model: String,
    pub reason: RouteReason,
    /// The preferred routes could not serve the request (missing capability
    /// or open circuit), so a default or other route was used instead.
    pub fallback: bool,
}

/// Circuit breaker settings shared by every route of a [`Router`].
///
/// After `failure_threshold` consecutive counted failures a route is skipped
/// for `cool_down`. Once the cool-down has passed the route is half-open: the
/// next call goes through, a success closes the circuit and a counted failure
/// opens it again for another `cool_down`. Concurrent requests during the
/// half-open state may all try the route.
#[derive(Clone)]
pub struct CircuitBreaker {
    failure_threshold: u32,
    cool_down: Duration,
    count_if: ErrorPredicate,
}

impl fmt::Debug for CircuitBreaker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CircuitBreaker")
            .field("failure_threshold", &self.failure_threshold)
            .field("cool_down", &self.cool_down)
            .finish()
    }
}

impl Default for CircuitBreaker {
    /// 5 consecutive failures open the circuit for 30 s; only
    /// [`Error::is_retryable`] errors count.
    fn default() -> Self {
        Self::new(5, Duration::from_secs(30))
    }
}

impl CircuitBreaker {
    pub fn new(failure_threshold: u32, cool_down: Duration) -> Self {
        Self {
            failure_threshold,
            cool_down,
            count_if: Arc::new(Error::is_retryable),
        }
    }

    /// Never opens.
    pub fn disabled() -> Self {
        Self::new(0, Duration::ZERO)
    }

    /// Which errors count as failures (default: [`Error::is_retryable`], so a
    /// 400 caused by the request itself does not open the circuit).
    pub fn failure_if(mut self, f: impl Fn(&Error) -> bool + Send + Sync + 'static) -> Self {
        self.count_if = Arc::new(f);
        self
    }
}

/// Circuit state of one route, as reported by [`Router::stats`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,
    /// Skipped for `remaining` more.
    Open {
        remaining: Duration,
    },
    /// The cool-down has passed; the next call decides.
    HalfOpen,
}

/// Counters for one route, from [`Router::stats`].
#[derive(Debug, Clone, PartialEq)]
pub struct RouteStats {
    pub name: String,
    /// Calls that succeeded.
    pub served: u64,
    /// Calls that failed (counted by the breaker or not).
    pub failures: u64,
    pub consecutive_failures: u32,
    pub circuit: CircuitState,
    /// Exponentially weighted moving average of successful `chat` latency;
    /// `None` until the first success.
    pub latency: Option<Duration>,
}

#[derive(Default)]
struct HealthState {
    consecutive_failures: u32,
    open_until: Option<Instant>,
    ewma: Option<Duration>,
}

#[derive(Default)]
struct Health {
    state: Mutex<HealthState>,
    served: AtomicU64,
    failures: AtomicU64,
}

impl Health {
    fn lock(&self) -> std::sync::MutexGuard<'_, HealthState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// `(circuit open, latency ewma)` at `now`.
    fn snapshot(&self, now: Instant) -> (bool, Option<Duration>) {
        let s = self.lock();
        (s.open_until.is_some_and(|t| now < t), s.ewma)
    }
}

struct RouteEntry {
    name: String,
    model: Arc<dyn ChatModel>,
    capabilities: Capabilities,
    health: Health,
}

/// Checks a cascade step's answer: `true` accepts it, `false` escalates to
/// the next route. Combine checks with [`and`](Self::and).
///
/// ```
/// use sinew_core::{router::Validator, ChatResponse, Message, Usage};
///
/// let resp = |text: &str| ChatResponse {
///     message: Message::assistant(text),
///     usage: Usage::default(),
///     model: "m".into(),
/// };
/// let v = Validator::no_refusal().and(Validator::new(|r| r.message.content.len() < 100));
/// assert!(v.check(&resp("Paris.")));
/// assert!(!v.check(&resp("I'm sorry, but I can't help with that.")));
/// assert!(!v.check(&resp("   ")));
/// ```
#[derive(Clone)]
pub struct Validator(Arc<dyn Fn(&ChatResponse) -> bool + Send + Sync>);

impl fmt::Debug for Validator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Validator")
    }
}

impl Validator {
    pub fn new(f: impl Fn(&ChatResponse) -> bool + Send + Sync + 'static) -> Self {
        Self(Arc::new(f))
    }

    /// Accepts any answer with non-whitespace text or at least one tool call.
    pub fn non_empty() -> Self {
        Self::new(|r| !r.message.text().trim().is_empty() || r.message.has_tool_calls())
    }

    /// [`non_empty`](Self::non_empty), and the text does not start like a
    /// refusal ("I can't", "I cannot", "I'm sorry", "I am unable", "As an
    /// AI", ...; case-insensitive). A heuristic for English replies; write
    /// your own check for other languages.
    pub fn no_refusal() -> Self {
        Self::new(|r| {
            if r.message.has_tool_calls() {
                return true;
            }
            let text = r.message.text();
            let text = text.trim();
            !text.is_empty() && !looks_like_refusal(text)
        })
    }

    /// Accepts answers that `output` can parse and validate
    /// ([`StructuredOutput::parse_message`]).
    pub fn structured(output: StructuredOutput) -> Self {
        Self::new(move |r| output.parse_message::<Value>(&r.message).is_ok())
    }

    /// Both checks must accept.
    pub fn and(self, other: Validator) -> Self {
        Self::new(move |r| (self.0)(r) && (other.0)(r))
    }

    pub fn check(&self, response: &ChatResponse) -> bool {
        (self.0)(response)
    }
}

fn looks_like_refusal(text: &str) -> bool {
    const PREFIXES: &[&str] = &[
        "i can't",
        "i cannot",
        "i can not",
        "i won't",
        "i will not",
        "i'm sorry",
        "i am sorry",
        "sorry, i",
        "sorry, but",
        "i'm unable",
        "i am unable",
        "i'm not able",
        "i am not able",
        "unfortunately, i",
        "as an ai",
    ];
    let head: String = text
        .chars()
        .take(40)
        .map(|c| if c == '\u{2019}' { '\'' } else { c })
        .collect::<String>()
        .to_lowercase();
    PREFIXES.iter().any(|p| head.starts_with(p))
}

struct Label {
    name: String,
    description: String,
    route: String,
}

/// Labels a request with a cheap model, and maps the label to a route.
///
/// The classifier sees the text of the last user message (truncated to
/// [`with_max_input_chars`](Self::with_max_input_chars)) and answers with
/// structured output `{"label": "..."}` restricted to the configured labels:
/// a forced tool call when the model reports tool support, otherwise a JSON
/// reply. Nothing is cached; every routed request that no rule matched costs
/// one classifier call. Wrap the classifier model with
/// [`ChatModelExt::with_cache`](crate::ChatModelExt::with_cache) if you want
/// caching.
///
/// ```
/// # use std::sync::Arc;
/// # use sinew_core::{mock::ScriptedModel, router::Classifier, ChatModel};
/// # let small: Arc<dyn ChatModel> = Arc::new(ScriptedModel::new([]));
/// let classifier = Classifier::new(small)
///     .label("easy", "greetings, lookups, short rewrites", "cheap")
///     .label("hard", "multi-step reasoning, math, code", "strong");
/// ```
pub struct Classifier {
    model: Arc<dyn ChatModel>,
    labels: Vec<Label>,
    instructions: String,
    max_input_chars: usize,
}

impl Classifier {
    pub fn new(model: Arc<dyn ChatModel>) -> Self {
        Self {
            model,
            labels: Vec::new(),
            instructions: "Classify how difficult the user's request is to answer well. \
                           Pick exactly one label."
                .into(),
            max_input_chars: 4_000,
        }
    }

    /// Adds a label, its description for the classifier prompt, and the
    /// route it selects.
    pub fn label(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        route: impl Into<String>,
    ) -> Self {
        self.labels.push(Label {
            name: name.into(),
            description: description.into(),
            route: route.into(),
        });
        self
    }

    /// Replaces the first line of the classifier's system prompt (the label
    /// list is always appended).
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = instructions.into();
        self
    }

    /// How much of the user message the classifier sees (default 4000
    /// characters).
    pub fn with_max_input_chars(mut self, n: usize) -> Self {
        self.max_input_chars = n;
        self
    }

    fn output(&self) -> StructuredOutput {
        let names: Vec<&str> = self.labels.iter().map(|l| l.name.as_str()).collect();
        let schema = json!({
            "type": "object",
            "properties": { "label": { "type": "string", "enum": names } },
            "required": ["label"]
        });
        let mode = if self.model.capabilities().tools {
            StructuredMode::Tool
        } else {
            StructuredMode::Json
        };
        StructuredOutput::new("route_label", schema)
            .with_mode(mode)
            .with_retries(0)
            .with_description("The label for the request.")
    }

    /// Asks the classifier model for a label.
    pub async fn classify(&self, request: &ChatRequest) -> Result<String> {
        let mut system = self.instructions.clone();
        system.push_str("\n\nLabels:");
        for l in &self.labels {
            system.push_str(&format!("\n- {}: {}", l.name, l.description));
        }
        let text: String = request
            .messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .map(|m| m.text())
            .unwrap_or_default()
            .chars()
            .take(self.max_input_chars)
            .collect();
        let req = ChatRequest::new(vec![
            Message::system(system),
            Message::user(format!("Request to classify:\n\n{text}")),
        ]);
        #[derive(serde::Deserialize)]
        struct Out {
            label: String,
        }
        let out: Out = self.output().generate(&*self.model, req).await?;
        Ok(out.label)
    }
}

struct Rule {
    name: String,
    predicate: Predicate,
    targets: Vec<String>,
}

struct ResolvedRule {
    name: String,
    predicate: Predicate,
    targets: Vec<usize>,
}

struct ResolvedClassifier {
    classifier: Classifier,
    targets: Vec<(String, usize)>,
}

struct Cascade {
    ladder: Vec<usize>,
    validator: Validator,
}

/// Builds a [`Router`]; see [`Router::builder`]. Route names given to rules,
/// labels, the default and the cascade are checked by [`build`](Self::build).
pub struct RouterBuilder {
    name: String,
    routes: Vec<(String, Arc<dyn ChatModel>, Option<Capabilities>)>,
    rules: Vec<Rule>,
    classifier: Option<Classifier>,
    default: Vec<String>,
    cascade: Option<(Vec<String>, Validator)>,
    breaker: CircuitBreaker,
    latency_alpha: f64,
    on_route: Option<RouteCallback>,
}

impl RouterBuilder {
    /// Name reported by [`ChatModel::name`] (default `"router"`). Responses
    /// keep the serving model's name in [`ChatResponse::model`].
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Adds a named route. Its [`ChatModel::capabilities`] decide which
    /// requests it may serve.
    pub fn route(mut self, name: impl Into<String>, model: Arc<dyn ChatModel>) -> Self {
        self.routes.push((name.into(), model, None));
        self
    }

    /// Adds a route with explicit capabilities, for models that report
    /// nothing (custom `ChatModel`s default to [`Capabilities::none`], which
    /// keeps them away from requests with tools or media).
    pub fn route_with_capabilities(
        mut self,
        name: impl Into<String>,
        model: Arc<dyn ChatModel>,
        capabilities: Capabilities,
    ) -> Self {
        self.routes.push((name.into(), model, Some(capabilities)));
        self
    }

    /// Sends requests matching `predicate` to `route`. Rules are tried in the
    /// order they were added; the first match wins.
    pub fn rule(
        self,
        name: impl Into<String>,
        predicate: impl Fn(&RouteInput<'_>) -> bool + Send + Sync + 'static,
        route: impl Into<String>,
    ) -> Self {
        self.rule_any(name, predicate, [route])
    }

    /// Like [`rule`](Self::rule) with several equivalent routes; the router
    /// picks the healthy, capable one with the lowest recent latency.
    pub fn rule_any(
        mut self,
        name: impl Into<String>,
        predicate: impl Fn(&RouteInput<'_>) -> bool + Send + Sync + 'static,
        routes: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.rules.push(Rule {
            name: name.into(),
            predicate: Arc::new(predicate),
            targets: routes.into_iter().map(Into::into).collect(),
        });
        self
    }

    /// Labels requests that no rule matched; see [`Classifier`].
    pub fn classifier(mut self, classifier: Classifier) -> Self {
        self.classifier = Some(classifier);
        self
    }

    /// The route used when no rule matches and there is no classifier (or it
    /// failed). Defaults to the first registered route.
    pub fn default_route(self, route: impl Into<String>) -> Self {
        self.default_routes([route])
    }

    /// Several equivalent default routes, chosen by health and latency.
    pub fn default_routes(mut self, routes: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.default = routes.into_iter().map(Into::into).collect();
        self
    }

    /// Escalation ladder for [`ChatModel::chat`]: when the chosen route is in
    /// `ladder` and its answer fails `validator` (or its call fails), the
    /// request is sent to the next route in the ladder, and so on. Routes
    /// that cannot serve the request or have an open circuit are skipped.
    /// Streams are not cascaded.
    pub fn cascade(
        mut self,
        ladder: impl IntoIterator<Item = impl Into<String>>,
        validator: Validator,
    ) -> Self {
        self.cascade = Some((ladder.into_iter().map(Into::into).collect(), validator));
        self
    }

    /// Circuit breaker for every route (default [`CircuitBreaker::default`]).
    pub fn circuit_breaker(mut self, breaker: CircuitBreaker) -> Self {
        self.breaker = breaker;
        self
    }

    /// Weight of the newest sample in the latency moving average, in
    /// `(0, 1]` (default 0.3).
    pub fn latency_alpha(mut self, alpha: f64) -> Self {
        self.latency_alpha = alpha;
        self
    }

    /// Called with every [`RouteDecision`], before the routed call. Use it to
    /// log, count or attach the route to your trace.
    pub fn on_route(mut self, f: impl Fn(&RouteDecision) + Send + Sync + 'static) -> Self {
        self.on_route = Some(Arc::new(f));
        self
    }

    /// Checks the configuration. Fails with [`Error::Config`] when there are
    /// no routes, a route name is repeated or unknown, a classifier has no
    /// labels, or `latency_alpha` is outside `(0, 1]`.
    pub fn build(self) -> Result<Router> {
        let cfg = |m: String| Error::Config(format!("router `{}`: {m}", self.name));
        if self.routes.is_empty() {
            return Err(cfg("no routes".into()));
        }
        if !(self.latency_alpha > 0.0 && self.latency_alpha <= 1.0) {
            return Err(cfg(format!(
                "latency_alpha must be in (0, 1], got {}",
                self.latency_alpha
            )));
        }
        let mut routes: Vec<RouteEntry> = Vec::with_capacity(self.routes.len());
        for (name, model, caps) in self.routes {
            if routes.iter().any(|r| r.name == name) {
                return Err(cfg(format!("route `{name}` is defined twice")));
            }
            let capabilities = caps.unwrap_or_else(|| model.capabilities());
            routes.push(RouteEntry {
                name,
                model,
                capabilities,
                health: Health::default(),
            });
        }
        let index = |name: &str| {
            routes
                .iter()
                .position(|r| r.name == name)
                .ok_or_else(|| cfg(format!("unknown route `{name}`")))
        };
        let resolve = |names: &[String], what: &str| -> Result<Vec<usize>> {
            if names.is_empty() {
                return Err(cfg(format!("{what} names no route")));
            }
            names.iter().map(|n| index(n)).collect()
        };
        let rules = self
            .rules
            .into_iter()
            .map(|r| {
                Ok(ResolvedRule {
                    targets: resolve(&r.targets, &format!("rule `{}`", r.name))?,
                    name: r.name,
                    predicate: r.predicate,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let default = if self.default.is_empty() {
            vec![0]
        } else {
            resolve(&self.default, "the default")?
        };
        let classifier = match self.classifier {
            None => None,
            Some(c) => {
                if c.labels.is_empty() {
                    return Err(cfg("the classifier has no labels".into()));
                }
                let targets = c
                    .labels
                    .iter()
                    .map(|l| Ok((l.name.clone(), index(&l.route)?)))
                    .collect::<Result<Vec<_>>>()?;
                Some(ResolvedClassifier {
                    classifier: c,
                    targets,
                })
            }
        };
        let cascade = match self.cascade {
            None => None,
            Some((ladder, validator)) => Some(Cascade {
                ladder: resolve(&ladder, "the cascade")?,
                validator,
            }),
        };
        Ok(Router {
            name: self.name,
            routes,
            rules,
            classifier,
            default,
            cascade,
            breaker: self.breaker,
            latency_alpha: self.latency_alpha,
            on_route: self.on_route,
            escalations: AtomicU64::new(0),
        })
    }
}

/// A [`ChatModel`] that picks one of several named models per request. See
/// the [module docs](self).
///
/// - [`capabilities`](ChatModel::capabilities) is the union of the routes'
///   capabilities: the router accepts a request when at least one route can
///   serve it, and only sends it to such a route.
/// - [`chat_stream`](ChatModel::chat_stream) routes once and delegates; the
///   cascade applies only to [`chat`](ChatModel::chat).
/// - The router does not retry. Wrap routes in
///   [`with_retry`](crate::ChatModelExt::with_retry) or the router in
///   [`with_fallbacks`](crate::ChatModelExt::with_fallbacks) as needed.
pub struct Router {
    name: String,
    routes: Vec<RouteEntry>,
    rules: Vec<ResolvedRule>,
    classifier: Option<ResolvedClassifier>,
    default: Vec<usize>,
    cascade: Option<Cascade>,
    breaker: CircuitBreaker,
    latency_alpha: f64,
    on_route: Option<RouteCallback>,
    escalations: AtomicU64,
}

impl fmt::Debug for Router {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Router")
            .field("name", &self.name)
            .field(
                "routes",
                &self.routes.iter().map(|r| &r.name).collect::<Vec<_>>(),
            )
            .field(
                "rules",
                &self.rules.iter().map(|r| &r.name).collect::<Vec<_>>(),
            )
            .field("classifier", &self.classifier.is_some())
            .field("cascade", &self.cascade.is_some())
            .finish()
    }
}

impl Router {
    pub fn builder() -> RouterBuilder {
        RouterBuilder {
            name: "router".into(),
            routes: Vec::new(),
            rules: Vec::new(),
            classifier: None,
            default: Vec::new(),
            cascade: None,
            breaker: CircuitBreaker::default(),
            latency_alpha: 0.3,
            on_route: None,
        }
    }

    /// How many times the cascade escalated to a stronger route.
    pub fn escalations(&self) -> u64 {
        self.escalations.load(Ordering::Relaxed)
    }

    /// Counters, circuit state and latency for every route, in registration
    /// order.
    pub fn stats(&self) -> Vec<RouteStats> {
        let now = Instant::now();
        self.routes
            .iter()
            .map(|r| {
                let s = r.health.lock();
                let circuit = match s.open_until {
                    Some(t) if now < t => CircuitState::Open { remaining: t - now },
                    Some(_) => CircuitState::HalfOpen,
                    None => CircuitState::Closed,
                };
                RouteStats {
                    name: r.name.clone(),
                    served: r.health.served.load(Ordering::Relaxed),
                    failures: r.health.failures.load(Ordering::Relaxed),
                    consecutive_failures: s.consecutive_failures,
                    circuit,
                    latency: s.ewma,
                }
            })
            .collect()
    }

    /// Decides which route would serve `request`, without calling it. Runs
    /// the classifier when it is consulted.
    pub async fn route(&self, request: &ChatRequest, hints: &RouteHints) -> Result<RouteDecision> {
        self.decide(request, hints).await.map(|(_, d)| d)
    }

    /// [`chat`](ChatModel::chat) with caller-supplied [`RouteHints`].
    pub async fn chat_with_hints(
        &self,
        request: ChatRequest,
        hints: &RouteHints,
    ) -> Result<ChatResponse> {
        let (idx, decision) = self.decide(&request, hints).await?;
        self.emit(&decision);
        let mut current = idx;
        let mut result = self.call(idx, request.clone()).await;
        let Some(cascade) = &self.cascade else {
            return result;
        };
        let Some(pos) = cascade.ladder.iter().position(|&i| i == idx) else {
            return result;
        };
        let needs = Needs::of(&request);
        for &next in &cascade.ladder[pos + 1..] {
            if matches!(&result, Ok(r) if cascade.validator.check(r)) {
                break;
            }
            if !self.available(next, needs, Instant::now()) {
                continue;
            }
            self.escalations.fetch_add(1, Ordering::Relaxed);
            self.emit(&RouteDecision {
                route: self.routes[next].name.clone(),
                model: self.routes[next].model.name().to_string(),
                reason: RouteReason::Escalation {
                    from: self.routes[current].name.clone(),
                },
                fallback: false,
            });
            let attempt = self.call(next, request.clone()).await;
            result = match (result, attempt) {
                (Ok(previous), Err(_)) => Ok(previous),
                (_, attempt) => attempt,
            };
            current = next;
        }
        result
    }

    /// [`chat_stream`](ChatModel::chat_stream) with caller-supplied
    /// [`RouteHints`].
    pub async fn chat_stream_with_hints(
        &self,
        request: ChatRequest,
        hints: &RouteHints,
    ) -> Result<ChatStream> {
        let (idx, decision) = self.decide(&request, hints).await?;
        self.emit(&decision);
        let res = self.routes[idx].model.chat_stream(request).await;
        self.record(idx, res.as_ref().err(), None);
        res
    }

    fn emit(&self, decision: &RouteDecision) {
        if let Some(f) = &self.on_route {
            f(decision);
        }
    }

    fn available(&self, idx: usize, needs: Needs, now: Instant) -> bool {
        let r = &self.routes[idx];
        needs.met_by(&r.capabilities) && !r.health.snapshot(now).0
    }

    /// The available candidate with the lowest latency (unknown counts as
    /// zero, so new routes get tried); ties go to the earlier candidate.
    fn pick(&self, candidates: &[usize], needs: Needs, now: Instant) -> Option<usize> {
        candidates
            .iter()
            .filter_map(|&i| {
                let r = &self.routes[i];
                if !needs.met_by(&r.capabilities) {
                    return None;
                }
                let (open, ewma) = r.health.snapshot(now);
                (!open).then_some((i, ewma.unwrap_or(Duration::ZERO)))
            })
            .min_by_key(|&(_, latency)| latency)
            .map(|(i, _)| i)
    }

    async fn decide(
        &self,
        request: &ChatRequest,
        hints: &RouteHints,
    ) -> Result<(usize, RouteDecision)> {
        let needs = Needs::of(request);
        let input = RouteInput {
            request,
            hints,
            estimated_tokens: estimate_request_tokens(request),
            needs,
        };
        let (preferred, reason): (&[usize], RouteReason) =
            if let Some(rule) = self.rules.iter().find(|r| (r.predicate)(&input)) {
                (
                    &rule.targets,
                    RouteReason::Rule {
                        name: rule.name.clone(),
                    },
                )
            } else if let Some(c) = &self.classifier {
                match c.classifier.classify(request).await {
                    Ok(label) => match c.targets.iter().find(|(l, _)| *l == label) {
                        Some((_, idx)) => {
                            (std::slice::from_ref(idx), RouteReason::Classifier { label })
                        }
                        None => (
                            &self.default,
                            RouteReason::ClassifierError {
                                message: format!("unknown label `{label}`"),
                            },
                        ),
                    },
                    Err(e) => (
                        &self.default,
                        RouteReason::ClassifierError {
                            message: e.to_string(),
                        },
                    ),
                }
            } else {
                (&self.default, RouteReason::Default)
            };

        let now = Instant::now();
        let all: Vec<usize> = (0..self.routes.len()).collect();
        let chosen = self
            .pick(preferred, needs, now)
            .map(|i| (i, false))
            .or_else(|| self.pick(&self.default, needs, now).map(|i| (i, true)))
            .or_else(|| self.pick(&all, needs, now).map(|i| (i, true)));
        let Some((idx, fallback)) = chosen else {
            return Err(
                if self.routes.iter().any(|r| needs.met_by(&r.capabilities)) {
                    Error::Other(format!(
                        "router `{}`: every route that can serve this request has an open circuit",
                        self.name
                    ))
                } else {
                    Error::Config(format!(
                        "router `{}`: no route supports a request with {needs}",
                        self.name
                    ))
                },
            );
        };
        let r = &self.routes[idx];
        Ok((
            idx,
            RouteDecision {
                route: r.name.clone(),
                model: r.model.name().to_string(),
                reason,
                fallback,
            },
        ))
    }

    async fn call(&self, idx: usize, request: ChatRequest) -> Result<ChatResponse> {
        let start = Instant::now();
        let res = self.routes[idx].model.chat(request).await;
        self.record(idx, res.as_ref().err(), Some(start.elapsed()));
        res
    }

    /// Updates health after a call; `latency` is recorded on success only.
    fn record(&self, idx: usize, err: Option<&Error>, latency: Option<Duration>) {
        let h = &self.routes[idx].health;
        let mut s = h.lock();
        match err {
            None => {
                h.served.fetch_add(1, Ordering::Relaxed);
                s.consecutive_failures = 0;
                s.open_until = None;
                if let Some(l) = latency {
                    s.ewma = Some(match s.ewma {
                        None => l,
                        Some(prev) => {
                            prev.mul_f64(1.0 - self.latency_alpha) + l.mul_f64(self.latency_alpha)
                        }
                    });
                }
            }
            Some(e) => {
                h.failures.fetch_add(1, Ordering::Relaxed);
                if self.breaker.failure_threshold > 0 && (self.breaker.count_if)(e) {
                    s.consecutive_failures = s.consecutive_failures.saturating_add(1);
                    if s.consecutive_failures >= self.breaker.failure_threshold {
                        s.open_until = Some(Instant::now() + self.breaker.cool_down);
                    }
                }
            }
        }
    }
}

#[async_trait]
impl ChatModel for Router {
    fn name(&self) -> &str {
        &self.name
    }

    /// The union of the routes' capabilities.
    fn capabilities(&self) -> Capabilities {
        self.routes
            .iter()
            .fold(Capabilities::none(), |a, r| union(a, r.capabilities))
    }

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        self.chat_with_hints(request, &RouteHints::default()).await
    }

    async fn chat_stream(&self, request: ChatRequest) -> Result<ChatStream> {
        self.chat_stream_with_hints(request, &RouteHints::default())
            .await
    }
}

fn union(a: Capabilities, b: Capabilities) -> Capabilities {
    Capabilities {
        tools: a.tools || b.tools,
        streaming: a.streaming || b.streaming,
        vision: a.vision || b.vision,
        pdf: a.pdf || b.pdf,
        audio: a.audio || b.audio,
        json_schema: a.json_schema || b.json_schema,
        cached_tokens: a.cached_tokens || b.cached_tokens,
        sampling: a.sampling || b.sampling,
        forced_tool_choice: a.forced_tool_choice || b.forced_tool_choice,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{collect_stream, mock::ScriptedModel, ToolCall, ToolSpec, Usage};
    use std::sync::atomic::AtomicUsize;

    /// Answers with its own name after `delay`; fails while `fail` is set.
    struct Fake {
        name: &'static str,
        delay: Duration,
        caps: Capabilities,
        fail: std::sync::atomic::AtomicBool,
        calls: AtomicUsize,
    }

    impl Fake {
        fn new(name: &'static str) -> Self {
            Self {
                name,
                delay: Duration::ZERO,
                caps: Capabilities::all(),
                fail: Default::default(),
                calls: AtomicUsize::new(0),
            }
        }
        fn delay(mut self, d: Duration) -> Self {
            self.delay = d;
            self
        }
        fn caps(mut self, c: Capabilities) -> Self {
            self.caps = c;
            self
        }
        fn failing(self, on: bool) -> Self {
            self.set_failing(on);
            self
        }
        fn set_failing(&self, on: bool) {
            self.fail.store(on, Ordering::SeqCst);
        }
        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl ChatModel for Fake {
        fn name(&self) -> &str {
            self.name
        }
        fn capabilities(&self) -> Capabilities {
            self.caps
        }
        async fn chat(&self, _r: ChatRequest) -> Result<ChatResponse> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if !self.delay.is_zero() {
                tokio::time::sleep(self.delay).await;
            }
            if self.fail.load(Ordering::SeqCst) {
                return Err(Error::Provider {
                    provider: self.name.into(),
                    status: 503,
                    body: "down".into(),
                });
            }
            Ok(ChatResponse {
                message: Message::assistant(self.name),
                usage: Usage::new(1, 1),
                model: self.name.into(),
            })
        }
    }

    fn text_only() -> Capabilities {
        Capabilities::all()
            .with_vision(false)
            .with_pdf(false)
            .with_audio(false)
    }

    fn collect_decisions(b: RouterBuilder) -> (RouterBuilder, Arc<Mutex<Vec<RouteDecision>>>) {
        let log = Arc::new(Mutex::new(Vec::new()));
        let l = log.clone();
        (b.on_route(move |d| l.lock().unwrap().push(d.clone())), log)
    }

    #[tokio::test]
    async fn rules_pick_routes_in_order_with_hints_and_default() {
        let router = Router::builder()
            .route("cheap", Arc::new(Fake::new("cheap")))
            .route("strong", Arc::new(Fake::new("strong")))
            .route("tools", Arc::new(Fake::new("tools")))
            .rule("long", |r| r.estimated_tokens() > 1_000, "strong")
            .rule("tools", |r| r.has_tools(), "tools")
            .rule(
                "enterprise",
                |r| r.hint("tier") == Some("enterprise"),
                "strong",
            )
            .default_route("cheap")
            .build()
            .unwrap();

        let ask = |req: ChatRequest, hints: RouteHints| {
            let router = &router;
            async move { router.chat_with_hints(req, &hints).await.unwrap().model }
        };
        assert_eq!(ask("hi".into(), RouteHints::new()).await, "cheap");
        assert_eq!(
            ask("word ".repeat(2_000).into(), RouteHints::new()).await,
            "strong"
        );
        let with_tools = ChatRequest::from("hi").with_tools(vec![ToolSpec {
            name: "t".into(),
            description: "d".into(),
            parameters: json!({"type": "object"}),
        }]);
        assert_eq!(ask(with_tools, RouteHints::new()).await, "tools");
        let tier = RouteHints::new().with("tier", "enterprise");
        assert_eq!(ask("hi".into(), tier.clone()).await, "strong");

        let d = router.route(&"hi".into(), &tier).await.unwrap();
        assert_eq!(
            d,
            RouteDecision {
                route: "strong".into(),
                model: "strong".into(),
                reason: RouteReason::Rule {
                    name: "enterprise".into()
                },
                fallback: false,
            }
        );
        assert_eq!(
            router
                .route(&"hi".into(), &RouteHints::new())
                .await
                .unwrap()
                .reason,
            RouteReason::Default
        );
    }

    #[tokio::test]
    async fn images_never_go_to_a_model_without_vision() {
        let image = ChatRequest::from(Message::user_with_images(
            "what is this?",
            [ContentPart::image_url("https://example.com/cat.png")],
        ));
        let cheap = Arc::new(Fake::new("cheap").caps(text_only()));
        let router = Router::builder()
            .route("cheap", cheap.clone())
            .route("vision", Arc::new(Fake::new("vision")))
            .default_route("cheap")
            .build()
            .unwrap();
        let d = router.route(&image, &RouteHints::new()).await.unwrap();
        assert_eq!(d.route, "vision");
        assert!(d.fallback);
        assert_eq!(router.chat(image.clone()).await.unwrap().model, "vision");
        assert_eq!(cheap.calls(), 0);
        assert!(router.capabilities().vision);

        // No route can see images: a configuration error, no call made.
        let router = Router::builder()
            .route("cheap", cheap.clone())
            .build()
            .unwrap();
        let err = router.chat(image).await.unwrap_err();
        assert!(
            matches!(&err, Error::Config(m) if m.contains("images")),
            "{err}"
        );
        assert_eq!(cheap.calls(), 0);
        assert!(!router.capabilities().vision);
    }

    fn label_call(label: &str) -> Message {
        Message::assistant_with_tools(
            "",
            vec![ToolCall {
                id: "c1".into(),
                name: "route_label".into(),
                arguments: json!({ "label": label }),
            }],
        )
    }

    #[tokio::test]
    async fn classifier_labels_pick_the_route() {
        let small = Arc::new(ScriptedModel::new([label_call("hard"), label_call("easy")]));
        let (b, log) = collect_decisions(
            Router::builder()
                .route("cheap", Arc::new(Fake::new("cheap")))
                .route("strong", Arc::new(Fake::new("strong")))
                .rule("forced", |r| r.hint("force") == Some("cheap"), "cheap")
                .classifier(
                    Classifier::new(small.clone())
                        .label("easy", "short lookups", "cheap")
                        .label("hard", "proofs and code", "strong"),
                ),
        );
        let router = b.build().unwrap();
        assert_eq!(
            router.chat("prove it".into()).await.unwrap().model,
            "strong"
        );
        assert_eq!(router.chat("hello".into()).await.unwrap().model, "cheap");
        // Rules run before the classifier.
        let hints = RouteHints::new().with("force", "cheap");
        assert_eq!(
            router
                .chat_with_hints("x".into(), &hints)
                .await
                .unwrap()
                .model,
            "cheap"
        );
        assert_eq!(small.requests().len(), 2);
        let first = &small.requests()[0];
        assert!(first.messages[1].content.contains("hard: proofs and code"));
        assert!(first.messages.last().unwrap().content.contains("prove it"));
        assert_eq!(
            log.lock().unwrap()[0].reason,
            RouteReason::Classifier {
                label: "hard".into()
            }
        );

        // A failing classifier (script exhausted) falls back to the default.
        let d = router
            .route(&"again".into(), &RouteHints::new())
            .await
            .unwrap();
        assert_eq!(d.route, "cheap");
        assert!(matches!(d.reason, RouteReason::ClassifierError { .. }));
    }

    #[tokio::test]
    async fn cascade_escalates_on_refusal_and_stops_on_a_good_answer() {
        let cheap = Arc::new(ScriptedModel::new([
            Message::assistant("I'm sorry, I can't help with that."),
            Message::assistant("Paris."),
        ]));
        let strong = Arc::new(ScriptedModel::new([Message::assistant(
            "A careful answer.",
        )]));
        let (b, log) = collect_decisions(
            Router::builder()
                .route("cheap", cheap.clone())
                .route("strong", strong.clone())
                .cascade(["cheap", "strong"], Validator::no_refusal()),
        );
        let router = b.build().unwrap();

        let r = router.chat("hard question".into()).await.unwrap();
        assert_eq!(r.message.content, "A careful answer.");
        assert_eq!(router.escalations(), 1);
        assert_eq!(
            log.lock().unwrap()[1].reason,
            RouteReason::Escalation {
                from: "cheap".into()
            }
        );

        let r = router.chat("capital of France?".into()).await.unwrap();
        assert_eq!(r.message.content, "Paris.");
        assert_eq!(router.escalations(), 1);
        assert_eq!(strong.requests().len(), 1);
        assert_eq!(cheap.requests().len(), 2);
    }

    #[tokio::test]
    async fn cascade_escalates_on_structured_parse_failure_and_errors() {
        let so = StructuredOutput::new(
            "n",
            json!({"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}),
        )
        .with_mode(StructuredMode::Json);
        let cheap = Arc::new(ScriptedModel::new([Message::assistant("three")]));
        let strong = Arc::new(ScriptedModel::new([Message::assistant("{\"n\": 3}")]));
        let router = Router::builder()
            .route("cheap", cheap)
            .route("strong", strong)
            .cascade(["cheap", "strong"], Validator::structured(so.clone()))
            .build()
            .unwrap();
        let n: Value = so
            .with_retries(0)
            .generate(&router, "how many?".into())
            .await
            .unwrap();
        assert_eq!(n["n"], 3);
        assert_eq!(router.escalations(), 1);

        // An error from the cheap model escalates too; a failing last step
        // keeps the earlier successful (unvalidated) answer.
        let router = Router::builder()
            .route("cheap", Arc::new(Fake::new("cheap").failing(true)))
            .route("strong", Arc::new(Fake::new("strong")))
            .cascade(["cheap", "strong"], Validator::non_empty())
            .build()
            .unwrap();
        assert_eq!(router.chat("q".into()).await.unwrap().model, "strong");
        let router = Router::builder()
            .route(
                "cheap",
                Arc::new(ScriptedModel::new([Message::assistant(" ")])),
            )
            .route("strong", Arc::new(Fake::new("strong").failing(true)))
            .cascade(["cheap", "strong"], Validator::non_empty())
            .build()
            .unwrap();
        assert_eq!(router.chat("q".into()).await.unwrap().model, "scripted");
        assert_eq!(router.escalations(), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn circuit_opens_after_failures_and_closes_after_cool_down() {
        let primary = Arc::new(Fake::new("primary").failing(true));
        let backup = Arc::new(Fake::new("backup"));
        let router = Router::builder()
            .route("primary", primary.clone())
            .route("backup", backup.clone())
            .circuit_breaker(CircuitBreaker::new(2, Duration::from_secs(10)))
            .build()
            .unwrap();

        assert!(router.chat("a".into()).await.is_err());
        assert!(router.chat("b".into()).await.is_err());
        assert!(matches!(
            router.stats()[0].circuit,
            CircuitState::Open { .. }
        ));
        // Open: the primary is skipped.
        let d = router.route(&"c".into(), &RouteHints::new()).await.unwrap();
        assert_eq!((d.route.as_str(), d.fallback), ("backup", true));
        assert_eq!(router.chat("c".into()).await.unwrap().model, "backup");
        assert_eq!(primary.calls(), 2);

        // Half-open after the cool-down: a failure re-opens immediately.
        tokio::time::advance(Duration::from_secs(11)).await;
        assert_eq!(router.stats()[0].circuit, CircuitState::HalfOpen);
        assert!(router.chat("d".into()).await.is_err());
        assert!(matches!(
            router.stats()[0].circuit,
            CircuitState::Open { .. }
        ));
        assert_eq!(router.chat("e".into()).await.unwrap().model, "backup");

        // Recovered: a half-open success closes the circuit.
        primary.set_failing(false);
        tokio::time::advance(Duration::from_secs(11)).await;
        assert_eq!(router.chat("f".into()).await.unwrap().model, "primary");
        let s = &router.stats()[0];
        assert_eq!(s.circuit, CircuitState::Closed);
        assert_eq!(s.consecutive_failures, 0);
        assert_eq!((s.served, s.failures), (1, 3));
        assert_eq!(router.stats()[1].served, 2);
    }

    #[tokio::test]
    async fn client_errors_do_not_open_the_circuit() {
        struct BadRequest;
        #[async_trait]
        impl ChatModel for BadRequest {
            fn name(&self) -> &str {
                "bad"
            }
            async fn chat(&self, _r: ChatRequest) -> Result<ChatResponse> {
                Err(Error::Provider {
                    provider: "p".into(),
                    status: 400,
                    body: "bad".into(),
                })
            }
        }
        let router = Router::builder()
            .route("bad", Arc::new(BadRequest))
            .circuit_breaker(CircuitBreaker::new(1, Duration::from_secs(60)))
            .build()
            .unwrap();
        for _ in 0..3 {
            assert!(router.chat("x".into()).await.is_err());
        }
        assert_eq!(router.stats()[0].circuit, CircuitState::Closed);
        assert_eq!(router.stats()[0].failures, 3);
    }

    #[tokio::test]
    async fn all_open_circuits_fail_fast() {
        let router = Router::builder()
            .route("only", Arc::new(Fake::new("only").failing(true)))
            .circuit_breaker(CircuitBreaker::new(1, Duration::from_secs(60)))
            .build()
            .unwrap();
        assert!(matches!(
            router.chat("x".into()).await,
            Err(Error::Provider { .. })
        ));
        let err = router.chat("x".into()).await.unwrap_err();
        assert!(err.to_string().contains("open circuit"), "{err}");
    }

    #[tokio::test(start_paused = true)]
    async fn equivalent_routes_prefer_lower_latency() {
        let slow = Arc::new(Fake::new("slow").delay(Duration::from_millis(400)));
        let fast = Arc::new(Fake::new("fast").delay(Duration::from_millis(50)));
        let router = Router::builder()
            .route("slow", slow.clone())
            .route("fast", fast.clone())
            .default_routes(["slow", "fast"])
            .build()
            .unwrap();
        // Unknown latency counts as zero, so each route is tried once.
        assert_eq!(router.chat("1".into()).await.unwrap().model, "slow");
        assert_eq!(router.chat("2".into()).await.unwrap().model, "fast");
        for _ in 0..3 {
            assert_eq!(router.chat("n".into()).await.unwrap().model, "fast");
        }
        let stats = router.stats();
        assert_eq!(stats[0].latency, Some(Duration::from_millis(400)));
        let fast_latency = stats[1].latency.unwrap();
        assert!(fast_latency.abs_diff(Duration::from_millis(50)) < Duration::from_micros(1));
        assert_eq!((slow.calls(), fast.calls()), (1, 4));
    }

    #[tokio::test]
    async fn streams_route_once() {
        let strong = Arc::new(ScriptedModel::new([Message::assistant("streamed answer")]));
        let router = Router::builder()
            .route("cheap", Arc::new(ScriptedModel::new([])))
            .route("strong", strong)
            .rule("all", |_| true, "strong")
            .cascade(["cheap", "strong"], Validator::non_empty())
            .build()
            .unwrap();
        let resp = collect_stream(router.chat_stream("hi".into()).await.unwrap())
            .await
            .unwrap();
        assert_eq!(resp.message.content, "streamed answer");
        assert_eq!(router.stats()[1].served, 1);
    }

    #[test]
    fn build_checks_route_names() {
        let m = || Arc::new(Fake::new("m")) as Arc<dyn ChatModel>;
        let err = |b: RouterBuilder| b.build().unwrap_err().to_string();
        assert!(err(Router::builder()).contains("no routes"));
        assert!(err(Router::builder().route("a", m()).route("a", m())).contains("twice"));
        assert!(err(Router::builder().route("a", m()).default_route("b")).contains("`b`"));
        assert!(err(Router::builder().route("a", m()).rule("r", |_| true, "x")).contains("`x`"));
        assert!(err(Router::builder()
            .route("a", m())
            .cascade(["a", "z"], Validator::non_empty()))
        .contains("`z`"));
        assert!(err(Router::builder()
            .route("a", m())
            .classifier(Classifier::new(m())))
        .contains("no labels"));
        assert!(err(Router::builder().route("a", m()).latency_alpha(0.0)).contains("alpha"));
    }

    #[test]
    fn refusal_heuristic() {
        for t in [
            "I can't do that.",
            "I’m sorry, but no.",
            "As an AI language model, ...",
            "Unfortunately, I cannot",
        ] {
            assert!(looks_like_refusal(t), "{t}");
        }
        for t in [
            "Paris.",
            "I can help: here it is.",
            "Sorry for the delay, here it is.",
        ] {
            assert!(!looks_like_refusal(t), "{t}");
        }
    }
}
