//! [`Agent`], its builder and the graph it runs on.

use std::{collections::BTreeMap, path::PathBuf, sync::Arc, time::Duration};

use futures_util::{future::join_all, StreamExt};
use serde_json::{json, Value};
use sinew_core::{
    providers::from_spec, ChatModel, ChatRequest, Message, Role, StreamEvent, StructuredOutput,
    Tool, ToolBox, ToolSpec,
};
use sinew_graph::durable::{durable_run, UnknownOutcome};
use sinew_graph::prebuilt::{LoopAction, LoopGuard, LOOP_STOP_REPLY};
use sinew_graph::{
    BoxError, Checkpointer, CompiledGraph, GraphEvent, MemoryCheckpointer, NodeContext, RunOutcome,
    StateGraph, END,
};
use sinew_trace::{traced_chat, traced_chat_stream, RunKind, Tracer};

use crate::{
    budget::{Budget, Pricing},
    context::{
        compact, estimate_messages, estimate_tokens, offload_output, repair_tool_pairs,
        ContextConfig,
    },
    events::{AgentEvent, AgentOutcome, AgentStream, RunResult},
    guardrails::{apply_guardrails, Guardrail, GuardrailStage},
    hooks::{Hook, HookContext, StepPhase, ToolDecision},
    instructions::{find_instruction_files, instructions_prompt},
    permissions::{Permission, PermissionPolicy},
    state::{
        now_ms, ApprovalDecision, CallDecision, HarnessState, PendingCall, StopReason, TurnState,
    },
    tools::{
        fs_tools,
        skills::{load_skills, skills_prompt, LoadSkillTool},
        task::{SubagentSpec, SubagentTools, TaskTool},
        todo, HttpFetchTool, HttpOptions, ShellOptions, ShellTool, ThinkTool, Workspace,
        WriteTodosTool,
    },
    HarnessError, Result,
};

/// Used when no instructions are given.
pub const DEFAULT_INSTRUCTIONS: &str = "You are a capable, careful AI agent. Use the available tools to complete the user's task: gather the information you need, act, and verify the result. Prefer doing over asking when the task is clear. When you are done, reply with a concise final answer.";

const NODE_AGENT: &str = "agent";
const NODE_TOOLS: &str = "tools";
const NODE_APPROVAL: &str = "approval";

/// Resolved, immutable configuration shared by the graph nodes.
pub(crate) struct Config {
    name: String,
    model: Arc<dyn ChatModel>,
    system_prompt: String,
    tools: BTreeMap<String, Arc<dyn Tool>>,
    specs: Vec<ToolSpec>,
    spec_tokens: usize,
    hooks: Vec<Arc<dyn Hook>>,
    policy: PermissionPolicy,
    auto_approve: bool,
    ask_as_deny: bool,
    context: ContextConfig,
    offload_dir: PathBuf,
    offload_display: String,
    budget: Budget,
    loop_guard: Option<LoopGuard>,
    unknown_outcome: UnknownOutcome,
    pricing: Option<Pricing>,
    guardrails: Vec<Arc<dyn Guardrail>>,
    output: Option<StructuredOutput>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    depth: usize,
    tracer: Option<Tracer>,
}

/// What a `task` tool needs to build subagents.
pub(crate) struct SubagentContext {
    pub(crate) specs: Vec<SubagentSpec>,
    model: Arc<dyn ChatModel>,
    tools: Vec<Arc<dyn Tool>>,
    hooks: Vec<Arc<dyn Hook>>,
    policy: PermissionPolicy,
    auto_approve: bool,
    context: ContextConfig,
    offload: (PathBuf, String),
    budget: Budget,
    loop_guard: Option<LoopGuard>,
    unknown_outcome: UnknownOutcome,
    pricing: Option<Pricing>,
    guardrails: Vec<Arc<dyn Guardrail>>,
    tracer: Option<Tracer>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    depth: usize,
    max_depth: usize,
    env_prompt: String,
}

/// Builds a subagent for the `task` tool.
pub(crate) fn build_subagent(ctx: &SubagentContext, spec: &SubagentSpec) -> Result<Agent> {
    let mut tools: Vec<Arc<dyn Tool>> = match &spec.tools {
        SubagentTools::Inherit => ctx.tools.clone(),
        SubagentTools::Only(names) => ctx
            .tools
            .iter()
            .filter(|t| names.contains(&t.spec().name))
            .cloned()
            .collect(),
        SubagentTools::None => Vec::new(),
    };
    tools.extend(spec.extra_tools.iter().cloned());
    let mut instructions = spec.instructions.clone();
    if !ctx.env_prompt.is_empty() {
        instructions.push_str("\n\n");
        instructions.push_str(&ctx.env_prompt);
    }
    let mut budget = ctx.budget.clone();
    if let Some(n) = spec.max_steps {
        budget.max_steps = n;
    }
    let b = AgentBuilder {
        model: Some(spec.model.clone().unwrap_or_else(|| ctx.model.clone())),
        name: Some(spec.name.clone()),
        instructions: Some(instructions),
        tools,
        hooks: ctx.hooks.clone(),
        policy: ctx.policy.clone(),
        auto_approve: ctx.auto_approve,
        ask_as_deny: true,
        context: ctx.context.clone(),
        offload: Some(ctx.offload.clone()),
        budget,
        loop_guard: ctx.loop_guard.clone(),
        unknown_outcome: ctx.unknown_outcome,
        pricing: ctx.pricing,
        guardrails: ctx.guardrails.clone(),
        tracer: ctx.tracer.clone(),
        temperature: ctx.temperature,
        max_tokens: ctx.max_tokens,
        depth: ctx.depth,
        max_depth: ctx.max_depth,
        subagents: if ctx.depth < ctx.max_depth {
            ctx.specs.clone()
        } else {
            Vec::new()
        },
        ..AgentBuilder::new()
    };
    b.build()
}

/// Configures and builds an [`Agent`]. Every built-in tool is opt-in.
///
/// ```no_run
/// # async fn demo() -> sinew_harness::Result<()> {
/// use sinew_harness::{Agent, PermissionPolicy, ShellOptions};
///
/// let agent = Agent::builder()
///     .model_spec("ollama:qwen3:4b-instruct-2507-q4_K_M")
///     .instructions("You are a coding assistant.")
///     .workspace(".")
///     .filesystem()
///     .shell(ShellOptions::default().allow("cargo *").allow("git status*"))
///     .todos()
///     .general_subagent()
///     .permissions(PermissionPolicy::interactive())
///     .max_steps(30)
///     .build()?;
/// let result = agent.ask("Summarize README.md").await?;
/// println!("{}", result.text);
/// # Ok(()) }
/// ```
#[derive(Clone)]
pub struct AgentBuilder {
    model: Option<Arc<dyn ChatModel>>,
    model_spec: Option<String>,
    name: Option<String>,
    instructions: Option<String>,
    tools: Vec<Arc<dyn Tool>>,
    workspace: Option<PathBuf>,
    fs: Option<bool>,
    shell: Option<ShellOptions>,
    http: Option<HttpOptions>,
    todos: bool,
    think: bool,
    subagents: Vec<SubagentSpec>,
    max_depth: usize,
    hooks: Vec<Arc<dyn Hook>>,
    policy: PermissionPolicy,
    auto_approve: bool,
    ask_as_deny: bool,
    context: ContextConfig,
    offload: Option<(PathBuf, String)>,
    budget: Budget,
    loop_guard: Option<LoopGuard>,
    unknown_outcome: UnknownOutcome,
    pricing: Option<Pricing>,
    guardrails: Vec<Arc<dyn Guardrail>>,
    project_instructions: bool,
    instruction_files: Vec<PathBuf>,
    skills_dirs: Vec<PathBuf>,
    output: Option<StructuredOutput>,
    checkpointer: Option<Arc<dyn Checkpointer>>,
    tracer: Option<Tracer>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    depth: usize,
}

impl Default for AgentBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentBuilder {
    pub fn new() -> Self {
        Self {
            model: None,
            model_spec: None,
            name: None,
            instructions: None,
            tools: Vec::new(),
            workspace: None,
            fs: None,
            shell: None,
            http: None,
            todos: false,
            think: false,
            subagents: Vec::new(),
            max_depth: 1,
            hooks: Vec::new(),
            policy: PermissionPolicy::allow_all(),
            auto_approve: false,
            ask_as_deny: false,
            context: ContextConfig::default(),
            offload: None,
            budget: Budget::default(),
            loop_guard: Some(LoopGuard::default()),
            unknown_outcome: UnknownOutcome::Fail,
            pricing: None,
            guardrails: Vec::new(),
            project_instructions: false,
            instruction_files: Vec::new(),
            skills_dirs: Vec::new(),
            output: None,
            checkpointer: None,
            tracer: None,
            temperature: None,
            max_tokens: None,
            depth: 0,
        }
    }

    // ---- model and prompt ----

    pub fn model(mut self, model: impl ChatModel + 'static) -> Self {
        self.model = Some(Arc::new(model));
        self
    }

    pub fn model_arc(mut self, model: Arc<dyn ChatModel>) -> Self {
        self.model = Some(model);
        self
    }

    /// A `provider:model` spec resolved with
    /// [`from_spec`] at build time, e.g.
    /// `ollama:qwen3:4b-instruct-2507-q4_K_M` or `anthropic:claude-sonnet-4-5`.
    pub fn model_spec(mut self, spec: impl Into<String>) -> Self {
        self.model_spec = Some(spec.into());
        self
    }

    /// Name used in traces and hook contexts (default `agent`).
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// The system prompt's main instructions.
    pub fn instructions(mut self, text: impl Into<String>) -> Self {
        self.instructions = Some(text.into());
        self
    }

    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }

    pub fn max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = Some(n);
        self
    }

    /// Asks for a final answer matching a JSON schema; the parsed value is
    /// returned in [`RunResult::output`].
    pub fn output_schema(mut self, output: StructuredOutput) -> Self {
        self.output = Some(output);
        self
    }

    // ---- tools ----

    pub fn tool(mut self, tool: impl Tool + 'static) -> Self {
        self.tools.push(Arc::new(tool));
        self
    }

    pub fn tool_arc(mut self, tool: Arc<dyn Tool>) -> Self {
        self.tools.push(tool);
        self
    }

    pub fn tools(mut self, tools: impl IntoIterator<Item = Arc<dyn Tool>>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Every tool of a [`ToolBox`] (e.g. `McpClient::toolbox()`).
    pub fn toolbox(mut self, toolbox: &ToolBox) -> Self {
        self.tools.extend(crate::tools::from_toolbox(toolbox));
        self
    }

    /// Root directory for the filesystem tools, the shell and project files.
    pub fn workspace(mut self, dir: impl Into<PathBuf>) -> Self {
        self.workspace = Some(dir.into());
        self
    }

    /// `read_file`, `write_file`, `edit_file`, `list_dir`, `glob`, `grep`.
    pub fn filesystem(mut self) -> Self {
        self.fs = Some(false);
        self
    }

    /// `read_file`, `list_dir`, `glob`, `grep`.
    pub fn filesystem_read_only(mut self) -> Self {
        self.fs = Some(true);
        self
    }

    /// `shell`, running in the workspace (or the current directory).
    pub fn shell(mut self, options: ShellOptions) -> Self {
        self.shell = Some(options);
        self
    }

    /// `http_fetch`.
    pub fn http_fetch(mut self, options: HttpOptions) -> Self {
        self.http = Some(options);
        self
    }

    /// `write_todos` (plan kept in the thread state).
    pub fn todos(mut self) -> Self {
        self.todos = true;
        self
    }

    /// `think` (scratchpad).
    pub fn think(mut self) -> Self {
        self.think = true;
        self
    }

    /// Skills folder(s): `*/SKILL.md` are listed in the system prompt and
    /// loaded with `load_skill`.
    pub fn skills_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.skills_dirs.push(dir.into());
        self
    }

    /// Loads `AGENTS.md` / `CLAUDE.md` from the workspace into the prompt.
    pub fn project_instructions(mut self) -> Self {
        self.project_instructions = true;
        self
    }

    /// Adds an instruction file to the prompt.
    pub fn instructions_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.instruction_files.push(path.into());
        self
    }

    // ---- subagents ----

    /// A subagent available through the `task` tool.
    pub fn subagent(mut self, spec: SubagentSpec) -> Self {
        self.subagents.push(spec);
        self
    }

    /// The built-in `general-purpose` subagent.
    pub fn general_subagent(self) -> Self {
        self.subagent(SubagentSpec::general_purpose())
    }

    /// How deep subagents may nest (default 1: subagents cannot spawn
    /// subagents of their own).
    pub fn max_subagent_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    // ---- control ----

    pub fn hook(mut self, hook: impl Hook + 'static) -> Self {
        self.hooks.push(Arc::new(hook));
        self
    }

    pub fn hook_arc(mut self, hook: Arc<dyn Hook>) -> Self {
        self.hooks.push(hook);
        self
    }

    pub fn permissions(mut self, policy: PermissionPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Turns every "ask" into "allow" (the CLI's `--yes`).
    pub fn auto_approve(mut self, on: bool) -> Self {
        self.auto_approve = on;
        self
    }

    pub fn guardrail(mut self, guardrail: impl Guardrail + 'static) -> Self {
        self.guardrails.push(Arc::new(guardrail));
        self
    }

    pub fn context(mut self, config: ContextConfig) -> Self {
        self.context = config;
        self
    }

    pub fn max_context_tokens(mut self, n: usize) -> Self {
        self.context.max_context_tokens = n;
        self
    }

    pub fn budget(mut self, budget: Budget) -> Self {
        self.budget = budget;
        self
    }

    /// Loop detection for tool calls (on by default with
    /// [`LoopGuard::default`]): a call repeated with the same arguments is
    /// answered with a nudge instead of running, and a turn whose tool calls
    /// keep failing stops with [`StopReason::Guardrail`]. With
    /// [`LoopAction::Interrupt`] the repeated calls go to approval instead.
    /// Pass `None` to turn it off.
    /// Tool calls are durable when the agent has a checkpointer with a tool
    /// ledger (the default in-memory one, SQLite or Postgres): calls that
    /// finished before a crash are not run again when the thread resumes.
    /// This sets what happens to a side-effecting call (`Tool::side_effects`)
    /// that started but has no recorded result (default `Fail`: the model is
    /// told the outcome is unknown). `Interrupt` behaves like `Fail` here;
    /// use approvals to have a human decide.
    pub fn unknown_tool_outcome(mut self, policy: UnknownOutcome) -> Self {
        self.unknown_outcome = policy;
        self
    }

    pub fn loop_guard(mut self, guard: impl Into<Option<LoopGuard>>) -> Self {
        self.loop_guard = guard.into();
        self
    }

    pub fn max_steps(mut self, n: u32) -> Self {
        self.budget.max_steps = n;
        self
    }

    pub fn max_total_tokens(mut self, n: u64) -> Self {
        self.budget.max_total_tokens = Some(n);
        self
    }

    pub fn max_cost_usd(mut self, usd: f64) -> Self {
        self.budget.max_cost_usd = Some(usd);
        self
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.budget.timeout = Some(timeout);
        self
    }

    pub fn pricing(mut self, pricing: Pricing) -> Self {
        self.pricing = Some(pricing);
        self
    }

    /// Where threads are checkpointed (default: in memory).
    pub fn checkpointer(mut self, cp: impl Checkpointer + 'static) -> Self {
        self.checkpointer = Some(Arc::new(cp));
        self
    }

    pub fn shared_checkpointer(mut self, cp: Arc<dyn Checkpointer>) -> Self {
        self.checkpointer = Some(cp);
        self
    }

    /// Traces every graph node, model call and tool call.
    pub fn tracer(mut self, tracer: Tracer) -> Self {
        self.tracer = Some(tracer);
        self
    }

    pub fn build(self) -> Result<Agent> {
        let model = match (&self.model, &self.model_spec) {
            (Some(m), _) => m.clone(),
            (None, Some(spec)) => from_spec(spec)?,
            (None, None) => return Err(HarnessError::Config("no model configured".into())),
        };
        let workspace =
            match &self.workspace {
                Some(dir) => Some(Workspace::new(dir).map_err(|e| {
                    HarnessError::Config(format!("workspace {}: {e}", dir.display()))
                })?),
                None => None,
            };

        let mut tools = self.tools.clone();
        if let Some(read_only) = self.fs {
            let ws = workspace
                .as_ref()
                .ok_or_else(|| HarnessError::Config("filesystem tools need a workspace".into()))?;
            tools.extend(fs_tools(ws, read_only));
        }
        if let Some(opts) = &self.shell {
            let cwd = match &workspace {
                Some(ws) => ws.root().to_path_buf(),
                None => std::env::current_dir()?,
            };
            tools.push(Arc::new(ShellTool::new(cwd, opts.clone())));
        }
        if let Some(opts) = &self.http {
            tools.push(Arc::new(HttpFetchTool::new(opts.clone())));
        }
        if self.todos {
            tools.push(Arc::new(WriteTodosTool));
        }
        if self.think {
            tools.push(Arc::new(ThinkTool));
        }
        let mut skills = Vec::new();
        for dir in &self.skills_dirs {
            skills.extend(
                load_skills(dir).map_err(|e| {
                    HarnessError::Config(format!("skills dir {}: {e}", dir.display()))
                })?,
            );
        }
        if !skills.is_empty() {
            tools.push(Arc::new(LoadSkillTool::new(skills.clone())));
        }

        // Prompt sections shared with subagents.
        let mut env = Vec::new();
        if let Some(ws) = &workspace {
            env.push(format!(
                "# Environment\nWorkspace root: {}\nFile tool paths are relative to the workspace root.",
                ws.root().display()
            ));
        }
        let mut files = self.instruction_files.clone();
        if self.project_instructions {
            let ws = workspace.as_ref().ok_or_else(|| {
                HarnessError::Config("project_instructions needs a workspace".into())
            })?;
            files.extend(find_instruction_files(ws.root()));
        }
        if let Some(p) = instructions_prompt(&files, workspace.as_ref().map(|w| w.root()), 20_000) {
            env.push(p);
        }
        if !skills.is_empty() {
            env.push(skills_prompt(&skills));
        }
        let env_prompt = env.join("\n\n");

        let offload =
            self.offload
                .clone()
                .unwrap_or_else(|| match (&self.context.offload_dir, &workspace) {
                    (Some(d), _) => (d.clone(), d.display().to_string()),
                    (None, Some(ws)) => (
                        ws.root().join(".sinew/tool-outputs"),
                        ".sinew/tool-outputs".to_string(),
                    ),
                    (None, None) => {
                        let d = std::env::temp_dir().join("sinew-harness");
                        let s = d.display().to_string();
                        (d, s)
                    }
                });

        let has_task = !self.subagents.is_empty() && self.depth < self.max_depth;
        if has_task {
            let ctx = SubagentContext {
                specs: self.subagents.clone(),
                model: model.clone(),
                tools: tools.clone(),
                hooks: self.hooks.clone(),
                policy: self.policy.clone(),
                auto_approve: self.auto_approve,
                context: self.context.clone(),
                offload: offload.clone(),
                budget: self.budget.clone(),
                loop_guard: self.loop_guard.clone(),
                unknown_outcome: self.unknown_outcome,
                pricing: self.pricing,
                guardrails: self.guardrails.clone(),
                tracer: self.tracer.clone(),
                temperature: self.temperature,
                max_tokens: self.max_tokens,
                depth: self.depth + 1,
                max_depth: self.max_depth,
                env_prompt: env_prompt.clone(),
            };
            tools.push(Arc::new(TaskTool { ctx: Arc::new(ctx) }));
        }

        let mut registry = BTreeMap::new();
        for t in tools {
            let name = t.spec().name;
            if registry.insert(name.clone(), t).is_some() {
                return Err(HarnessError::Config(format!(
                    "two tools are named `{name}`"
                )));
            }
        }
        let specs: Vec<ToolSpec> = registry.values().map(|t| t.spec()).collect();

        let mut prompt = vec![self
            .instructions
            .clone()
            .unwrap_or_else(|| DEFAULT_INSTRUCTIONS.to_string())];
        let mut hints = Vec::new();
        if registry.contains_key(todo::NAME) {
            hints.push("- For tasks with several steps, keep a plan with `write_todos` and update it as you go.");
        }
        if has_task {
            hints.push("- Delegate self-contained subtasks (broad searches, reading many files) with `task` to keep your context small.");
        }
        if registry.contains_key("think") {
            hints.push("- Use `think` to reason about tool results before acting.");
        }
        if !hints.is_empty() {
            prompt.push(format!("# Working style\n{}", hints.join("\n")));
        }
        if !env_prompt.is_empty() {
            prompt.push(env_prompt);
        }
        let system_prompt = prompt.join("\n\n");
        let spec_tokens = estimate_tokens(&serde_json::to_string(&specs).unwrap_or_default());

        let cfg = Arc::new(Config {
            name: self.name.clone().unwrap_or_else(|| "agent".into()),
            model,
            system_prompt,
            tools: registry,
            specs,
            spec_tokens,
            hooks: self.hooks.clone(),
            policy: self.policy.clone(),
            auto_approve: self.auto_approve,
            ask_as_deny: self.ask_as_deny,
            context: self.context.clone(),
            offload_dir: offload.0,
            offload_display: offload.1,
            budget: self.budget.clone(),
            loop_guard: self.loop_guard.clone(),
            unknown_outcome: self.unknown_outcome,
            pricing: self.pricing,
            guardrails: self.guardrails.clone(),
            output: self.output.clone(),
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            depth: self.depth,
            tracer: self.tracer.clone(),
        });
        let checkpointer = self
            .checkpointer
            .clone()
            .unwrap_or_else(|| Arc::new(MemoryCheckpointer::new()));
        let mut graph = build_graph(cfg.clone())?
            .with_shared_checkpointer(checkpointer.clone())
            .with_recursion_limit(cfg.budget.max_steps as usize * 3 + 10)
            .interrupt_before([NODE_APPROVAL]);
        if let Some(t) = &self.tracer {
            graph = graph.with_tracer(t.clone());
        }
        Ok(Agent {
            inner: Arc::new(Inner {
                cfg,
                graph,
                checkpointer,
            }),
        })
    }
}

struct Inner {
    cfg: Arc<Config>,
    graph: CompiledGraph<HarnessState>,
    checkpointer: Arc<dyn Checkpointer>,
}

/// A configured agent. Cheap to clone; every clone shares the checkpointer,
/// so threads can be continued from any clone.
#[derive(Clone)]
pub struct Agent {
    inner: Arc<Inner>,
}

enum Prepared {
    State(Box<HarnessState>),
    Blocked(Box<RunResult>),
}

impl Agent {
    pub fn builder() -> AgentBuilder {
        AgentBuilder::new()
    }

    /// The full system prompt sent with every model call.
    pub fn system_prompt(&self) -> &str {
        &self.inner.cfg.system_prompt
    }

    /// Names of every tool the model can call.
    pub fn tool_names(&self) -> Vec<String> {
        self.inner.cfg.tools.keys().cloned().collect()
    }

    pub fn model_name(&self) -> &str {
        self.inner.cfg.model.name()
    }

    pub fn checkpointer(&self) -> Arc<dyn Checkpointer> {
        self.inner.checkpointer.clone()
    }

    /// The underlying graph (for inspection or custom streaming).
    pub fn graph(&self) -> &CompiledGraph<HarnessState> {
        &self.inner.graph
    }

    /// Latest checkpointed state of a thread.
    pub async fn state(&self, thread_id: &str) -> Result<Option<HarnessState>> {
        match self
            .inner
            .checkpointer
            .latest(thread_id)
            .await
            .map_err(HarnessError::Other)?
        {
            Some(cp) => Ok(Some(serde_json::from_value(cp.state).map_err(|e| {
                HarnessError::Other(format!("corrupt thread state: {e}"))
            })?)),
            None => Ok(None),
        }
    }

    /// Runs one turn on a fresh thread and returns the result. Fails with
    /// [`HarnessError::ApprovalRequired`] if a tool needs approval.
    pub async fn ask(&self, input: impl Into<String>) -> Result<RunResult> {
        let thread = uuid::Uuid::new_v4().to_string();
        match self.run(&thread, input).await? {
            AgentOutcome::Completed(r) => Ok(r),
            AgentOutcome::ApprovalRequired { requests, .. } => Err(HarnessError::ApprovalRequired(
                requests
                    .iter()
                    .map(|r| r.tool.clone())
                    .collect::<Vec<_>>()
                    .join(", "),
            )),
        }
    }

    /// Sends `input` as the next user message of `thread_id` (created if
    /// new) and runs until the model answers, a budget stops the turn, or a
    /// tool needs approval.
    pub async fn run(&self, thread_id: &str, input: impl Into<String>) -> Result<AgentOutcome> {
        match self.prepare_turn(thread_id, input.into()).await? {
            Prepared::State(s) => self.drive(thread_id, Some(*s)).await,
            Prepared::Blocked(r) => Ok(AgentOutcome::Completed(*r)),
        }
    }

    /// Continues a thread from its last checkpoint after the process stopped
    /// mid-turn (a crash, a deploy, a cancelled request). Tool calls that
    /// finished before the stop are not run again; see
    /// [`AgentBuilder::unknown_tool_outcome`] for calls that were running.
    pub async fn recover(&self, thread_id: &str) -> Result<AgentOutcome> {
        self.drive(thread_id, None).await
    }

    /// Answers the approval requests of a paused thread and continues it.
    /// Calls without a decision are denied.
    pub async fn resume(
        &self,
        thread_id: &str,
        decisions: Vec<ApprovalDecision>,
    ) -> Result<AgentOutcome> {
        self.apply_decisions(thread_id, decisions).await?;
        self.drive(thread_id, None).await
    }

    /// Like [`run`](Self::run), streaming [`AgentEvent`]s.
    pub async fn stream(&self, thread_id: &str, input: impl Into<String>) -> Result<AgentStream> {
        match self.prepare_turn(thread_id, input.into()).await? {
            Prepared::State(s) => Ok(self.event_stream(thread_id, Some(*s))),
            Prepared::Blocked(result) => {
                let result = *result;
                Ok(futures_util::stream::once(async move { AgentEvent::Done { result } }).boxed())
            }
        }
    }

    /// Like [`resume`](Self::resume), streaming [`AgentEvent`]s.
    pub async fn resume_stream(
        &self,
        thread_id: &str,
        decisions: Vec<ApprovalDecision>,
    ) -> Result<AgentStream> {
        self.apply_decisions(thread_id, decisions).await?;
        Ok(self.event_stream(thread_id, None))
    }

    fn deadline_ms(&self) -> Option<u64> {
        self.inner
            .cfg
            .budget
            .timeout
            .map(|t| now_ms() + t.as_millis() as u64)
    }

    async fn prepare_turn(&self, thread_id: &str, input: String) -> Result<Prepared> {
        let cfg = &self.inner.cfg;
        let mut state = self.state(thread_id).await?.unwrap_or_default();
        state.pending.clear();
        repair_tool_pairs(&mut state.messages);
        state.turn = TurnState {
            deadline_ms: self.deadline_ms(),
            ..TurnState::default()
        };
        match apply_guardrails(&cfg.guardrails, GuardrailStage::Input, &input) {
            Ok(text) => {
                state.messages.push(Message::user(text));
                Ok(Prepared::State(Box::new(state)))
            }
            Err(reason) => Ok(Prepared::Blocked(Box::new(RunResult {
                thread_id: thread_id.into(),
                text: String::new(),
                output: None,
                stop: Some(StopReason::Guardrail(reason)),
                steps: 0,
                usage: Default::default(),
                thread_usage: state.usage,
                todos: state.todos,
            }))),
        }
    }

    async fn apply_decisions(
        &self,
        thread_id: &str,
        decisions: Vec<ApprovalDecision>,
    ) -> Result<()> {
        let graph = &self.inner.graph;
        let snap = graph
            .get_state_snapshot(thread_id)
            .await
            .map_err(|_| HarnessError::NoPendingApproval(thread_id.into()))?;
        if !snap.next.iter().any(|n| n == NODE_APPROVAL) {
            return Err(HarnessError::NoPendingApproval(thread_id.into()));
        }
        let mut state = snap.state;
        let mut always = Vec::new();
        for p in &mut state.pending {
            if !matches!(p.decision, CallDecision::Ask { .. }) {
                continue;
            }
            p.decision = match decisions.iter().find(|d| d.call_id == p.call.id) {
                Some(d) if d.approve => {
                    if d.always {
                        always.push(p.call.name.clone());
                    }
                    CallDecision::Allow
                }
                Some(d) => CallDecision::Deny {
                    reason: d
                        .reason
                        .clone()
                        .unwrap_or_else(|| "denied by the user".into()),
                },
                None => CallDecision::Deny {
                    reason: "no approval given".into(),
                },
            };
        }
        for name in always {
            if !state.always_allow.contains(&name) {
                state.always_allow.push(name);
            }
        }
        state.turn.deadline_ms = self.deadline_ms();
        graph.update_state(thread_id, state).await?;
        Ok(())
    }

    async fn drive(&self, thread_id: &str, input: Option<HarnessState>) -> Result<AgentOutcome> {
        let fut = self.inner.graph.run(thread_id, input);
        let outcome = match self.inner.cfg.budget.timeout {
            Some(t) => match tokio::time::timeout(t, fut).await {
                Ok(r) => r?,
                Err(_) => return self.timed_out(thread_id).await,
            },
            None => fut.await?,
        };
        match outcome {
            RunOutcome::Completed(state) => Ok(AgentOutcome::Completed(
                self.finish(thread_id, state).await?,
            )),
            RunOutcome::Interrupted { state, .. } => Ok(AgentOutcome::ApprovalRequired {
                thread_id: thread_id.into(),
                requests: state.approval_requests(),
            }),
        }
    }

    /// Result for a turn cancelled by the wall-clock timeout, from the last
    /// checkpoint.
    async fn timed_out(&self, thread_id: &str) -> Result<AgentOutcome> {
        let state = self.state(thread_id).await?.unwrap_or_default();
        Ok(AgentOutcome::Completed(self.result(
            thread_id,
            &state,
            Some(StopReason::Timeout),
            None,
        )))
    }

    fn result(
        &self,
        thread_id: &str,
        state: &HarnessState,
        stop: Option<StopReason>,
        output: Option<Value>,
    ) -> RunResult {
        RunResult {
            thread_id: thread_id.into(),
            text: state.final_text().unwrap_or_default().to_string(),
            output,
            stop,
            steps: state.turn.steps,
            usage: state.turn.usage,
            thread_usage: state.usage,
            todos: state.todos.clone(),
        }
    }

    async fn finish(&self, thread_id: &str, state: HarnessState) -> Result<RunResult> {
        let cfg = &self.inner.cfg;
        let stop = state.turn.stop.clone();
        let output = match (&cfg.output, &stop) {
            (Some(so), None) => Some(self.structured(so, &state).await?),
            _ => None,
        };
        Ok(self.result(thread_id, &state, stop, output))
    }

    async fn structured(&self, so: &StructuredOutput, state: &HarnessState) -> Result<Value> {
        let cfg = &self.inner.cfg;
        let mut messages = vec![Message::system(cfg.system_prompt.clone())];
        let mut history = state.messages.clone();
        repair_tool_pairs(&mut history);
        messages.extend(history);
        messages.push(Message::user(
            "Now give your final answer to my request in the required structure.",
        ));
        let run = cfg.tracer.as_ref().map(|t| {
            t.start(
                "structured_output",
                RunKind::Chain,
                None,
                json!({ "schema": so.schema() }),
            )
        });
        let result = so
            .generate::<Value, _>(cfg.model.as_ref(), ChatRequest::new(messages))
            .await;
        if let Some(run) = run {
            match &result {
                Ok(v) => run.finish(v.clone()),
                Err(e) => run.fail(e),
            }
        }
        Ok(result?)
    }

    fn event_stream(&self, thread_id: &str, input: Option<HarnessState>) -> AgentStream {
        let gs = self.inner.graph.stream(thread_id.to_string(), input);
        let deadline = self
            .inner
            .cfg
            .budget
            .timeout
            .map(|t| tokio::time::Instant::now() + t);
        let seed = Some((gs, self.clone(), thread_id.to_string()));
        futures_util::stream::unfold(seed, move |st| async move {
            let (mut gs, agent, thread) = st?;
            loop {
                let next = match deadline {
                    Some(d) => match tokio::time::timeout_at(d, gs.next()).await {
                        Ok(n) => n,
                        Err(_) => {
                            drop(gs);
                            let ev = match agent.timed_out(&thread).await {
                                Ok(AgentOutcome::Completed(result)) => AgentEvent::Done { result },
                                Ok(AgentOutcome::ApprovalRequired { requests, .. }) => {
                                    AgentEvent::ApprovalRequired { requests }
                                }
                                Err(e) => AgentEvent::Error {
                                    message: e.to_string(),
                                },
                            };
                            return Some((ev, None));
                        }
                    },
                    None => gs.next().await,
                };
                let ev = next?;
                let out = match ev {
                    GraphEvent::Token { delta, .. } => AgentEvent::Token { text: delta },
                    GraphEvent::Custom { value, .. } => match serde_json::from_value(value) {
                        Ok(e) => e,
                        Err(_) => continue,
                    },
                    GraphEvent::Completed { state } => {
                        let ev = match agent.finish(&thread, state).await {
                            Ok(result) => AgentEvent::Done { result },
                            Err(e) => AgentEvent::Error {
                                message: e.to_string(),
                            },
                        };
                        return Some((ev, None));
                    }
                    GraphEvent::Interrupted { state, .. } => {
                        return Some((
                            AgentEvent::ApprovalRequired {
                                requests: state.approval_requests(),
                            },
                            None,
                        ))
                    }
                    GraphEvent::Error(e) => {
                        return Some((
                            AgentEvent::Error {
                                message: e.to_string(),
                            },
                            None,
                        ))
                    }
                    _ => continue,
                };
                return Some((out, Some((gs, agent, thread))));
            }
        })
        .boxed()
    }
}

// ---------------------------------------------------------------- graph --

fn boxed(e: HarnessError) -> BoxError {
    Box::new(e)
}

fn build_graph(
    cfg: Arc<Config>,
) -> std::result::Result<CompiledGraph<HarnessState>, sinew_graph::GraphError> {
    let mut g = StateGraph::<HarnessState>::new();
    let c = cfg.clone();
    g.add_node_with_context(NODE_AGENT, move |s, ctx| {
        let c = c.clone();
        async move { agent_node(&c, s, ctx).await.map_err(boxed) }
    });
    let c = cfg.clone();
    g.add_node_with_context(NODE_TOOLS, move |s, ctx| {
        let c = c.clone();
        async move { tools_node(&c, s, ctx).await.map_err(boxed) }
    });
    g.add_node(NODE_APPROVAL, |mut s: HarnessState| async move {
        for p in &mut s.pending {
            if matches!(p.decision, CallDecision::Ask { .. }) {
                p.decision = CallDecision::Deny {
                    reason: "not approved".into(),
                };
            }
        }
        Ok(s)
    });
    g.add_conditional_edges(
        NODE_AGENT,
        |s: &HarnessState| {
            if s.turn.stop.is_some() || s.pending.is_empty() {
                END.to_string()
            } else if s
                .pending
                .iter()
                .any(|p| matches!(p.decision, CallDecision::Ask { .. }))
            {
                NODE_APPROVAL.to_string()
            } else {
                NODE_TOOLS.to_string()
            }
        },
        [NODE_TOOLS, NODE_APPROVAL, END],
    );
    g.add_edge(NODE_APPROVAL, NODE_TOOLS);
    g.add_edge(NODE_TOOLS, NODE_AGENT);
    g.set_entry_point(NODE_AGENT);
    g.compile()
}

fn emit(ctx: &NodeContext, ev: &AgentEvent) {
    if ctx.is_streaming() {
        if let Ok(v) = serde_json::to_value(ev) {
            ctx.emit(v);
        }
    }
}

fn hook_ctx(cfg: &Config, s: &HarnessState, ctx: &NodeContext) -> HookContext {
    HookContext {
        thread_id: ctx.thread_id.clone(),
        step: s.turn.steps,
        depth: cfg.depth,
        agent: cfg.name.clone(),
    }
}

fn budget_stop(cfg: &Config, s: &HarnessState) -> Option<StopReason> {
    let b = &cfg.budget;
    if s.turn.steps >= b.max_steps {
        return Some(StopReason::MaxSteps);
    }
    if s.turn.deadline_ms.is_some_and(|d| now_ms() >= d) {
        return Some(StopReason::Timeout);
    }
    if b.max_total_tokens
        .is_some_and(|max| s.turn.usage.total_tokens() >= max)
    {
        return Some(StopReason::TokenBudget);
    }
    if cfg.pricing.is_some()
        && b.max_cost_usd
            .is_some_and(|max| s.turn.usage.cost_usd >= max)
    {
        return Some(StopReason::CostBudget);
    }
    None
}

/// The reason of the first hook that asks to end the turn.
async fn hook_stop(cfg: &Config, hctx: &HookContext, phase: StepPhase) -> Option<String> {
    for h in &cfg.hooks {
        if let Some(reason) = h.should_stop(hctx, phase).await {
            return Some(reason);
        }
    }
    None
}

async fn agent_node(cfg: &Config, mut s: HarnessState, ctx: NodeContext) -> Result<HarnessState> {
    if s.turn.stop.is_none() {
        s.turn.stop = budget_stop(cfg, &s);
    }
    if s.turn.stop.is_none() {
        s.turn.stop = hook_stop(cfg, &hook_ctx(cfg, &s, &ctx), StepPhase::BeforeModel)
            .await
            .map(StopReason::Guardrail);
    }
    if s.turn.stop.is_some() {
        s.pending.clear();
        return Ok(s);
    }

    // Context budget: summarize older turns when the prompt is too big.
    let prompt_tokens =
        estimate_tokens(&cfg.system_prompt) + cfg.spec_tokens + estimate_messages(&s.messages);
    if prompt_tokens > cfg.context.max_context_tokens {
        if let Some((messages, info)) = compact(
            cfg.model.as_ref(),
            &s.messages,
            cfg.context.keep_recent,
            ctx.tracer.as_ref(),
            ctx.run.as_ref(),
        )
        .await?
        {
            s.messages = messages;
            s.compactions += 1;
            emit(&ctx, &AgentEvent::Compacted(info));
        }
    }

    emit(
        &ctx,
        &AgentEvent::Step {
            step: s.turn.steps + 1,
        },
    );
    let hctx = hook_ctx(cfg, &s, &ctx);
    let mut messages = Vec::with_capacity(s.messages.len() + 1);
    messages.push(Message::system(cfg.system_prompt.clone()));
    messages.extend(s.messages.iter().cloned());
    let mut req = ChatRequest::new(messages).with_tools(cfg.specs.clone());
    req.temperature = cfg.temperature;
    req.max_tokens = cfg.max_tokens;
    for h in &cfg.hooks {
        h.before_model(&hctx, &mut req).await?;
    }
    let estimated_input = estimate_messages(&req.messages) + cfg.spec_tokens;
    let mut resp = if ctx.is_streaming() {
        traced_chat_stream(
            cfg.model.as_ref(),
            req,
            ctx.tracer.as_ref(),
            ctx.run.as_ref(),
            |ev| {
                if let StreamEvent::TextDelta { text } = ev {
                    ctx.emit_token(text.clone());
                }
            },
        )
        .await?
    } else {
        traced_chat(
            cfg.model.as_ref(),
            req,
            ctx.tracer.as_ref(),
            ctx.run.as_ref(),
        )
        .await?
    };
    for h in &cfg.hooks {
        h.after_model(&hctx, &mut resp).await?;
    }

    let mut usage = resp.usage;
    if usage.input_tokens == 0 {
        usage.input_tokens = estimated_input as u32;
    }
    if usage.output_tokens == 0 {
        usage.output_tokens = crate::context::estimate_message_tokens(&resp.message) as u32;
    }
    s.turn.usage.add(usage, cfg.pricing.as_ref());
    s.usage.add(usage, cfg.pricing.as_ref());
    s.turn.steps += 1;

    let mut msg = resp.message;
    msg.role = Role::Assistant;
    msg.tool_call_id = None;
    let step = s.turn.steps;
    for (i, c) in msg.tool_calls.iter_mut().enumerate() {
        if c.id.is_empty() {
            c.id = format!("call_{step}_{i}");
        }
    }

    if msg.tool_calls.is_empty() {
        match apply_guardrails(&cfg.guardrails, GuardrailStage::Output, &msg.content) {
            Ok(text) => msg.content = text,
            Err(reason) => {
                msg.content = format!("[answer withheld: {reason}]");
                s.turn.stop = Some(StopReason::Guardrail(reason));
            }
        }
        s.messages.push(msg);
        s.pending.clear();
        return Ok(s);
    }

    let mut pending = Vec::with_capacity(msg.tool_calls.len());
    for call in &mut msg.tool_calls {
        let decision = decide(cfg, &s, &hctx, call).await;
        pending.push(PendingCall {
            call: call.clone(),
            decision,
        });
    }
    s.messages.push(msg);
    if let Some(guard) = &cfg.loop_guard {
        let check = guard.check(&s.messages);
        if check.is_loop() {
            if check.stuck && guard.action != LoopAction::Interrupt
                || guard.action == LoopAction::Stop
            {
                for p in &pending {
                    s.messages.push(Message::tool_result(
                        p.call.id.clone(),
                        "skipped: the turn was stopped by the loop guard",
                    ));
                }
                s.messages.push(Message::assistant(LOOP_STOP_REPLY));
                s.turn.stop = Some(StopReason::Guardrail(if check.stuck {
                    format!(
                        "loop detected: {} failing tool calls in a row",
                        check.consecutive_errors
                    )
                } else {
                    "loop detected: repeated tool calls".into()
                }));
                return Ok(s);
            }
            for r in &check.repeated {
                if let Some(p) = pending.iter_mut().find(|p| p.call.id == r.id) {
                    let reason = LoopGuard::nudge_message(r)
                        .trim_start_matches("error: ")
                        .to_string();
                    p.decision = match guard.action {
                        LoopAction::Interrupt => CallDecision::Ask { reason },
                        _ => CallDecision::Deny { reason },
                    };
                }
            }
            if check.stuck {
                // Interrupt mode: ask about every call of a stuck turn.
                for p in &mut pending {
                    if p.decision == CallDecision::Allow {
                        p.decision = CallDecision::Ask {
                            reason: format!(
                                "loop detected: {} failing tool calls in a row",
                                check.consecutive_errors
                            ),
                        };
                    }
                }
            }
        }
    }
    s.pending = pending;
    Ok(s)
}

/// Hooks first (they may rewrite the call), then the policy on the final
/// call; the most restrictive verdict wins.
async fn decide(
    cfg: &Config,
    s: &HarnessState,
    hctx: &HookContext,
    call: &mut sinew_core::ToolCall,
) -> CallDecision {
    let mut perm = Permission::Allow;
    let mut reason = String::new();
    for h in &cfg.hooks {
        let (p, r) = match h.before_tool(hctx, call).await {
            ToolDecision::Allow => continue,
            ToolDecision::Ask(r) => (Permission::Ask, r),
            ToolDecision::Deny(r) => (Permission::Deny, r),
        };
        if p > perm {
            perm = p;
            reason = r;
        }
    }
    if cfg.tools.contains_key(&call.name) && !s.always_allow.contains(&call.name) {
        let p = cfg.policy.check(call);
        if p > perm {
            perm = p;
            reason = match p {
                Permission::Deny => format!("the permission policy denies `{}`", call.name),
                _ => format!(
                    "the permission policy requires approval for `{}`",
                    call.name
                ),
            };
        }
    }
    if perm == Permission::Ask && cfg.auto_approve {
        perm = Permission::Allow;
    }
    if perm == Permission::Ask && cfg.ask_as_deny {
        perm = Permission::Deny;
        reason = format!("{reason} (approval is not available inside a subagent)");
    }
    match perm {
        Permission::Allow => CallDecision::Allow,
        Permission::Ask => CallDecision::Ask { reason },
        Permission::Deny => CallDecision::Deny { reason },
    }
}

async fn tools_node(cfg: &Config, mut s: HarnessState, ctx: NodeContext) -> Result<HarnessState> {
    let pending = std::mem::take(&mut s.pending);
    let hctx = hook_ctx(cfg, &s, &ctx);
    if let Some(reason) = hook_stop(cfg, &hctx, StepPhase::BeforeTools).await {
        for p in &pending {
            let out = format!("skipped: {reason}");
            emit(
                &ctx,
                &AgentEvent::ToolEnd {
                    id: p.call.id.clone(),
                    name: p.call.name.clone(),
                    output: out.clone(),
                    is_error: true,
                },
            );
            s.messages
                .push(Message::tool_result(p.call.id.clone(), out));
        }
        s.turn.stop = Some(StopReason::Guardrail(reason));
        return Ok(s);
    }
    // Where these calls were requested, for durable ledger keys.
    let position = s.messages.len().saturating_sub(1).to_string();
    let unknown = match cfg.unknown_outcome {
        UnknownOutcome::Interrupt => UnknownOutcome::Fail,
        u => u,
    };

    // `write_todos` updates the state directly, in order.
    let mut outputs: Vec<Option<String>> = vec![None; pending.len()];
    for (i, p) in pending.iter().enumerate() {
        if p.call.name == todo::NAME && p.decision == CallDecision::Allow {
            emit(
                &ctx,
                &AgentEvent::ToolStart {
                    id: p.call.id.clone(),
                    name: p.call.name.clone(),
                    arguments: p.call.arguments.clone(),
                },
            );
            let (out, is_error) = match todo::parse_todos(&p.call.arguments) {
                Ok(todos) => {
                    s.todos = todos;
                    (
                        format!("Todo list updated:\n{}", todo::render_todos(&s.todos)),
                        false,
                    )
                }
                Err(e) => (format!("error: {e}"), true),
            };
            emit(
                &ctx,
                &AgentEvent::ToolEnd {
                    id: p.call.id.clone(),
                    name: p.call.name.clone(),
                    output: out.clone(),
                    is_error,
                },
            );
            outputs[i] = Some(out);
        }
    }

    // Everything else runs concurrently.
    let runs = pending.iter().zip(&outputs).map(|(p, done)| {
        let ctx = &ctx;
        let hctx = &hctx;
        let position = &position;
        async move {
            match done {
                Some(out) => out.clone(),
                None if p.decision == CallDecision::Allow => {
                    let side_effects = cfg
                        .tools
                        .get(&p.call.name)
                        .is_some_and(|t| t.side_effects());
                    durable_run(ctx, position, &p.call, side_effects, unknown, || {
                        run_tool(cfg, p, ctx, hctx)
                    })
                    .await
                    .unwrap_or_else(|e| format!("error: {e}"))
                }
                None => run_tool(cfg, p, ctx, hctx).await,
            }
        }
    });
    let results = join_all(runs).await;
    for (p, out) in pending.iter().zip(results) {
        s.messages
            .push(Message::tool_result(p.call.id.clone(), out));
    }
    Ok(s)
}

async fn run_tool(cfg: &Config, p: &PendingCall, ctx: &NodeContext, hctx: &HookContext) -> String {
    let call = &p.call;
    emit(
        ctx,
        &AgentEvent::ToolStart {
            id: call.id.clone(),
            name: call.name.clone(),
            arguments: call.arguments.clone(),
        },
    );
    let (mut out, mut is_error, ran) = match &p.decision {
        CallDecision::Deny { reason } | CallDecision::Ask { reason } => {
            (format!("error: permission denied: {reason}"), true, false)
        }
        CallDecision::Allow => match cfg.tools.get(&call.name) {
            None => (
                format!(
                    "error: tool `{}` does not exist. Available tools: {}",
                    call.name,
                    cfg.tools.keys().cloned().collect::<Vec<_>>().join(", ")
                ),
                true,
                false,
            ),
            Some(tool) => {
                let run = ctx.tracer.as_ref().map(|t| {
                    t.start(
                        call.name.clone(),
                        RunKind::Tool,
                        ctx.run.as_ref(),
                        call.arguments.clone(),
                    )
                });
                let r = tool.call(call.arguments.clone()).await;
                let (out, err) = match r {
                    Ok(Value::String(s)) => (s, false),
                    Ok(v) => (v.to_string(), false),
                    Err(e) => (format!("error: {e}"), true),
                };
                if let Some(run) = run {
                    if err {
                        run.fail(&out["error: ".len()..]);
                    } else {
                        run.finish(json!(out));
                    }
                }
                (out, err, true)
            }
        },
    };
    if ran {
        for h in &cfg.hooks {
            h.after_tool(hctx, call, &mut out).await;
        }
        match apply_guardrails(&cfg.guardrails, GuardrailStage::ToolOutput, &out) {
            Ok(t) => out = t,
            Err(reason) => {
                out = format!("error: tool output blocked by guardrail ({reason})");
                is_error = true;
            }
        }
        if let Some(threshold) = cfg.context.offload_threshold {
            out = offload_output(
                out,
                &cfg.offload_dir,
                &cfg.offload_display,
                &call.id,
                threshold,
                cfg.context.offload_preview,
            );
        }
    }
    emit(
        ctx,
        &AgentEvent::ToolEnd {
            id: call.id.clone(),
            name: call.name.clone(),
            output: out.clone(),
            is_error,
        },
    );
    out
}
