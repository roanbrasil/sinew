//! Enforcing a [`ControlPlane`] in `ReactAgent` ([`ControlPlaneGate`]),
//! on a plain `ToolBox` ([`GovernedTool`]) and on any `ChatModel`
//! ([`GovernedModel`]).

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use sinew_core::{
    guardrails::GuardrailStage, Capabilities, ChatModel, ChatRequest, ChatResponse, ChatStream,
    Error, Tool, ToolBox, ToolCall, ToolSpec,
};
use sinew_graph::{
    prebuilt::{AgentGate, GateVerdict},
    NodeContext,
};

use crate::{
    control::ControlPlane,
    decision::{Decision, Verdict},
    request::{Identity, Request},
};

impl ControlPlane {
    /// An [`AgentGate`] for
    /// [`ReactAgent::gate`](sinew_graph::prebuilt::ReactAgent::gate): every
    /// model call and tool call of the agent is checked as `identity`.
    ///
    /// | Verdict | Before a model call | Before a tool call |
    /// |---|---|---|
    /// | agent suspended | run ends | run ends (`GateVerdict::Stop`) |
    /// | `Deny` | run ends | the model sees `error: denied: ...` |
    /// | `Ask` | interrupt | interrupt (`approval_required`), resume with `true` / `false` |
    /// | `Redact` | allowed | runs with the fields masked |
    ///
    /// The decisions are recorded under the node's trace run.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use sinew_core::{mock::ScriptedModel, ChatModel, ToolBox};
    /// use sinew_govern::{AgentProfile, ControlPlane, Identity};
    /// use sinew_graph::prebuilt::ReactAgent;
    ///
    /// let cp = ControlPlane::builder().agent(AgentProfile::new("support")).build();
    /// let model: Arc<dyn ChatModel> = Arc::new(ScriptedModel::new([]));
    /// let agent = ReactAgent::new(model, ToolBox::new())
    ///     .gate(Arc::new(cp.gate(Identity::agent("support").user("ana"))))
    ///     .compile()
    ///     .unwrap();
    /// # let _ = agent;
    /// ```
    pub fn gate(&self, identity: impl Into<Identity>) -> ControlPlaneGate {
        ControlPlaneGate {
            cp: self.clone(),
            identity: identity.into(),
        }
    }

    /// `toolbox` with every tool wrapped in a [`GovernedTool`] acting as
    /// `identity`.
    pub fn governed_toolbox(&self, toolbox: &ToolBox, identity: impl Into<Identity>) -> ToolBox {
        let identity = identity.into();
        let mut out = ToolBox::new();
        for t in toolbox.tools() {
            out.register(GovernedTool {
                inner: t.clone(),
                cp: self.clone(),
                identity: identity.clone(),
            });
        }
        out
    }

    /// `model` with every call checked as a model call by `identity` (the
    /// model is identified by its [`name`](ChatModel::name)).
    pub fn govern_model<M: ChatModel>(
        &self,
        model: M,
        identity: impl Into<Identity>,
    ) -> GovernedModel<M> {
        GovernedModel {
            inner: model,
            cp: self.clone(),
            identity: identity.into(),
            spec: None,
        }
    }
}

/// Enforces a [`ControlPlane`] in a `ReactAgent`; see
/// [`ControlPlane::gate`].
#[derive(Clone)]
pub struct ControlPlaneGate {
    cp: ControlPlane,
    identity: Identity,
}

impl ControlPlaneGate {
    fn request(&self, ctx: &NodeContext, req: Request) -> Request {
        let mut req = req.identity(&self.identity).thread(ctx.thread_id.clone());
        if let Some(run) = &ctx.run {
            req = req.trace(run.trace_id.clone());
        }
        req
    }
}

#[async_trait]
impl AgentGate for ControlPlaneGate {
    async fn before_model(&self, ctx: &NodeContext, model: &str) -> GateVerdict {
        let req = self.request(ctx, Request::model("", model));
        let v = self
            .cp
            .check_traced(req, ctx.tracer.as_ref(), ctx.run.as_ref())
            .await;
        match v.decision {
            Decision::Allow | Decision::Redact { .. } => GateVerdict::Allow,
            Decision::Ask { reason } => GateVerdict::Ask(reason),
            Decision::Deny { reason } => GateVerdict::Stop(reason),
        }
    }

    async fn before_tool(
        &self,
        ctx: &NodeContext,
        call: &mut ToolCall,
        side_effects: bool,
    ) -> GateVerdict {
        let req = self.request(
            ctx,
            Request::tool("", call.name.clone(), call.arguments.clone()).side_effects(side_effects),
        );
        let v = self
            .cp
            .check_traced(req, ctx.tracer.as_ref(), ctx.run.as_ref())
            .await;
        apply(&v, call);
        match v.decision {
            Decision::Deny { reason } if v.stop => GateVerdict::Stop(reason),
            Decision::Deny { reason } => GateVerdict::Deny(reason),
            Decision::Ask { reason } => GateVerdict::Ask(reason),
            Decision::Allow | Decision::Redact { .. } => GateVerdict::Allow,
        }
    }
}

/// Masks the verdict's redacted fields in the call's arguments.
pub(crate) fn apply(v: &Verdict, call: &mut ToolCall) {
    if !v.redact.is_empty() {
        call.arguments = v.redacted(&call.arguments);
    }
}

/// A tool that asks a [`ControlPlane`] before every call. `Deny` and `Ask`
/// (there is no approval channel on a bare tool) fail the call with
/// [`Error::Tool`]; `Redact` masks the fields before the inner tool runs.
/// Built by [`ControlPlane::governed_toolbox`].
pub struct GovernedTool {
    inner: Arc<dyn Tool>,
    cp: ControlPlane,
    identity: Identity,
}

#[async_trait]
impl Tool for GovernedTool {
    fn spec(&self) -> ToolSpec {
        self.inner.spec()
    }

    fn side_effects(&self) -> bool {
        self.inner.side_effects()
    }

    async fn call(&self, arguments: Value) -> sinew_core::Result<Value> {
        let name = self.inner.spec().name;
        let req = Request::tool("", name.clone(), arguments.clone())
            .identity(&self.identity)
            .side_effects(self.inner.side_effects());
        let v = self.cp.check(req).await;
        let reason = |r: &str| {
            let by = v.policy_id.as_deref().unwrap_or("default");
            format!("{r} (policy `{by}`)")
        };
        match &v.decision {
            Decision::Deny { reason: r } => Err(Error::Tool {
                name,
                message: format!("denied: {}", reason(r)),
            }),
            Decision::Ask { reason: r } => Err(Error::Tool {
                name,
                message: format!(
                    "needs approval, which this runtime cannot ask for: {}",
                    reason(r)
                ),
            }),
            Decision::Allow | Decision::Redact { .. } => {
                self.inner.call(v.redacted(&arguments)).await
            }
        }
    }
}

/// A `ChatModel` that asks a [`ControlPlane`] before every call; built by
/// [`ControlPlane::govern_model`]. A denied (or approval-requiring) call
/// fails with [`Error::Guardrail`] (`guard` = `control_plane`).
pub struct GovernedModel<M> {
    inner: M,
    cp: ControlPlane,
    identity: Identity,
    spec: Option<String>,
}

impl<M: ChatModel> GovernedModel<M> {
    /// Checks calls against this model id (e.g. the `provider:model` spec)
    /// instead of the model's name.
    pub fn as_model(mut self, spec: impl Into<String>) -> Self {
        self.spec = Some(spec.into());
        self
    }

    async fn check(&self) -> sinew_core::Result<()> {
        let model = self.spec.as_deref().unwrap_or(self.inner.name());
        let v = self
            .cp
            .check(Request::model("", model).identity(&self.identity))
            .await;
        match v.decision {
            Decision::Allow | Decision::Redact { .. } => Ok(()),
            Decision::Ask { reason } | Decision::Deny { reason } => Err(Error::Guardrail {
                guard: "control_plane".into(),
                stage: GuardrailStage::Input,
                reason,
            }),
        }
    }
}

#[async_trait]
impl<M: ChatModel> ChatModel for GovernedModel<M> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn capabilities(&self) -> Capabilities {
        self.inner.capabilities()
    }

    async fn chat(&self, request: ChatRequest) -> sinew_core::Result<ChatResponse> {
        self.check().await?;
        self.inner.chat(request).await
    }

    async fn chat_stream(&self, request: ChatRequest) -> sinew_core::Result<ChatStream> {
        self.check().await?;
        self.inner.chat_stream(request).await
    }
}
