use std::{collections::BTreeMap, sync::Arc};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{Error, Message, Result, ToolCall};

/// Tool description sent to the model. `parameters` is a JSON Schema object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn spec(&self) -> ToolSpec;
    async fn call(&self, arguments: Value) -> Result<Value>;

    /// True when calling the tool changes something outside the program
    /// (sends a message, charges a card, writes a record), so running it twice
    /// is not safe. Durable runtimes then never repeat a call whose outcome
    /// is unknown after a crash without a policy saying so. Default `false`.
    fn side_effects(&self) -> bool {
        false
    }
}

tokio::task_local! {
    static IDEMPOTENCY_KEY: String;
}

/// The idempotency key of the tool call being executed, when the runtime
/// provides one (durable graph runs do). Stable across retries and resumes of
/// the same call, so a tool can pass it to an external API (an
/// `Idempotency-Key` header, a unique constraint) and let that system drop
/// duplicates.
///
/// ```
/// # async fn demo() {
/// use sinew_core::tool::{idempotency_key, with_idempotency_key};
/// assert_eq!(idempotency_key(), None);
/// let k = with_idempotency_key("thread-1/tools/3/call_a".into(), async { idempotency_key() }).await;
/// assert_eq!(k.as_deref(), Some("thread-1/tools/3/call_a"));
/// # }
/// ```
pub fn idempotency_key() -> Option<String> {
    IDEMPOTENCY_KEY.try_with(Clone::clone).ok()
}

/// Runs `f` with [`idempotency_key`] returning `key`.
pub async fn with_idempotency_key<F: std::future::Future>(key: String, f: F) -> F::Output {
    IDEMPOTENCY_KEY.scope(key, f).await
}

/// Wraps a synchronous closure as a [`Tool`].
pub struct FnTool<F> {
    spec: ToolSpec,
    f: F,
}

impl<F> FnTool<F>
where
    F: Fn(Value) -> Result<Value> + Send + Sync,
{
    pub fn new(name: &str, description: &str, parameters: Value, f: F) -> Self {
        Self {
            spec: ToolSpec {
                name: name.into(),
                description: description.into(),
                parameters,
            },
            f,
        }
    }
}

#[async_trait]
impl<F> Tool for FnTool<F>
where
    F: Fn(Value) -> Result<Value> + Send + Sync,
{
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }
    async fn call(&self, arguments: Value) -> Result<Value> {
        (self.f)(arguments)
    }
}

/// Wraps an async closure as a [`Tool`]: the usual shape for tools that call
/// a database, another service or the file system.
///
/// ```
/// # use sinew_core::{AsyncFnTool, Tool};
/// # use serde_json::{json, Value};
/// let lookup = AsyncFnTool::new(
///     "get_order",
///     "Fetch an order by id",
///     json!({"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}),
///     |args: Value| async move { Ok(json!({"id": args["id"], "status": "shipped"})) },
/// );
/// assert_eq!(lookup.spec().name, "get_order");
/// ```
pub struct AsyncFnTool<F> {
    spec: ToolSpec,
    f: F,
}

impl<F, Fut> AsyncFnTool<F>
where
    F: Fn(Value) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = Result<Value>> + Send,
{
    pub fn new(name: &str, description: &str, parameters: Value, f: F) -> Self {
        Self {
            spec: ToolSpec {
                name: name.into(),
                description: description.into(),
                parameters,
            },
            f,
        }
    }
}

#[async_trait]
impl<F, Fut> Tool for AsyncFnTool<F>
where
    F: Fn(Value) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = Result<Value>> + Send,
{
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }
    async fn call(&self, arguments: Value) -> Result<Value> {
        (self.f)(arguments).await
    }
}

/// A tool with typed arguments and output (feature `schemars`): the JSON
/// Schema comes from `A`, arguments are deserialized into `A` before the call
/// and the result is serialized from `R`.
///
/// ```
/// # #[cfg(feature = "schemars")] {
/// use sinew_core::{Tool, TypedTool};
///
/// #[derive(serde::Deserialize, schemars::JsonSchema)]
/// struct Add { a: f64, b: f64 }
///
/// let add = TypedTool::new("add", "Add two numbers", |x: Add| async move { Ok(x.a + x.b) });
/// assert_eq!(add.spec().parameters["required"], serde_json::json!(["a", "b"]));
/// # }
/// ```
#[cfg(feature = "schemars")]
pub struct TypedTool<A, F> {
    spec: ToolSpec,
    f: F,
    _args: std::marker::PhantomData<fn(A)>,
}

#[cfg(feature = "schemars")]
impl<A, R, F, Fut> TypedTool<A, F>
where
    A: serde::de::DeserializeOwned + schemars::JsonSchema + Send,
    R: Serialize,
    F: Fn(A) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = Result<R>> + Send,
{
    pub fn new(name: &str, description: &str, f: F) -> Self {
        let parameters = tool_schema::<A>();
        Self {
            spec: ToolSpec {
                name: name.into(),
                description: description.into(),
                parameters,
            },
            f,
            _args: std::marker::PhantomData,
        }
    }
}

#[cfg(feature = "schemars")]
#[async_trait]
impl<A, R, F, Fut> Tool for TypedTool<A, F>
where
    A: serde::de::DeserializeOwned + schemars::JsonSchema + Send,
    R: Serialize,
    F: Fn(A) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = Result<R>> + Send,
{
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }
    async fn call(&self, arguments: Value) -> Result<Value> {
        let args: A = serde_json::from_value(arguments).map_err(|e| Error::Tool {
            name: self.spec.name.clone(),
            message: format!("invalid arguments: {e}"),
        })?;
        let out = (self.f)(args).await?;
        Ok(serde_json::to_value(out)?)
    }
}

/// JSON Schema for tool parameters: subschemas inlined (no `$ref` / `$defs`,
/// which several providers reject), without `$schema` and `title`.
#[cfg(feature = "schemars")]
pub(crate) fn tool_schema<T: schemars::JsonSchema>() -> Value {
    let schema = schemars::generate::SchemaSettings::draft2020_12()
        .with(|s| s.inline_subschemas = true)
        .into_generator()
        .into_root_schema_for::<T>();
    let mut v = serde_json::to_value(schema).unwrap_or_default();
    if let Some(obj) = v.as_object_mut() {
        obj.remove("$schema");
        obj.remove("title");
    }
    // Recursive types keep `$ref`s into `$defs`; only drop them when unused.
    let uses_refs = v
        .get("properties")
        .is_some_and(|p| p.to_string().contains("\"$ref\""));
    if !uses_refs {
        if let Some(obj) = v.as_object_mut() {
            obj.remove("$defs");
        }
    }
    v
}

/// Registry of tools, keyed by name. Cheap to clone.
#[derive(Clone, Default)]
pub struct ToolBox {
    tools: BTreeMap<String, Arc<dyn Tool>>,
}

impl ToolBox {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with(mut self, tool: impl Tool + 'static) -> Self {
        self.register(tool);
        self
    }

    pub fn register(&mut self, tool: impl Tool + 'static) {
        let tool: Arc<dyn Tool> = Arc::new(tool);
        self.tools.insert(tool.spec().name, tool);
    }

    /// Registers an already shared tool.
    pub fn register_arc(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.spec().name, tool);
    }

    /// The tool registered under `name`.
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.get(name)
    }

    /// The registered tools, by name.
    pub fn tools(&self) -> impl Iterator<Item = &Arc<dyn Tool>> {
        self.tools.values()
    }

    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools.values().map(|t| t.spec()).collect()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    pub async fn call(&self, call: &ToolCall) -> Result<Value> {
        let tool = self
            .tools
            .get(&call.name)
            .ok_or_else(|| Error::ToolNotFound(call.name.clone()))?;
        tool.call(call.arguments.clone()).await
    }

    /// Runs a tool call and turns the outcome into a tool-result message.
    /// Failures become an `error: ...` message so the model can recover.
    pub async fn run(&self, call: &ToolCall) -> Message {
        let content = match self.call(call).await {
            Ok(Value::String(s)) => s,
            Ok(v) => v.to_string(),
            Err(e) => format!("error: {e}"),
        };
        Message::tool_result(call.id.clone(), content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn adder() -> FnTool<impl Fn(Value) -> Result<Value> + Send + Sync> {
        FnTool::new("add", "Adds a and b", json!({"type": "object"}), |args| {
            let a = args["a"].as_f64().ok_or(Error::Tool {
                name: "add".into(),
                message: "a".into(),
            })?;
            let b = args["b"].as_f64().unwrap_or(0.0);
            Ok(json!(a + b))
        })
    }

    #[tokio::test]
    async fn runs_registered_tool() {
        let tb = ToolBox::new().with(adder());
        let call = ToolCall {
            id: "c1".into(),
            name: "add".into(),
            arguments: json!({"a": 2, "b": 3}),
        };
        let msg = tb.run(&call).await;
        assert_eq!(msg.content, "5.0");
        assert_eq!(msg.tool_call_id.as_deref(), Some("c1"));
    }

    #[tokio::test]
    async fn unknown_tool_becomes_error_message() {
        let tb = ToolBox::new();
        let call = ToolCall {
            id: "c1".into(),
            name: "nope".into(),
            arguments: json!({}),
        };
        assert!(tb
            .run(&call)
            .await
            .content
            .starts_with("error: tool `nope`"));
    }
}
