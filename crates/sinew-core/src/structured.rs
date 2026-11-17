//! Typed output from a chat model.
//!
//! [`StructuredOutput`] asks the model for JSON matching a schema, checks the
//! reply against that schema, deserializes it into your type and, if any of
//! that fails, tells the model what went wrong and asks again (once by
//! default).
//!
//! ```no_run
//! # use sinew_core::*;
//! # async fn demo(model: &dyn ChatModel) -> Result<()> {
//! #[derive(serde::Deserialize)]
//! struct Person { name: String, age: u32 }
//!
//! let schema = serde_json::json!({
//!     "type": "object",
//!     "properties": { "name": {"type": "string"}, "age": {"type": "integer"} },
//!     "required": ["name", "age"]
//! });
//! let p: Person = StructuredOutput::new("person", schema)
//!     .generate(model, ChatRequest::new(vec![Message::user("Ada Lovelace, 36")]))
//!     .await?;
//! # Ok(()) }
//! ```

use serde::de::DeserializeOwned;
use serde_json::{json, Value};

use crate::{ChatModel, ChatRequest, Error, Message, ResponseFormat, Result, ToolChoice, ToolSpec};

/// How the schema is presented to the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StructuredMode {
    /// Offer a single tool whose parameters are the schema and force the
    /// model to call it. Works with any tool-calling model (default).
    #[default]
    Tool,
    /// Ask for a JSON text reply: the schema goes into a system message and
    /// into `response_format` for providers that support it.
    Json,
}

/// Request builder for typed model output. See the [module docs](self).
#[derive(Debug, Clone)]
pub struct StructuredOutput {
    name: String,
    description: String,
    schema: Value,
    mode: StructuredMode,
    retries: usize,
}

impl StructuredOutput {
    /// `name` names the tool / schema (letters, digits, `_` and `-`).
    pub fn new(name: impl Into<String>, schema: Value) -> Self {
        Self {
            name: name.into(),
            description: "Return the final answer in this exact structure.".into(),
            schema,
            mode: StructuredMode::Tool,
            retries: 1,
        }
    }

    /// Derives the schema from `T` with `schemars`.
    #[cfg(feature = "schemars")]
    pub fn for_type<T: schemars::JsonSchema>() -> Self {
        let schema = serde_json::to_value(schemars::schema_for!(T)).unwrap_or_default();
        let name: String = T::schema_name()
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        Self::new(name, schema)
    }

    pub fn with_mode(mut self, mode: StructuredMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Extra attempts after a reply that fails to parse or validate (default 1).
    pub fn with_retries(mut self, retries: usize) -> Self {
        self.retries = retries;
        self
    }

    pub fn schema(&self) -> &Value {
        &self.schema
    }

    /// Runs the request and returns the parsed value.
    pub async fn generate<T, M>(&self, model: &M, request: ChatRequest) -> Result<T>
    where
        T: DeserializeOwned,
        M: ChatModel + ?Sized,
    {
        let mut request = self.prepare(request);
        let mut attempt = 0;
        loop {
            let resp = model.chat(request.clone()).await?;
            let (raw, call_id) = self.extract(&resp.message);
            match self.parse::<T>(&raw) {
                Ok(v) => return Ok(v),
                Err(message) if attempt >= self.retries => {
                    return Err(Error::Structured { message, raw })
                }
                Err(message) => {
                    attempt += 1;
                    let fix = format!(
                        "That output was invalid: {message}. Reply again with JSON that matches the schema exactly."
                    );
                    request.messages.push(resp.message.clone());
                    match call_id {
                        Some(id) => request.messages.push(Message::tool_result(id, fix)),
                        None => request.messages.push(Message::user(fix)),
                    }
                }
            }
        }
    }

    /// Parses and validates one assistant message without calling a model:
    /// the same extraction, schema check and deserialization as
    /// [`generate`](Self::generate), but no retry. Useful to check a reply
    /// produced elsewhere, for example in a
    /// [`router::Validator`](crate::router::Validator).
    ///
    /// ```
    /// use sinew_core::{Message, StructuredOutput};
    ///
    /// let so = StructuredOutput::new("n", serde_json::json!({
    ///     "type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]
    /// }));
    /// let v: serde_json::Value = so.parse_message(&Message::assistant("{\"n\": 3}")).unwrap();
    /// assert_eq!(v["n"], 3);
    /// assert!(so.parse_message::<serde_json::Value>(&Message::assistant("three")).is_err());
    /// ```
    pub fn parse_message<T: DeserializeOwned>(&self, message: &Message) -> Result<T> {
        let (raw, _) = self.extract(message);
        self.parse(&raw)
            .map_err(|message| Error::Structured { message, raw })
    }

    fn prepare(&self, mut req: ChatRequest) -> ChatRequest {
        match self.mode {
            StructuredMode::Tool => {
                req.tools = vec![ToolSpec {
                    name: self.name.clone(),
                    description: self.description.clone(),
                    parameters: self.schema.clone(),
                }];
                req.tool_choice = Some(ToolChoice::Tool(self.name.clone()));
                req.messages.insert(
                    0,
                    Message::system(format!(
                        "Answer by calling the `{}` tool exactly once with the complete result.",
                        self.name
                    )),
                );
            }
            StructuredMode::Json => {
                req.response_format = Some(ResponseFormat::JsonSchema {
                    name: self.name.clone(),
                    schema: self.schema.clone(),
                });
                req.messages.insert(
                    0,
                    Message::system(format!(
                        "{} Reply with a single JSON value and nothing else. It must match this JSON Schema:\n{}",
                        self.description, self.schema
                    )),
                );
            }
        }
        req
    }

    /// The raw JSON text and, for tool mode, the id of the call to answer.
    fn extract(&self, msg: &Message) -> (String, Option<String>) {
        if let Some(call) = msg
            .tool_calls
            .iter()
            .find(|c| c.name == self.name)
            .or(msg.tool_calls.first())
        {
            let raw = match &call.arguments {
                Value::String(s) => s.clone(),
                v => v.to_string(),
            };
            return (raw, Some(call.id.clone()));
        }
        (msg.content.clone(), None)
    }

    fn parse<T: DeserializeOwned>(&self, raw: &str) -> std::result::Result<T, String> {
        let text = extract_json(raw).ok_or_else(|| "no JSON value found".to_string())?;
        let value: Value = serde_json::from_str(text).map_err(|e| format!("invalid JSON: {e}"))?;
        validate(&self.schema, &value, "$")?;
        serde_json::from_value(value).map_err(|e| format!("does not fit the expected type: {e}"))
    }
}

/// Convenience for [`StructuredOutput::new`] in tool mode.
pub async fn structured<T, M>(model: &M, request: ChatRequest, schema: Value) -> Result<T>
where
    T: DeserializeOwned,
    M: ChatModel + ?Sized,
{
    StructuredOutput::new("respond", schema)
        .generate(model, request)
        .await
}

/// Finds the JSON in a reply: strips Markdown fences and surrounding prose.
pub(crate) fn extract_json(raw: &str) -> Option<&str> {
    let mut s = raw.trim();
    if let Some(rest) = s.strip_prefix("```") {
        let rest = rest.trim_start_matches(|c: char| c.is_ascii_alphanumeric());
        s = rest.trim_end().strip_suffix("```").unwrap_or(rest).trim();
    }
    if serde_json::from_str::<Value>(s).is_ok() {
        return Some(s);
    }
    let start = s.find(['{', '['])?;
    let close = if s.as_bytes()[start] == b'{' {
        '}'
    } else {
        ']'
    };
    let end = s.rfind(close)?;
    (end > start).then(|| &s[start..=end])
}

/// Checks the common JSON Schema keywords: `type`, `required`, `properties`,
/// `items`, `enum` and `additionalProperties: false`. Unknown keywords are
/// ignored; serde does the final, exact check against the Rust type.
pub fn validate(schema: &Value, value: &Value, path: &str) -> std::result::Result<(), String> {
    if let Some(ty) = schema.get("type") {
        let types: Vec<&str> = match ty {
            Value::String(t) => vec![t.as_str()],
            Value::Array(ts) => ts.iter().filter_map(Value::as_str).collect(),
            _ => vec![],
        };
        if !types.is_empty() && !types.iter().any(|t| type_matches(t, value)) {
            return Err(format!(
                "{path} should be {}, got {value}",
                types.join(" or ")
            ));
        }
    }
    if let Some(options) = schema.get("enum").and_then(Value::as_array) {
        if !options.contains(value) {
            return Err(format!("{path} must be one of {}", json!(options)));
        }
    }
    if let Value::Object(obj) = value {
        for key in schema
            .get("required")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
        {
            if !obj.contains_key(key) {
                return Err(format!("{path} is missing required field `{key}`"));
            }
        }
        let props = schema.get("properties").and_then(Value::as_object);
        for (k, v) in obj {
            match props.and_then(|p| p.get(k)) {
                Some(sub) => validate(sub, v, &format!("{path}.{k}"))?,
                None if schema.get("additionalProperties") == Some(&Value::Bool(false)) => {
                    return Err(format!("{path} has unexpected field `{k}`"))
                }
                None => {}
            }
        }
    }
    if let (Value::Array(items), Some(sub)) = (value, schema.get("items")) {
        for (i, item) in items.iter().enumerate() {
            validate(sub, item, &format!("{path}[{i}]"))?;
        }
    }
    Ok(())
}

fn type_matches(t: &str, v: &Value) -> bool {
    match t {
        "object" => v.is_object(),
        "array" => v.is_array(),
        "string" => v.is_string(),
        "boolean" => v.is_boolean(),
        "null" => v.is_null(),
        "number" => v.is_number(),
        "integer" => v.is_i64() || v.is_u64(),
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{mock::ScriptedModel, Role, ToolCall};
    use serde::Deserialize;

    #[derive(Debug, Deserialize, PartialEq)]
    struct Person {
        name: String,
        age: u32,
    }

    fn schema() -> Value {
        json!({
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"]
        })
    }

    fn call(args: Value) -> Message {
        Message::assistant_with_tools(
            "",
            vec![ToolCall {
                id: "c1".into(),
                name: "person".into(),
                arguments: args,
            }],
        )
    }

    #[tokio::test]
    async fn tool_mode_forces_the_tool_and_parses_arguments() {
        let model = ScriptedModel::new([call(json!({"name": "Ada", "age": 36}))]);
        let p: Person = StructuredOutput::new("person", schema())
            .generate(&model, ChatRequest::new(vec![Message::user("Ada, 36")]))
            .await
            .unwrap();
        assert_eq!(
            p,
            Person {
                name: "Ada".into(),
                age: 36
            }
        );
        let req = &model.requests()[0];
        assert_eq!(req.tools[0].name, "person");
        assert_eq!(req.tool_choice, Some(ToolChoice::Tool("person".into())));
    }

    #[tokio::test]
    async fn retries_once_with_the_validation_error() {
        let model = ScriptedModel::new([
            call(json!({"name": "Ada", "age": "thirty-six"})),
            call(json!({"name": "Ada", "age": 36})),
        ]);
        let p: Person = StructuredOutput::new("person", schema())
            .generate(&model, ChatRequest::new(vec![Message::user("Ada")]))
            .await
            .unwrap();
        assert_eq!(p.age, 36);
        let retry = &model.requests()[1];
        let last = retry.messages.last().unwrap();
        assert_eq!(last.role, Role::Tool);
        assert!(
            last.content.contains("$.age should be integer"),
            "{}",
            last.content
        );
    }

    #[tokio::test]
    async fn gives_up_after_the_retry() {
        let model = ScriptedModel::new([
            Message::assistant("no idea"),
            Message::assistant("{\"name\": \"Ada\"}"),
        ]);
        let err = StructuredOutput::new("person", schema())
            .with_mode(StructuredMode::Json)
            .generate::<Person, _>(&model, ChatRequest::new(vec![Message::user("Ada")]))
            .await
            .unwrap_err();
        assert!(
            matches!(&err, Error::Structured { message, .. } if message.contains("`age`")),
            "{err}"
        );
        // Json mode puts the schema in the prompt and response_format.
        let req = &model.requests()[0];
        assert!(req.messages[0].content.contains("\"required\""));
        assert!(matches!(
            req.response_format,
            Some(ResponseFormat::JsonSchema { .. })
        ));
        assert_eq!(
            model.requests()[1].messages.last().unwrap().role,
            Role::User
        );
    }

    #[tokio::test]
    async fn json_mode_tolerates_fences_and_prose() {
        let model = ScriptedModel::new([Message::assistant(
            "Sure!\n```json\n{\"name\": \"Grace\", \"age\": 85}\n```",
        )]);
        let p: Person = StructuredOutput::new("person", schema())
            .with_mode(StructuredMode::Json)
            .generate(&model, ChatRequest::new(vec![]))
            .await
            .unwrap();
        assert_eq!(p.name, "Grace");
        assert_eq!(extract_json("text {\"a\": [1]} more"), Some("{\"a\": [1]}"));
    }

    #[cfg(feature = "schemars")]
    #[tokio::test]
    async fn derives_schema_from_type() {
        #[derive(Debug, Deserialize, schemars::JsonSchema)]
        struct Point {
            x: i64,
            y: i64,
        }
        let so = StructuredOutput::for_type::<Point>();
        assert_eq!(so.schema()["required"], json!(["x", "y"]));
        let model = ScriptedModel::new([Message::assistant_with_tools(
            "",
            vec![ToolCall {
                id: "c".into(),
                name: "Point".into(),
                arguments: json!({"x": 1, "y": 2}),
            }],
        )]);
        let p: Point = so.generate(&model, ChatRequest::new(vec![])).await.unwrap();
        assert_eq!((p.x, p.y), (1, 2));
    }

    #[test]
    fn validator_checks_nested_items_enum_and_extra_fields() {
        let s = json!({"type": "object", "additionalProperties": false,
            "properties": {"tags": {"type": "array", "items": {"enum": ["a", "b"]}}}});
        assert!(validate(&s, &json!({"tags": ["a", "b"]}), "$").is_ok());
        assert!(validate(&s, &json!({"tags": ["c"]}), "$")
            .unwrap_err()
            .contains("$.tags[0]"));
        assert!(validate(&s, &json!({"x": 1}), "$")
            .unwrap_err()
            .contains("unexpected"));
    }
}
