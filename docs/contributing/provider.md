# Adding a provider

Providers live in `crates/sinew-core/src/providers/`. There are two ways to add one:

| Path | Use it when | Work |
|---|---|---|
| [OpenAI-compatible preset](#path-a-openai-compatible-preset) | The provider exposes a Chat Completions endpoint (`POST {base}/chat/completions`) that `OpenAiCompat` already speaks | One table entry, one constructor, regenerated docs |
| [Native client](#path-b-native-client) | The API has its own request format (as Anthropic and Gemini do), or the compatible endpoint lacks features the native one has | A new module implementing `ChatModel`, with request/response/stream mapping and recorded-payload tests |

Start with a **New provider** issue: it asks for the base URL, the key variable, the capabilities and a link to
the API reference. If the provider has both, prefer the preset first; a native client can follow.

## The trait

Every provider is a `ChatModel` (`crates/sinew-core/src/model.rs`):

```rust
#[async_trait]
pub trait ChatModel: Send + Sync {
    /// Model identifier used in traces.
    fn name(&self) -> &str;
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse>;

    /// Streams the response as [`StreamEvent`](crate::StreamEvent)s.
    async fn chat_stream(&self, request: ChatRequest) -> Result<ChatStream> { /* default: replays chat() */ }

    /// What this model supports; see [`Capabilities`]. The default reports
    /// nothing; providers and wrappers override it.
    fn capabilities(&self) -> Capabilities {
        Capabilities::none()
    }
}
```

`Capabilities` has one `bool` per feature: `tools`, `streaming`, `vision`, `pdf`, `audio`, `json_schema`,
`cached_tokens`, `sampling`, `forced_tool_choice`. List a feature only when the provider documents it for the
endpoint Sinew calls. When unsure, leave it off: callers check capabilities before sending images or forcing a
tool, and an unlisted feature fails safe.

## Path A: OpenAI-compatible preset

All presets are in `crates/sinew-core/src/providers/presets.rs`.

1. **Add a table entry.** Hosted services go in `HOSTED`, servers you run yourself in `LOCAL`:

   ```rust
   // HOSTED
   Hosted {
       name: "acme",                       // lowercase; the `acme:` in `acme:model`
       aliases: &["acme-ai"],              // optional extra spellings
       base_url: "https://api.acme.ai/v1", // must start with https://
       key_var: "ACME_API_KEY",            // must end with _KEY (or be HF_TOKEN)
       caps: caps(true, true, false, false, true, false, true),
   },

   // LOCAL
   Local {
       name: "acmeserve",
       aliases: &[],
       default_host: "http://127.0.0.1:9000",
       host_var: "ACMESERVE_HOST",
       caps: caps(true, false, false, false, false, false, false),
   },
   ```

   The arguments of `caps` are, in order: `tools`, `vision`, `pdf`, `audio`, `json_schema`, `cached_tokens`,
   `forced_tool_choice` (`streaming` and `sampling` are always on for presets). The table order is the order of
   `providers::catalog()` and of the generated docs.

2. **Add a constructor** in the `impl OpenAiCompat` block of the same file, with the variable in the doc comment:

   ```rust
   /// Acme (`ACME_API_KEY`).
   pub fn acme(model: impl Into<String>) -> Result<Self> {
       hosted("acme", model)
   }
   /// AcmeServe (`ACMESERVE_HOST` or 127.0.0.1:9000).
   pub fn acmeserve(model: impl Into<String>) -> Self {
       local("acmeserve", model)
   }
   ```

   `from_spec`, `provider_names`, `catalog` and `capabilities_for` read the tables, so nothing else needs to
   change for `"acme:model"` to work.

3. **Quirks.** If the provider deviates from the OpenAI format (a field it rejects, a different usage block, an
   extra header), handle it in `crates/sinew-core/src/providers/openai.rs` behind a narrow condition, or set
   headers with `OpenAiCompat::with_header` / `with_query` / `with_api_key_header` in the constructor. Every
   deviation needs a recorded-payload test (below).

4. **Tests.**
   - `presets.rs` tests: add a local preset's name to the list in
     `spec_resolves_local_providers_and_keeps_model_tags`. `hosted_presets_need_their_key` checks hosted entries.
   - `crates/sinew-core/tests/provider_matrix.rs` checks that `catalog()`, `provider_names()` and
     `capabilities_for()` agree; it needs no change.
   - For each quirk, a unit test in `openai.rs` that encodes a `ChatRequest` with `encode_request` or decodes a
     **recorded** response with `decode_response` / a recorded SSE body with `decode_all` (see Path B, step 4).

5. **Regenerate the provider tables** in `docs/providers.md`:

   ```bash
   SINEW_UPDATE_DOCS=1 cargo test -p sinew-core --test provider_matrix
   ```

   Without the variable, the same test fails when the tables are stale, so CI catches a forgotten regeneration.
   Add a bullet to the **Notes** below the tables if the provider has behavior users must know about.

## Path B: native client

Use `anthropic.rs` and `gemini.rs` as references; they follow the same layout.

1. **Create the module** `crates/sinew-core/src/providers/acme.rs`, and register it in
   `crates/sinew-core/src/providers/mod.rs`:

   ```rust
   pub mod acme;
   pub use acme::Acme;
   ```

2. **Client struct and constructors.** Keep the conventions of the existing clients:

   ```rust
   #[derive(Clone)]
   pub struct Acme {
       base_url: String,
       api_key: String,
       model: String,
       client: reqwest::Client,
   }

   impl Acme {
       pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
           Self {
               base_url: "https://api.acme.ai/v1".into(),
               api_key: api_key.into(),
               model: model.into(),
               client: super::shared_client(), // one connection pool for all providers
           }
       }

       /// Reads `ACME_API_KEY`.
       pub fn from_env(model: impl Into<String>) -> Result<Self> {
           let key = std::env::var("ACME_API_KEY")
               .map_err(|_| Error::Config("ACME_API_KEY not set".into()))?;
           Ok(Self::new(key, model))
       }

       pub fn with_base_url(mut self, url: impl Into<String>) -> Self { /* ... */ }
       pub fn with_http_client(mut self, client: reqwest::Client) -> Self { /* ... */ }
   }
   ```

3. **Mapping functions.** Keep them pure and `pub(crate)` so tests can call them without a network:

   - `encode_request(model: &str, req: &ChatRequest) -> Value`: messages (system, user, assistant, tool results),
     multimodal `ContentPart`s, tools and `ToolChoice`, `temperature`, `max_tokens`, `ResponseFormat`. Drop
     parameters the model rejects instead of sending them; see `accepts_sampling` and
     `accepts_forced_tool_choice` in `anthropic.rs`.
   - `decode_response(model: &str, v: &Value) -> Result<ChatResponse>`: text, tool calls, `Usage` (including
     `cached_input_tokens` when the API reports cache hits). Return `Error::InvalidResponse` for bodies without
     the expected fields.
   - A stream decoder implementing the crate's `SseDecoder` (`crates/sinew-core/src/stream.rs`):

     ```rust
     pub(crate) trait SseDecoder: Send + 'static {
         fn on_event(&mut self, ev: &SseEvent) -> Result<Vec<StreamEvent>>;
         /// Called once when the provider signalled the end or the body ended.
         fn finish(&mut self) -> Vec<StreamEvent>;
         /// The provider sent its end-of-stream marker.
         fn is_done(&self) -> bool;
     }
     ```

     Turn in-stream error events into `Error::Stream`.

   Then implement the trait with the shared helpers, which map HTTP errors to `Error::Provider` and 429 /
   `Retry-After` to `Error::RateLimited`:

   ```rust
   #[async_trait]
   impl ChatModel for Acme {
       fn name(&self) -> &str {
           &self.model
       }
       fn capabilities(&self) -> crate::Capabilities {
           acme_capabilities(&self.model)
       }
       async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
           let body = encode_request(&self.model, &request);
           let v = super::post_json("acme", self.request(), &body).await?;
           decode_response(&self.model, &v)
       }
       async fn chat_stream(&self, request: ChatRequest) -> Result<ChatStream> {
           let mut body = encode_request(&self.model, &request);
           body["stream"] = json!(true);
           let resp = super::post_sse("acme", self.request(), &body).await?;
           Ok(sse_stream(resp, AcmeStreamDecoder::new(&self.model)))
       }
   }
   ```

4. **Recorded-payload tests** in a `#[cfg(test)] mod tests` at the bottom of the module. Record real bodies once
   (with `curl` or the provider's console), strip ids and secrets, and paste them as constants:

   - request encoding: system prompt placement, tool results, images/documents, tool choice, dropped parameters;
   - response decoding: text, tool calls, usage, cached tokens;
   - streaming: a recorded SSE body with text, a tool call split across chunks and the final usage, decoded with
     `crate::stream::decode_all(AcmeStreamDecoder::new("m"), BODY)` and checked with `StreamAccumulator`;
   - an in-stream error event.

   `TOOL_STREAM` and `stream_decoder_handles_text_tool_use_and_usage` in `anthropic.rs` are a complete example.
   A live test, if you add one, goes in `crates/sinew-core/tests/` with `#[ignore = "needs ACME_API_KEY"]`.

5. **Wire it into specs and the catalog** in `presets.rs`:
   - a match arm in `from_spec` (`"acme" => return Ok(Arc::new(Acme::from_env(model)?))`);
   - the name in `provider_names()` (same position as in `catalog()`);
   - a `ProviderInfo` with `kind: ProviderKind::Native` in `catalog()`;
   - an arm in `capabilities_for()`.

   `provider_matrix.rs` fails if these disagree.

6. **Regenerate `docs/providers.md`** with `SINEW_UPDATE_DOCS=1 cargo test -p sinew-core --test provider_matrix`,
   and add a Notes bullet for model-specific behavior.

## Prices (optional)

If the provider publishes per-token prices, add entries to `BUILTIN_PRICES` in
`crates/sinew-core/src/pricing.rs` (USD per million tokens, matched by model-id prefix) so usage cost is
reported. Users can always set their own with `pricing::set_price`.

## Feature flag

Providers are not feature-gated: they only use `reqwest` and `serde_json`, which `sinew-core` already depends on.
If a native client needs a new dependency (a signing library, a protocol crate), put it behind a feature in
`crates/sinew-core/Cargo.toml` and forward it from `crates/sinew/Cargo.toml`.

## Docs to update

- `docs/providers.md`: regenerated tables, plus Notes.
- `README.md`: the provider list in the `sinew-core` row of the crate table.
- `docs/troubleshooting.md`: provider-specific errors users are likely to hit (a 400 on a parameter, a required
  header).

## CHANGELOG

Under `## [Unreleased]` → `### Added`:

```markdown
- Acme provider: `acme:<model>` spec, `OpenAiCompat::acme` (`ACME_API_KEY`), capabilities in the provider matrix (#123).
```
