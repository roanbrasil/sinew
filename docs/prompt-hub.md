# Prompt hub

`sinew-scope` stores prompts as versioned, immutable templates. Applications pull them by name and
tag at runtime, so a prompt can change (and be rolled back) without a redeploy, and every trace can
record exactly which version produced it.

- A **version** has a template (`{variable}` placeholders, `{{` / `}}` for literal braces, the
  same syntax as `sinew_core::PromptTemplate`), optional chat messages, the input variables, a
  model config (any JSON: model spec, temperature, ...), metadata and a commit message.
- Each version gets a sequential **number** (`v1`, `v2`, ...) and a **commit**: the first 12 hex
  digits of a SHA-256 over the name, the parent commit and the content.
- Pushing content identical to the latest version stores nothing (`created: false`).
- **Tags** (`prod`, `staging`, `canary`, ...) point at a commit and can be moved at any time.

## References

| Reference | Resolves to |
|---|---|
| `name` or `name@latest` | the newest version |
| `name@prod` | the version the `prod` tag points at (tags are checked first) |
| `name@v3` or `name@3` | version number 3 |
| `name@1a2b3c4d` | the version whose commit starts with these hex digits (4 or more; an ambiguous prefix is an error) |

Names and tags use letters, digits, `-`, `_` and `.` (up to 100 characters, starting with a letter
or digit). `latest` cannot be a tag.

## Rust client: `PromptHub`

`sinew_eval::PromptHub` (with the umbrella crate: `sinew::eval::PromptHub`, feature `eval`):

```rust
use std::{collections::HashMap, time::Duration};
use serde_json::json;
use sinew_eval::{PromptHub, PromptPush};

let hub = PromptHub::new("http://127.0.0.1:7878")      // or PromptHub::from_env()
    .with_ttl(Duration::from_secs(60));                  // cache for tags / latest

// Push (returns {name, commit, version, created}).
hub.push(
    PromptPush::new("support-answer", "Answer {question} for {company}.")
        .description("customer support")
        .model_config(json!({"model": "openai:gpt-4.1-mini", "temperature": 0.2}))
        .message("first version")
        .tag("prod"),
).await?;

// Chat prompts: one template per message.
hub.push(PromptPush::chat("qa", [("system", "Reply in {lang}."), ("user", "{question}")])).await?;

// Pull as a sinew-core PromptTemplate...
let template = hub.pull("support-answer@prod").await?;
let text = template.render(&HashMap::from([
    ("question", "Where is my order?".to_string()),
    ("company", "Acme".to_string()),
]))?;

// ...or the full version (commit, tags, model config, chat messages).
let v = hub.pull_version("qa@latest").await?;
let messages = v.render_messages(&HashMap::from([
    ("lang", "French".to_string()),
    ("question", "2 + 2?".to_string()),
]))?;                                                   // Vec<sinew_core::Message>

// Move a tag, list versions.
hub.tag("support-answer", "prod", &v.commit).await?;
let versions = hub.versions("support-answer").await?;  // newest first
```

### Caching

`pull` and `pull_version` keep an in-memory cache shared by clones of the hub:

- references that are full commits (12+ hex digits) never expire: versions are immutable;
- tags, `latest`, version numbers and short prefixes are refetched after the TTL (default 60 s;
  `with_ttl(Duration::ZERO)` always asks the server);
- when the server cannot be reached (network error, timeout, 5xx), a stale cached copy is returned
  instead of an error, so a scope outage does not take the application down. "Not found" is
  never masked;
- `push` and `tag` invalidate the prompt's tag/`latest` entries; `invalidate(name)` and
  `clear_cache()` are available too.

A good pattern is to record the resolved commit on the trace:

```rust
let v = hub.pull_version("support-answer@prod").await?;
root.set_metadata("prompt", json!(format!("support-answer@{}", v.commit)));
```

## HTTP API

| Method | Path | Body / result |
|---|---|---|
| `GET` | `/api/prompts` | `[{name, description, created_ms, updated_ms, version_count, latest_commit, latest_version, tags: {tag: commit}}]` |
| `POST` | `/api/prompts` | push: `{name, template, messages?, description?, model_config?, metadata?, message?, tags?}` → `{name, commit, version, created, prompt}` |
| `GET` | `/api/prompts/{name}` | `{name, description, created_ms, updated_ms, tags, versions}` (versions newest first) |
| `GET` | `/api/prompts/{name}@{ref}` | one version (same as below) |
| `GET` | `/api/prompts/{name}/{ref}` | one version: `{name, commit, version, template, messages, input_variables, model_config, metadata, message, parent_commit, created_ms, tags}` |
| `GET` | `/api/prompts/{name}/diff?from={ref}&to={ref}` | `{from, to, template: [{op: "eq" / "add" / "del", text}], variables_added, variables_removed, model_config: [{key, from, to}]}`; `to` defaults to `latest` |
| `PUT` | `/api/prompts/{name}/tags/{tag}` | `{"commit": "<any ref>"}` → the tagged version |
| `DELETE` | `/api/prompts/{name}/tags/{tag}` | remove a tag |
| `DELETE` | `/api/prompts/{name}` | delete the prompt, its versions and tags |

Invalid templates (an unclosed `{`, a bad variable name) and unknown message roles are rejected
with `400`. Chat messages use roles `system`, `user` and `assistant`.

```bash
curl -s localhost:7878/api/prompts -H 'content-type: application/json' \
  -d '{"name": "greeter", "template": "Say hi to {name}.", "tags": ["prod"]}'
curl -s localhost:7878/api/prompts/greeter@prod
curl -s 'localhost:7878/api/prompts/greeter/diff?from=v1&to=latest'
```

## Web UI

`#/prompts` lists prompts with their tags and creates new ones. `#/prompt/<name>` shows the
version history on the left and the selected version on the right (template or chat messages,
variables, model config, a ready-to-paste `PromptHub::pull` line). From there you can:

- **Edit as new version**: a form prefilled with the version; committing pushes a new version;
- **Point tag here** / **untag**: promote a version to `prod` or roll back;
- **Compare with…**: a line diff of the templates plus changed variables and model-config keys.

## Example

`samples/21-prompt-hub-and-feedback` pushes two versions, tags them `prod` and `staging`, serves
traffic with `support-answer@prod`, and runs a pairwise experiment between the two tags.
