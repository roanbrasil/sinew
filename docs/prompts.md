# Prompt templates

Sinew has two ways to write a prompt template:

| Way | Checked | Use it when |
|---|---|---|
| [`prompt!` / `chat_prompt!`](#compile-time-checked-prompts) | at compile time | The template is in your source tree (a literal or a file). |
| [`PromptTemplate` / `ChatPromptTemplate`](#runtime-templates) | at runtime | The template arrives at runtime: config, a database, the [prompt hub](prompt-hub.md). |

Both use the same syntax and render the same text, and the compile-time form converts into the runtime
types, so you can switch between them without changing a template.

## Template syntax

- `{name}` is a variable. Spaces inside the braces are ignored: `{ name }` is the variable `name`.
- Variable names use letters, digits and `_`.
- `{{` and `}}` are a literal `{` and `}`. Use them for JSON examples: `Reply as {{"answer": "..."}}`.
- An unclosed `{`, a `}` without a matching `{`, or an empty or invalid name (`{}`, `{first name}`) is an
  error: a compile error with `prompt!`, `Error::Template` from `PromptTemplate::new`.

## Compile-time checked prompts

```rust
use sinew::prelude::*;

let text = prompt!("Summarize {text} in {lang}")
    .text(doc)
    .lang("pt-BR")
    .format();                       // String
```

`prompt!` parses the template while your crate compiles and returns a builder with one setter per
variable. `format()` compiles only after every variable has a value:

```text
error[E0277]: prompt variable `lang` is not set
 --> src/main.rs:4:6
  |
4 |     .format();
  |      ^^^^^^ call `.lang(...)` before this
  |
  = note: the template uses `text`, `lang`; each needs a value
```

A misspelled setter is a missing method, and rustc suggests the right name:

```text
error[E0599]: no method named `txt` found for struct `Prompt<__T0>` in the current scope
help: there is a method `text` with a similar name
```

Template syntax errors point at the literal:

```text
error: invalid prompt template: unclosed `{` at byte 10; write `{{` for a literal brace
```

### Why a builder

The issue that introduced the macro sketched `p.format(text = doc, lang = "pt-BR")`, but Rust methods
have no named arguments. The builder keeps a plain method-call syntax that IDEs complete, accepts the
variables in any order, reports each missing variable by name, and lets one builder render many times.

### The builder

| Method | Returns | Notes |
|---|---|---|
| one setter per variable, e.g. `.text(value)` | the builder | Any `impl Display`: `&str`, `String`, numbers, your own types. Calling it again replaces the value. `{type}` becomes `.r#type(...)`. |
| `.format()` | `String` | Needs every variable. Borrows the builder, so you can call it again. |
| `.to_vars()` | `Vars` | Needs every variable. The values as a map, for a runtime template. |
| `.to_template()` / `.into()` | `PromptTemplate` | Available before any variable is set. |
| `.variables()` | `&'static [&'static str]` | In order of first appearance, like `PromptTemplate::variables`. |

The builder's type is generated at the call site and has no name, like a closure. To reuse a prompt, wrap
it in a function:

```rust
fn summarize(text: &str, lang: &str) -> String {
    prompt!("Summarize {text} in {lang}").text(text).lang(lang).format()
}
```

### Templates in files

```rust
let text = prompt!(file = "prompts/summarize.txt").text(doc).lang("en").format();
```

The path is relative to the calling crate's `Cargo.toml` directory (`CARGO_MANIFEST_DIR`); absolute paths
also work. The file is read at compile time and the expansion includes it with `include_str!`, so editing
the file rebuilds the crate. A missing file is a compile error:

```text
error: cannot read prompt file `prompts/summarize.txt` (paths are relative to the crate's CARGO_MANIFEST_DIR): file not found
```

### Chat prompts

`chat_prompt!` takes `role: template` pairs (`system`, `user`, `assistant`; a template is a literal or
`file = "..."`). The builder has one setter per variable across all messages:

```rust
let request = chat_prompt!(
    system: "You are a {persona}. Answer in {lang}.",
    user: file = "prompts/question.txt",
)
.persona("support agent")
.lang("fr")
.question("Where is my order?")     // a variable from prompts/question.txt
.format();                          // ChatRequest

let reply = model.chat(request).await?;
```

`format()` returns a `ChatRequest`; set temperature or other options on it. `to_messages()` returns only
the `Vec<Message>`, and `to_template()` / `.into()` give a `ChatPromptTemplate`.

### Using it with runnables and the prompt hub

The runtime types are runnables, so convert to compose a chain:

```rust
let chain = PromptTemplate::from(prompt!("Tell me a joke about {topic}"))
    .pipe(model)
    .pipe(StrOutputParser);
let joke = chain.invoke(vars([("topic", "borrow checker")])).await?;
```

With the [prompt hub](prompt-hub.md), keep the template in code and push it, or pull a newer version and
render it with values whose names were checked at compile time:

```rust
let p = prompt!(file = "prompts/summarize.txt");
hub.push(PromptPush::new("summarize", p.to_template().as_str())).await?;

let pulled = hub.pull("summarize@prod").await?;          // PromptTemplate
let text = pulled.render_vars(&p.text(doc).lang("en").to_vars())?;
```

`render_vars` still fails at runtime if the pulled version uses a variable the code does not set.

### Differences from `PromptTemplate`

Every template `prompt!` accepts renders exactly like `PromptTemplate::render`; the test suite checks this
on the same inputs. Two kinds of names that `PromptTemplate` accepts are compile errors with `prompt!`,
because they cannot be method names:

- names that are not Rust identifiers: `{1}`, `{_}`, `{self}`;
- the builder's own method names: `format`, `to_template`, `to_vars`, `to_messages`, `variables`.

Rename the variable or use `PromptTemplate::new`.

Both macros need the `macros` feature: on by default in `sinew`, `features = ["macros"]` on `sinew-core`.
If you re-export `sinew_core` under another name, pass `crate = "path::to::sinew_core"` after the
template.

## Runtime templates

```rust
let t = PromptTemplate::new("Summarize {text} in {lang}")?;   // Error::Template on bad syntax
let text = t.render_vars(&vars([("text", doc), ("lang", "en")]))?;  // Error::MissingVariable

let chat = ChatPromptTemplate::from_messages([
    (Role::System, "You are a {persona}."),
    (Role::User, "{question}"),
])?;
let request = chat.format(&vars([("persona", "pirate"), ("question", "Why Rust?")]))?;
```

Both are runnables: see [Runnables](runnables.md) for composing them with models and parsers.
