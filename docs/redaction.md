# Redacting secrets and personal data in traces

Runs store full model inputs, model outputs, tool arguments and error messages. Those often
contain API keys, bearer tokens, e-mail addresses or card numbers. `sinew_trace::Redactor`
rewrites every run before any sink sees it, so the JSONL file, the in-memory sink, `HttpSink`
(sent to `sinew-scope`), `OtlpSink` and your own sinks all receive the same redacted run.

```rust
use sinew_trace::{HttpSink, HttpSinkConfig, Redactor, Tracer};

let tracer = Tracer::new()
    .with_redactor(
        Redactor::new()                       // secrets, e-mails, cards, sensitive field names
            .phones(true)                     // opt in to phone numbers
            .pattern("customer", r"\bCUST-\d{6}\b")?
            .field("session_id")              // mask this field wherever it appears
            .allow("metadata.model"),         // never touch this path
    )
    .with_sink(HttpSink::new(HttpSinkConfig::new("http://127.0.0.1:7878")));
```

Without `with_redactor`, runs are recorded unchanged.

## What is redacted

The redactor rewrites a run's `inputs`, `outputs`, `error` and `metadata`. It does not change
ids, the run name, timings or token counts, and it does not rewrite object keys.

### Value rules

Every JSON string is scanned; matches are replaced with `[REDACTED:<label>]`.

| Rule | Label | Default | Matches |
|---|---|---|---|
| Secrets | `secret` | on | `sk-…` (OpenAI, Anthropic `sk-ant-…`), `xai-…`, `gsk_…`, `AIza…`, `ghp_` / `gho_` / `ghu_` / `ghs_` / `ghr_…`, `github_pat_…`, AWS `AKIA…` / `ASIA…`, Slack `xox?-…`, JWTs (`eyJ….eyJ….…`), PEM private key blocks, `Bearer <token>` (the word `Bearer` is kept) |
| E-mail | `email` | on | e-mail addresses |
| Cards | `card` | on | 13–19 digits (spaces or dashes allowed) that pass the Luhn check and start with 2–6; JSON integers with 13 or more digits are checked too |
| Phones | `phone` | off | phone numbers with 10–15 digits |
| Custom | your label | – | any regular expression added with `Redactor::pattern(label, regex)` |

The card rule requires a leading digit of 2–6 (the major card networks) on top of the Luhn
check. Without it, about one in ten order numbers or millisecond timestamps (which start with
`1`) would pass Luhn and be masked.

Phones are off by default because the pattern also matches some long numeric identifiers.

Each rule can be switched with `.secrets(bool)`, `.emails(bool)`, `.cards(bool)` and
`.phones(bool)`. `Redactor::empty()` starts with every rule off.

### Field rules

Many credentials have no recognizable shape (a password, a session cookie, a custom header),
so the redactor also masks by key name. Any object field whose key is in the list has its whole
value (string, number, object or array) replaced with `[REDACTED:field]`; `null` stays `null`.
Keys are compared ignoring ASCII case, with `-` and `_` treated as equal, so `X-Api-Key`
matches `x_api_key`.

The default list (`sinew_trace::redact::DEFAULT_FIELDS`) is `password`, `passwd`, `secret`,
`client_secret`, `api_key`, `apikey`, `x_api_key`, `authorization`, `proxy_authorization`,
`cookie`, `set_cookie`, `access_token`, `refresh_token`, `id_token` and `private_key`. Add
more with `.field(name)`; remove them all with `.clear_fields()`.

### Allow-list

`.allow(path)` exempts the value at `path`, and everything below it, from every rule. A path
is dot-separated and starts with a run field (`inputs`, `outputs`, `metadata` or `error`); each
further segment is an object key or an array index, and `*` matches any single key or index:

```rust
Redactor::new()
    .allow("metadata.model")
    .allow("inputs.messages.*.role")
    .allow("error");
```

## Order of rules

For each string, the rules run in this order: secrets, custom patterns, e-mail, cards, phones.
Field rules and the allow-list are checked first, while walking the JSON tree: an allowed path
is skipped entirely, and a masked field is replaced without scanning its content.

## Cost

Regular expressions are compiled once per `Redactor`, on first use. Each string is checked
against a single `RegexSet` that combines every enabled rule; only the rules that matched run a
replacement, and a string with no match is not copied. Redaction runs once per finished run,
on the thread that finishes it, before the run is handed to the sinks.

## Using the redactor directly

`Redactor::redact_str` and `Redactor::redact_value` work on any text or JSON value, for
example to clean a payload before logging it. `Redactor::redact_run` is what the tracer calls.

The building blocks are public in `sinew_trace::redact`: `EMAIL_PATTERN`, `CARD_PATTERN`,
`PHONE_PATTERN`, `luhn_valid`, `is_card_number` and `is_phone_number`. They are defined in
`sinew_core::guardrails::patterns` (with `SECRET_PATTERNS`), and the `PiiRedactor` and
`SecretRedactor` guardrails use the same patterns and validators to redact messages before a
model sees them. See [Guardrails](guardrails.md).

## Limits

- Redaction is pattern based. A secret with no known prefix, inside a field whose name is not
  in the list, is not detected; add a field rule or a custom pattern for it.
- Placeholders are fixed strings, so two runs that saw the same e-mail address cannot be
  correlated from the trace.
- Redaction happens in the process that records the run. Traces ingested by `sinew-scope` over
  OTLP from other applications are stored as received.
