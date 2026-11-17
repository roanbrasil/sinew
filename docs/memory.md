# Long-term memory

Checkpoints keep one thread's state. Long-term memory keeps what an agent
should know about a user in every thread: facts ("works as a nurse"),
preferences ("answers in Portuguese") and episodes ("moved to Montreal in
May"). The `sinew-memory` crate (umbrella feature `memory`, module
`sinew::memory`) builds it on top of the [`Store`](store.md), in the style
of LangMem and Mem0.

| Piece | Does |
|---|---|
| `MemoryManager` | `remember`, `recall`, `context`, `add`, `put`, `get`, `list`, `forget`, `forget_all` for one namespace per user |
| `Memory` | `id`, `content`, `kind` (`Fact`, `Preference`, `Episode`), `created_at`, `updated_at`, `source_thread`, `importance` (serde) |
| `MemoryAgent` | a prebuilt `ReactAgent` that recalls before each turn and saves after it |
| `MemoryHook` (feature `harness`) | the same for the `sinew-harness` agent |
| `ReactAgent::system_context` (`sinew-graph`) | the general mechanism `MemoryAgent` uses: async system text per model call |

## Setup

```rust
use std::sync::Arc;
use sinew_checkpoint::SqliteStore;
use sinew_memory::MemoryManager;
use sinew_rag::embeddings_from_spec;

let store = Arc::new(SqliteStore::connect("sqlite://memory.db").await?);
let memory = MemoryManager::new(store, model.clone())          // model: Arc<dyn ChatModel>
    .with_embeddings_arc(embeddings_from_spec("ollama:bge-m3")?) // optional
    .with_namespace_prefix(["memories"]);                         // the default
```

Memories of user `ana` live under the namespace `["memories", "ana"]`, one
store item per memory, keyed by its id. The item value is the `Memory` as JSON
plus, with embeddings, its vector (`embedding`) and the embeddings model name
(`embedding_model`). Any `Store` works: `InMemoryStore`, `SqliteStore`
(`sinew-checkpoint`, feature `sqlite`), or your own implementation. There is
no Postgres store yet.

## Remembering

```rust
let report = memory.remember("ana", &messages, Some("thread-42")).await?;
println!("{} added, {} updated, {} deleted", report.added.len(), report.updated.len(), report.deleted.len());
```

`remember` works in up to two model calls, both through `StructuredOutput`
(a forced tool call by default, so any tool-calling model works):

1. **Extraction** (`save_memories`). The conversation is sent as
   `user: ...` / `assistant: ...` lines (system and tool messages are
   skipped) with `DEFAULT_EXTRACTION_PROMPT` as the system prompt. The model
   returns a list of `{content, kind, importance}`.
2. **Exact duplicates** of stored memories (same text ignoring case,
   spacing and a final period) are dropped without a model call.
3. **Consolidation** (`update_memories`). For each remaining fact, the
   `with_candidates(n)` (default 5) most similar stored memories are looked
   up. If no stored memory is similar, every fact is added and the second
   call is skipped. Otherwise the model sees those memories, numbered `"0"`,
   `"1"`, ... (short ids instead of UUIDs, so it cannot invent one), and the
   new facts, and returns one operation per fact:

| Operation | Effect |
|---|---|
| `add` | new memory (dropped if its text duplicates a stored or just-added one) |
| `update` | the stored memory gets the merged text, keeps its id and `created_at`; `updated_at` and `source_thread` change; an unknown id is treated as `add` |
| `delete` | the stored memory is removed; unknown ids are ignored |
| `none` | nothing |

Each stored memory is changed at most once per call. Replace either prompt
with `with_extraction_prompt` / `with_update_prompt`, for example to
restrict what is kept ("only remember dietary constraints") or to change the
language of the memories.

`add(user, content, kind)` and `put(user, memory)` save a memory directly,
without a model call and without deduplication.

## Recalling

```rust
for hit in memory.recall("ana", "what should I cook tonight?", 5).await? {
    println!("{:.2} [{}] {}", hit.score, hit.memory.kind, hit.memory.content);
}
// Or as a block for a system prompt (None when nothing is relevant):
let block = memory.context("ana", "what should I cook tonight?", 5).await?;
```

| Configuration | Ranking | `score` |
|---|---|---|
| `with_embeddings(..)` | cosine similarity between the query and each memory | -1 to 1 |
| no embeddings | shared words (via `sinew_rag::english_tokenize`: lower case, stop words removed, light stemming) over the geometric mean of both word counts; memories sharing no word are left out | 0 to 1 |
| empty query | most recently updated first | 0 |

Ties go to higher `importance`, then to the more recent memory. `k = 0`
returns every match. `with_min_score(s)` drops hits below `s`.

Memories are ranked in process after loading the user's namespace, which is
fine for the thousands of memories a user accumulates; the store's own
[index](store.md#semantic-search-with-your-embeddings) is not used (with
embeddings configured, items are written with `index: Some(false)` so the
store does not embed them a second time). A memory saved without a vector,
or with a different embeddings model, is embedded on the next recall and
written back.

## Forgetting

```rust
memory.forget("ana", &memory_id).await?;   // true if it existed
memory.forget_all("ana").await?;           // number deleted
```

A user asking the assistant to forget something goes through `remember`:
the extraction prompt asks for the corrected statement, and consolidation
turns it into `delete` or `update`.

## Prebuilt agent: `MemoryAgent`

```rust
use sinew_graph::prebuilt::{AgentState, ReactAgent};
use sinew_memory::{MemoryAgent, MemoryOptions, RememberScope};

let agent = MemoryAgent::new(
    ReactAgent::new(model.clone(), tools).system_prompt("You are a cooking assistant."),
    memory,
)?
.with_options(MemoryOptions { recall_limit: 5, background: true, scope: RememberScope::LastTurn })
.map_graph(|g| g.with_checkpointer(checkpointer));

let out = agent.invoke("ana", AgentState::new(vec![Message::user("Dinner ideas?")])).await?;
let outcome = agent.run("ana", "thread-42", Some(state)).await?;   // with a checkpointer
```

- **Before the turn.** The memories most relevant to the last user message
  are recalled once per turn and appended to the system prompt of every
  model call of that turn, through `ReactAgent::system_context`. They are
  not added to the state, so they are not checkpointed or returned.
- **After the run.** When the run completes, `remember` gets the last turn
  (from the last user message on; `RememberScope::AllMessages` passes the
  whole state) with the run's thread id as `source_thread`. Interrupted runs
  save nothing until they complete.
- **Background saving.** By default the save runs in a spawned task, so the
  answer is returned without waiting for the extraction calls. `flush()`
  waits for pending saves and returns their results (and errors); call it
  at shutdown, in tests, or periodically. With `background: false` the save
  is awaited and its error returned from `invoke` / `run`.

With a `ReactAgent::context_policy`, the recalled memories are added before
the policy runs: they count against the token budget, and they are never
written to the state, even with `persist_context(true)` (see
[context engineering](context.md)).

The user of a run is looked up by its thread id, so `MemoryAgent` should be
the top-level graph rather than a subgraph. `MemoryAgent::new` sets the
agent's `system_context`; for custom wiring call `ReactAgent::system_context`
yourself with `MemoryManager::context`:

```rust
let memory2 = memory.clone();
let agent = ReactAgent::new(model, tools)
    .system_context(move |state: AgentState, _ctx| {
        let memory = memory2.clone();
        async move {
            let query = sinew_memory::last_user_text(&state.messages).unwrap_or_default();
            Ok(memory.context("ana", &query, 5).await?)
        }
    })
    .compile()?;
```

## Harness hook

`MemoryHook` implements `sinew_harness::Hook` (feature `harness` of
`sinew-memory`; on the umbrella, enabling both `memory` and `harness` turns
it on):

```rust
use sinew_harness::Agent;
use sinew_memory::MemoryHook;

let hook = MemoryHook::new(memory.clone(), "ana");
// or per thread: MemoryHook::with_user_resolver(memory, |ctx| ctx.thread_id.split('/').next().unwrap().into())
let agent = Agent::builder().model_spec("ollama:qwen3").hook(hook.clone()).build()?;
agent.ask("I'm vegetarian, suggest a dinner").await?;
hook.flush().await;
```

`before_model` recalls on the first model call of a turn and appends the
block to the system message (later calls of the turn reuse it);
`after_model` saves the turn when the reply has no tool calls. Subagents
(`depth > 0`) are skipped. The hook is cheap to clone; keep a clone to call
`flush`. `with_options(MemoryOptions { .. })` sets the recall limit,
background saving and scope as for `MemoryAgent`.

## Testing

Script the model with `ScriptedModel`: the extraction reply is a call to the
`save_memories` tool with `{"memories": [...]}`, the consolidation reply a
call to `update_memories` with `{"operations": [...]}`. `FakeEmbeddings`
from `sinew-rag`, or any deterministic `Embeddings`, makes recall order
reproducible. The crate's tests in
[crates/sinew-memory/tests](../crates/sinew-memory/tests) cover extraction,
update and delete, recall ranking, namespace isolation, the SQLite store,
the `MemoryAgent` loop and the harness hook.
