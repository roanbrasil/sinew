// Sinew Scope web UI. Plain JavaScript, no build step, no external requests.
"use strict";

// ---------- small helpers ----------

const store = {
  get(k) { try { return localStorage.getItem(k); } catch (_) { return null; } },
  set(k, v) { try { v == null ? localStorage.removeItem(k) : localStorage.setItem(k, v); } catch (_) {} },
};

/** Creates a DOM element. Children may be nodes, strings, arrays or null. */
function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  append(el, children);
  return el;
}
function append(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false) continue;
    el.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}
function svg(path, size = 12) {
  const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  s.setAttribute("viewBox", "0 0 24 24");
  s.setAttribute("width", size);
  s.setAttribute("height", size);
  const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
  p.setAttribute("d", path);
  p.setAttribute("fill", "currentColor");
  s.appendChild(p);
  return s;
}

const fmt = {
  ms(v) {
    if (v == null) return "–";
    if (v >= 60000) return (v / 60000).toFixed(1) + "m";
    if (v >= 1000) return (v / 1000).toFixed(2) + "s";
    return Math.round(v) + "ms";
  },
  time(ms) {
    if (!ms) return "–";
    const d = new Date(ms);
    return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" });
  },
  num(v) { return v == null ? "–" : Number(v).toLocaleString(); },
  pct(v) { return v == null ? "–" : (v * 100).toFixed(1) + "%"; },
  score(v) { return v == null ? "–" : Number(v).toFixed(2); },
  usd(v) {
    if (v == null) return "–";
    const n = Number(v);
    if (n === 0) return "$0";
    if (Math.abs(n) < 0.01) return "$" + n.toFixed(4);
    if (Math.abs(n) < 10) return "$" + n.toFixed(3);
    return "$" + n.toFixed(2);
  },
  short(id) { return id ? String(id).slice(0, 8) : ""; },
  text(v) { return typeof v === "string" ? v : JSON.stringify(v); },
};

function scoreClass(v) {
  if (v == null) return "";
  return v >= 0.8 ? "score-good" : v >= 0.5 ? "score-mid" : "score-bad";
}
function scorePill(v) {
  return h("span", { class: "pill " + scoreClass(v) }, fmt.score(v));
}
function kindBadge(kind) {
  return h("span", { class: "badge k-" + kind }, kind);
}
function deltaSpan(d) {
  if (d == null) return h("span", { class: "muted" }, "–");
  const cls = d > 1e-9 ? "delta-up" : d < -1e-9 ? "delta-down" : "muted";
  return h("span", { class: cls }, (d > 0 ? "+" : "") + d.toFixed(2));
}

/** Pretty-printed, syntax-highlighted JSON built from DOM nodes (no innerHTML). */
function jsonView(value, cls = "json") {
  const pre = h("pre", { class: cls });
  const walk = (v, indent) => {
    const pad = "  ".repeat(indent);
    if (v === null || typeof v === "boolean") return pre.append(h("span", { class: "j-lit" }, String(v)));
    if (typeof v === "number") return pre.append(h("span", { class: "j-num" }, String(v)));
    if (typeof v === "string") return pre.append(h("span", { class: "j-str" }, JSON.stringify(v)));
    if (Array.isArray(v)) {
      if (!v.length) return pre.append("[]");
      pre.append("[\n");
      v.forEach((x, i) => {
        pre.append(pad + "  ");
        walk(x, indent + 1);
        pre.append(i < v.length - 1 ? ",\n" : "\n");
      });
      return pre.append(pad + "]");
    }
    const keys = Object.keys(v);
    if (!keys.length) return pre.append("{}");
    pre.append("{\n");
    keys.forEach((k, i) => {
      pre.append(pad + "  ", h("span", { class: "j-key" }, JSON.stringify(k)), ": ");
      walk(v[k], indent + 1);
      pre.append(i < keys.length - 1 ? ",\n" : "\n");
    });
    pre.append(pad + "}");
  };
  walk(value === undefined ? null : value, 0);
  return pre;
}

/** Compact text for table cells. */
function cellJson(v) {
  if (v == null) return h("span", { class: "muted" }, "–");
  const s = typeof v === "string" ? v : JSON.stringify(v, null, 1);
  return h("pre", { class: "cell-json" }, s.length > 600 ? s.slice(0, 600) + "…" : s);
}

function copyButton(getText) {
  return h("button", {
    class: "copy",
    onclick: (e) => {
      e.stopPropagation();
      const done = () => { e.target.textContent = "Copied"; setTimeout(() => (e.target.textContent = "Copy"), 1200); };
      if (navigator.clipboard) navigator.clipboard.writeText(getText()).then(done, () => {});
    },
  }, "Copy");
}

// ---------- API ----------

class ApiError extends Error {
  constructor(status, message) { super(message); this.status = status; }
}

async function api(path, opts = {}) {
  const headers = { Accept: "application/json" };
  const token = store.get("sinew.token");
  if (token) headers.Authorization = "Bearer " + token;
  const init = { headers, method: opts.method || "GET" };
  if (opts.body !== undefined) {
    headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(opts.body);
  }
  const res = await fetch(path, init);
  const text = await res.text();
  let body = null;
  try { body = text ? JSON.parse(text) : null; } catch (_) { body = { error: text }; }
  if (!res.ok) throw new ApiError(res.status, (body && body.error) || res.statusText);
  return body;
}

const apiPost = (path, body) => api(path, { method: "POST", body: body === undefined ? {} : body });
const apiPut = (path, body) => api(path, { method: "PUT", body });
const apiDelete = (path) => api(path, { method: "DELETE" });

function qs(params) {
  const u = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== "" && v != null && v !== false) u.set(k, v);
  const s = u.toString();
  return s ? "?" + s : "";
}

// ---------- routing ----------

const app = document.getElementById("app");
const projectSelect = document.getElementById("project");

function parseHash() {
  const raw = location.hash.replace(/^#/, "") || "/traces";
  const [path, query] = raw.split("?");
  const parts = path.split("/").filter(Boolean).map(decodeURIComponent);
  return { parts, params: new URLSearchParams(query || "") };
}

function go(path, params) {
  location.hash = "#" + path + (params ? qs(params) : "");
}

let renderSeq = 0;
async function route() {
  const seq = ++renderSeq;
  const { parts, params } = parseHash();
  const page = parts[0] || "traces";
  document.querySelectorAll("[data-nav]").forEach((a) => {
    const nav = a.getAttribute("data-nav");
    const active = nav === page || (nav === "traces" && page === "trace") ||
      (nav === "datasets" && page === "dataset") ||
      (nav === "prompts" && page === "prompt") ||
      (nav === "queues" && page === "queue") ||
      (nav === "experiments" && (page === "experiment" || page === "compare" || page === "pairwise"));
    a.classList.toggle("active", active);
  });
  const views = {
    traces: () => tracesPage(params),
    trace: () => tracePage(parts[1], params),
    datasets: () => datasetsPage(),
    dataset: () => datasetPage(parts[1]),
    experiments: () => experimentsPage(params),
    experiment: () => experimentPage(parts[1]),
    compare: () => comparePage(params.get("a"), params.get("b")),
    monitor: () => monitorPage(params),
    runs: () => runsPage(params),
    prompts: () => promptsPage(),
    prompt: () => promptPage(parts[1], params),
    queues: () => queuesPage(),
    queue: () => queuePage(parts[1]),
    rules: () => rulesPage(),
    settings: () => settingsPage(),
    pairwise: () => pairwisePage(parts[1]),
  };
  try {
    const node = await (views[page] || views.traces)();
    if (seq !== renderSeq) return; // a newer navigation won
    app.replaceChildren(node);
  } catch (e) {
    if (seq !== renderSeq) return;
    app.replaceChildren(errorView(e));
  }
}

function errorView(e) {
  if (e instanceof ApiError && e.status === 401) {
    return h("div", { class: "card pad" },
      h("h2", {}, "Authentication required"),
      h("p", { class: "muted" }, "This server was started with a token. Paste it to continue; it is stored in this browser only."),
      tokenForm());
  }
  return h("div", { class: "error-box" }, String(e.message || e));
}

function tokenForm() {
  const input = h("input", { type: "password", placeholder: "Bearer token", size: 40, value: store.get("sinew.token") || "" });
  return h("form", {
    class: "toolbar",
    onsubmit: (ev) => { ev.preventDefault(); store.set("sinew.token", input.value.trim() || null); loadProjects(); route(); },
  }, input, h("button", { class: "primary", type: "submit" }, "Save"),
    h("button", { type: "button", onclick: () => { store.set("sinew.token", null); route(); } }, "Clear"));
}

// ---------- traces list ----------

const KINDS = ["", "graph", "node", "llm", "tool", "chain"];

function toLocalInput(ms) {
  if (!ms) return "";
  const d = new Date(Number(ms));
  const p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}`;
}

async function tracesPage(params) {
  const project = projectSelect.value;
  const limit = 50;
  const offset = Number(params.get("offset") || 0);
  const q = {
    project,
    name: params.get("name") || "",
    kind: params.get("kind") || "",
    errors: params.get("errors") === "true",
    from_ms: params.get("from_ms") || "",
    to_ms: params.get("to_ms") || "",
    q: params.get("q") || "",
    tag: params.get("tag") || "",
    limit,
    offset,
  };
  const page = await api("/api/traces" + qs(q));
  const s = page.stats;

  const name = h("input", { type: "search", placeholder: "root run name", value: q.name });
  const text = h("input", { type: "search", placeholder: "words in inputs / outputs", value: q.q, size: 22 });
  const tag = h("input", { type: "search", placeholder: "e.g. prod", value: q.tag, size: 10 });
  const kind = h("select", {}, KINDS.map((k) => h("option", { value: k, selected: k === q.kind }, k || "any")));
  const errors = h("input", { type: "checkbox", checked: q.errors });
  const from = h("input", { type: "datetime-local", value: toLocalInput(q.from_ms) });
  const to = h("input", { type: "datetime-local", value: toLocalInput(q.to_ms) });
  const apply = (ev) => {
    if (ev) ev.preventDefault();
    go("/traces", {
      q: text.value.trim(),
      tag: tag.value.trim(),
      name: name.value.trim(),
      kind: kind.value,
      errors: errors.checked ? "true" : "",
      from_ms: from.value ? new Date(from.value).getTime() : "",
      to_ms: to.value ? new Date(to.value).getTime() : "",
    });
  };
  kind.addEventListener("change", apply);
  errors.addEventListener("change", apply);

  const filters = h("form", { class: "card filters", onsubmit: apply },
    h("label", {}, "Search", text),
    h("label", {}, "Name", name),
    h("label", {}, "Tag", tag),
    h("label", {}, "Contains run kind", kind),
    h("label", {}, "From", from),
    h("label", {}, "To", to),
    h("label", { class: "check" }, errors, "Errors only"),
    h("button", { class: "primary", type: "submit" }, "Apply"),
    h("button", { type: "button", onclick: () => go("/traces") }, "Reset"),
    h("button", { type: "button", onclick: route, title: "Reload" }, "Refresh"));

  const stat = (label, value, bad) => h("div", { class: "card stat" },
    h("div", { class: "label" }, label), h("div", { class: "value" + (bad ? " bad" : "") }, value));
  const stats = h("div", { class: "stats" },
    stat("Traces", fmt.num(s.trace_count)),
    stat("Error rate", fmt.pct(s.error_rate), s.error_count > 0),
    stat("Latency p50", fmt.ms(s.p50_latency_ms)),
    stat("Latency p95", fmt.ms(s.p95_latency_ms)),
    stat("Latency p99", fmt.ms(s.p99_latency_ms)),
    stat("Runs", fmt.num(s.run_count)),
    stat("Tokens in / out", `${fmt.num(s.input_tokens)} / ${fmt.num(s.output_tokens)}`),
    stat("Cost", fmt.usd(s.cost_usd)));

  let table;
  if (!page.traces.length) {
    table = h("div", { class: "card empty" }, "No traces match. Send runs with HttpSink, POST /api/runs, or OTLP to /v1/traces.");
  } else {
    const rows = page.traces.map((t) => h("tr", { class: "click", onclick: () => go("/trace/" + encodeURIComponent(t.trace_id)) },
      h("td", { class: "small muted", style: { whiteSpace: "nowrap" } }, fmt.time(t.start_ms)),
      h("td", {}, h("a", { href: "#/trace/" + encodeURIComponent(t.trace_id) }, t.name),
        h("div", { class: "small muted mono" }, fmt.short(t.trace_id), project ? null : " · " + t.project)),
      h("td", {}, kindBadge(t.kind)),
      h("td", { class: "num" }, fmt.ms(t.latency_ms)),
      h("td", { class: "num" }, t.input_tokens || t.output_tokens ? `${fmt.num(t.input_tokens)} → ${fmt.num(t.output_tokens)}` : "–"),
      h("td", { class: "num cost" }, t.cost_usd ? fmt.usd(t.cost_usd) : "–"),
      h("td", { class: "num" }, t.run_count),
      h("td", {}, t.error_count
        ? h("span", { class: "status-err", title: t.error || "" }, "● ", t.error_count === 1 ? "error" : t.error_count + " errors")
        : h("span", { class: "status-ok" }, "● ok"))));
    const prev = h("button", { disabled: offset === 0, onclick: () => go("/traces", { ...Object.fromEntries(params), offset: Math.max(0, offset - limit) }) }, "← Newer");
    const next = h("button", { disabled: offset + limit >= page.total, onclick: () => go("/traces", { ...Object.fromEntries(params), offset: offset + limit }) }, "Older →");
    table = h("div", { class: "card" },
      h("div", { class: "table-wrap" }, h("table", {},
        h("thead", {}, h("tr", {}, h("th", {}, "Started"), h("th", {}, "Trace"), h("th", {}, "Root kind"),
          h("th", { class: "num" }, "Latency"), h("th", { class: "num" }, "Tokens"), h("th", { class: "num" }, "Cost"), h("th", { class: "num" }, "Runs"), h("th", {}, "Status"))),
        h("tbody", {}, rows))),
      h("div", { class: "pager" },
        h("span", { class: "muted small" }, `${offset + 1}–${Math.min(offset + limit, page.total)} of ${fmt.num(page.total)}`), prev, next));
  }

  return h("div", {},
    h("div", { class: "page-head" }, h("div", {}, h("h1", {}, "Traces"),
      h("div", { class: "muted" }, project ? "Project " + project : "All projects"))),
    filters, stats, table);
}

// ---------- trace detail ----------

async function tracePage(traceId, params) {
  const t = await api("/api/traces/" + encodeURIComponent(traceId));
  const runs = t.runs;
  const byId = new Map(runs.map((r) => [r.id, r]));
  const kids = new Map();
  const roots = [];
  for (const r of runs) {
    if (r.parent_id && byId.has(r.parent_id)) {
      if (!kids.has(r.parent_id)) kids.set(r.parent_id, []);
      kids.get(r.parent_id).push(r);
    } else roots.push(r);
  }
  const t0 = Math.min(...runs.map((r) => r.start_ms));
  const t1 = Math.max(...runs.map((r) => r.end_ms || r.start_ms));
  const span = Math.max(1, t1 - t0);
  const collapsed = new Set();
  let selected = params.get("run") && byId.has(params.get("run")) ? params.get("run") : (roots[0] || runs[0]).id;

  const treeBox = h("div", { class: "tree" });
  const detailBox = h("div", { class: "card detail" });

  const renderTree = () => {
    const rows = [];
    const walk = (r, depth) => {
      const children = kids.get(r.id) || [];
      const isClosed = collapsed.has(r.id);
      const dur = r.end_ms != null ? r.end_ms - r.start_ms : null;
      const left = ((r.start_ms - t0) / span) * 100;
      const width = Math.max(0.4, (((r.end_ms || t1) - r.start_ms) / span) * 100);
      const caret = h("button", {
        class: "caret" + (children.length ? "" : " leaf") + (isClosed ? " closed" : ""),
        "aria-label": isClosed ? "Expand" : "Collapse",
        onclick: (e) => { e.stopPropagation(); isClosed ? collapsed.delete(r.id) : collapsed.add(r.id); renderTree(); },
      }, svg("M7 9l5 6 5-6z", 14));
      const tokens = r.input_tokens != null || r.output_tokens != null ? ` · ${r.input_tokens ?? 0}→${r.output_tokens ?? 0} tok` : "";
      rows.push(h("div", {
        class: "tree-row" + (r.id === selected ? " selected" : ""),
        onclick: () => { selected = r.id; renderTree(); renderDetail(); },
      },
        h("div", { class: "tree-label", style: { paddingLeft: depth * 16 + "px" } },
          caret, kindBadge(r.kind), h("span", { class: "name", title: r.name }, r.name),
          r.error ? h("span", { class: "err-dot", title: r.error }) : null,
          h("span", { class: "meta" }, fmt.ms(dur) + tokens + (r.cost_usd ? " · " + fmt.usd(r.cost_usd) : ""))),
        h("div", { class: "wf k-" + r.kind, title: `+${fmt.ms(r.start_ms - t0)} · ${fmt.ms(dur)}` },
          h("div", { class: "wf-bar" + (r.error ? " err" : ""), style: { left: left + "%", width: Math.min(width, 100 - left) + "%" } }))));
      if (!isClosed) children.forEach((c) => walk(c, depth + 1));
    };
    roots.forEach((r) => walk(r, 0));
    treeBox.replaceChildren(
      h("div", { class: "tree-head" }, h("span", {}, `Runs (${runs.length})`), h("span", {}, `Timeline · ${fmt.ms(span)}`)),
      ...rows);
  };

  const ioBlock = (title, value) => {
    const isMessages = Array.isArray(value) && value.length && value.every((m) => m && typeof m === "object" && "role" in m);
    const isMessageObj = value && typeof value === "object" && !Array.isArray(value) && "role" in value && "content" in value;
    const nice = isMessages ? value : isMessageObj ? [value] : value && Array.isArray(value.messages) ? value.messages : null;
    let mode = nice ? "pretty" : "json";
    const body = h("div");
    const tabs = h("div", { class: "tabs" });
    const draw = () => {
      tabs.replaceChildren(...(nice ? ["pretty", "json"] : ["json"]).map((m) =>
        h("button", { class: m === mode ? "on" : "", onclick: () => { mode = m; draw(); } }, m === "pretty" ? "Messages" : "JSON")));
      body.replaceChildren(mode === "pretty" ? messagesView(nice) : jsonView(value));
    };
    draw();
    return h("div", { class: "detail-section" },
      h("div", { class: "section-title" }, h("span", {}, title), copyButton(() => JSON.stringify(value, null, 2))),
      nice ? tabs : null, body);
  };

  const renderDetail = () => {
    const r = byId.get(selected);
    if (!r) return detailBox.replaceChildren(h("div", { class: "empty" }, "Select a run"));
    const dur = r.end_ms != null ? r.end_ms - r.start_ms : null;
    const kv = [
      ["Run id", h("span", { class: "mono" }, r.id)],
      ["Parent", r.parent_id ? h("span", { class: "mono" }, r.parent_id) : "–"],
      ["Started", new Date(r.start_ms).toLocaleString() + ` (+${fmt.ms(r.start_ms - t0)})`],
      ["Duration", dur == null ? "still open" : fmt.ms(dur)],
    ];
    if (r.input_tokens != null || r.output_tokens != null)
      kv.push(["Tokens", `${fmt.num(r.input_tokens)} in · ${fmt.num(r.output_tokens)} out · ${fmt.num((r.input_tokens || 0) + (r.output_tokens || 0))} total`]);
    const model = r.model || (r.metadata && r.metadata.model);
    if (model) kv.push(["Model", model]);
    if (r.cost_usd != null) kv.push(["Cost", fmt.usd(r.cost_usd)]);
    if (r.ttft_ms != null) kv.push(["Time to first token", fmt.ms(r.ttft_ms)]);
    const tags = r.metadata && Array.isArray(r.metadata.tags) ? r.metadata.tags : [];
    if (tags.length) kv.push(["Tags", tags.map((t) => h("span", { class: "tag" }, String(t)))]);
    detailBox.replaceChildren(
      h("div", { class: "detail-head" },
        h("div", { class: "toolbar" }, kindBadge(r.kind), h("h2", { style: { margin: 0 } }, r.name)),
        r.error ? h("div", { class: "error-box", style: { marginTop: "10px", marginBottom: 0 } }, r.error) : null),
      h("div", { class: "detail-section" }, h("dl", { class: "kv" }, kv.map(([k, v]) => [h("dt", {}, k), h("dd", {}, v)]))),
      h("div", { class: "detail-section" }, h("div", { class: "section-title" }, h("span", {}, "Feedback")), feedbackPanel(r.id)),
      h("div", { class: "detail-section" }, runActions(r)),
      ioBlock("Inputs", r.inputs),
      r.outputs !== undefined && r.outputs !== null ? ioBlock("Outputs", r.outputs) : null,
      r.metadata && Object.keys(r.metadata).length ? ioBlock("Metadata", r.metadata) : null);
  };

  renderTree();
  renderDetail();
  const s = t.summary;
  return h("div", {},
    h("div", { class: "page-head" },
      h("div", {},
        h("div", { class: "crumbs" }, h("a", { href: "#/traces" }, "Traces"), " / ", s.project),
        h("h1", {}, s.name),
        h("div", { class: "muted toolbar" },
          h("span", { class: "mono" }, s.trace_id),
          h("span", {}, [fmt.time(s.start_ms), fmt.ms(s.latency_ms),
            `${fmt.num(s.input_tokens)} → ${fmt.num(s.output_tokens)} tokens`, `${s.run_count} runs`,
            s.cost_usd ? fmt.usd(s.cost_usd) : null].filter(Boolean).join(" · ")),
          s.error_count ? h("span", { class: "status-err" }, `${s.error_count} error(s)`) : null)),
      h("button", { onclick: route }, "Refresh")),
    h("div", { class: "trace-layout" }, h("div", { class: "card" }, treeBox), detailBox));
}

function messagesView(list) {
  return h("div", { class: "messages" }, list.map((m) => {
    let content = m.content;
    if (content == null && Array.isArray(m.parts)) content = m.parts.map((p) => p.content ?? p.text ?? JSON.stringify(p)).join("\n");
    if (typeof content !== "string") content = content == null ? "" : JSON.stringify(content, null, 2);
    const calls = m.tool_calls && m.tool_calls.length ? m.tool_calls : null;
    return h("div", { class: "msg" },
      h("div", { class: "msg-role" }, m.role + (m.tool_call_id ? " · " + m.tool_call_id : "")),
      content ? h("div", { class: "msg-body" }, content) : null,
      calls ? h("div", { class: "msg-tools" }, calls.map((c) =>
        h("div", { class: "small" }, "→ ", h("b", {}, c.name || (c.function && c.function.name) || "tool"), " ",
          h("code", {}, fmt.text(c.arguments ?? (c.function && c.function.arguments) ?? {}))))) : null);
  }));
}

// ---------- datasets ----------

async function datasetsPage() {
  const list = await api("/api/datasets");
  const body = list.length
    ? h("div", { class: "card table-wrap" }, h("table", {},
      h("thead", {}, h("tr", {}, h("th", {}, "Dataset"), h("th", {}, "Description"), h("th", { class: "num" }, "Examples"),
        h("th", { class: "num" }, "Experiments"), h("th", {}, "Updated"))),
      h("tbody", {}, list.map((d) => h("tr", { class: "click", onclick: () => go("/dataset/" + encodeURIComponent(d.name)) },
        h("td", {}, h("a", { href: "#/dataset/" + encodeURIComponent(d.name) }, d.name)),
        h("td", { class: "muted" }, d.description || "–"),
        h("td", { class: "num" }, d.example_count),
        h("td", { class: "num" }, d.experiment_count),
        h("td", { class: "small muted" }, fmt.time(d.updated_ms)))))))
    : h("div", { class: "card empty" }, "No datasets yet. Upload one with sinew-eval's ScopeClient or POST /api/datasets.");
  return h("div", {}, h("div", { class: "page-head" }, h("h1", {}, "Datasets")), body);
}

function scoreKeys(experiments) {
  const keys = new Set();
  experiments.forEach((e) => Object.keys((e.aggregates && e.aggregates.scores) || {}).forEach((k) => keys.add(k)));
  return [...keys].sort();
}

function experimentsTable(experiments, { selectable, showDataset } = {}) {
  const keys = scoreKeys(experiments);
  const picked = [];
  const compareBtn = h("button", { class: "primary", disabled: true, onclick: () => go("/compare", { a: picked[0], b: picked[1] }) }, "Compare selected");
  const sync = () => { compareBtn.disabled = picked.length !== 2; };
  const table = h("table", {},
    h("thead", {}, h("tr", {}, selectable ? h("th", {}, "") : null, h("th", {}, "Experiment"), showDataset ? h("th", {}, "Dataset") : null,
      h("th", {}, "Created"), h("th", { class: "num" }, "Examples"), h("th", { class: "num" }, "Errors"),
      h("th", { class: "num" }, "p50"), h("th", { class: "num" }, "p95"), keys.map((k) => h("th", { class: "num" }, k)))),
    h("tbody", {}, experiments.map((e) => {
      const a = e.aggregates || {};
      const box = selectable ? h("input", {
        type: "checkbox", "aria-label": "Select " + e.name,
        onclick: (ev) => ev.stopPropagation(),
        onchange: (ev) => {
          if (ev.target.checked) { picked.push(e.id); if (picked.length > 2) { const drop = picked.shift(); table.querySelector(`input[data-id="${drop}"]`).checked = false; } }
          else picked.splice(picked.indexOf(e.id), 1);
          sync();
        },
        "data-id": e.id,
      }) : null;
      return h("tr", { class: "click", onclick: () => go("/experiment/" + encodeURIComponent(e.id)) },
        selectable ? h("td", {}, box) : null,
        h("td", {}, h("a", { href: "#/experiment/" + encodeURIComponent(e.id) }, e.name)),
        showDataset ? h("td", {}, h("a", { href: "#/dataset/" + encodeURIComponent(e.dataset), onclick: (ev) => ev.stopPropagation() }, e.dataset)) : null,
        h("td", { class: "small muted" }, fmt.time(e.created_ms)),
        h("td", { class: "num" }, a.count ?? "–"),
        h("td", { class: "num" + (a.errors ? " status-err" : "") }, a.errors ?? "–"),
        h("td", { class: "num" }, fmt.ms(a.p50_latency_ms)),
        h("td", { class: "num" }, fmt.ms(a.p95_latency_ms)),
        keys.map((k) => h("td", { class: "num" }, scorePill(a.scores && a.scores[k] ? a.scores[k].mean : null))));
    })));
  return h("div", {},
    selectable ? h("div", { class: "toolbar", style: { marginBottom: "8px" } }, compareBtn, h("span", { class: "muted small" }, "Pick two experiments to compare side by side.")) : null,
    h("div", { class: "card table-wrap" }, table));
}

async function datasetPage(name) {
  const { dataset, experiments } = await api("/api/datasets/" + encodeURIComponent(name));
  const examples = dataset.examples.length
    ? h("div", { class: "card table-wrap" }, h("table", {},
      h("thead", {}, h("tr", {}, h("th", { class: "num" }, "#"), h("th", {}, "Inputs"), h("th", {}, "Reference"))),
      h("tbody", {}, dataset.examples.map((ex, i) => h("tr", {},
        h("td", { class: "num muted" }, i + 1), h("td", {}, cellJson(ex.inputs)), h("td", {}, cellJson(ex.reference)))))))
    : h("div", { class: "card empty" }, "This dataset has no examples.");
  return h("div", {},
    h("div", { class: "page-head" }, h("div", {},
      h("div", { class: "crumbs" }, h("a", { href: "#/datasets" }, "Datasets")),
      h("h1", {}, dataset.name),
      h("div", { class: "muted" }, dataset.description || `${dataset.examples.length} examples`))),
    h("div", { class: "section" }, h("h2", {}, `Experiments (${experiments.length})`),
      experiments.length ? experimentsTable(experiments, { selectable: true }) : h("div", { class: "card empty" }, "No experiments on this dataset yet.")),
    h("div", { class: "section" }, h("h2", {}, `Examples (${dataset.examples.length})`), examples));
}

// ---------- experiments ----------

async function experimentsPage(params) {
  const dataset = params.get("dataset") || "";
  const [list, pairwise] = await Promise.all([api("/api/experiments" + qs({ dataset })), api("/api/pairwise" + qs({ dataset }))]);
  return h("div", {},
    h("div", { class: "page-head" }, h("div", {}, h("h1", {}, "Experiments"),
      h("div", { class: "muted" }, dataset ? "Dataset " + dataset : "All datasets"))),
    list.length ? experimentsTable(list, { selectable: true, showDataset: true })
      : h("div", { class: "card empty" }, "No experiments yet. Run sinew_eval::evaluate and upload the report."),
    pairwise.length ? h("div", { class: "section", style: { marginTop: "18px" } }, h("h2", {}, `Pairwise comparisons (${pairwise.length})`), pairwiseTable(pairwise)) : null);
}

function scoreCell(s) {
  if (!s) return h("td", { class: "num muted" }, "–");
  return h("td", { class: "num" }, scorePill(s.score), s.reason ? h("div", { class: "reason", title: s.reason }, s.reason.length > 120 ? s.reason.slice(0, 120) + "…" : s.reason) : null);
}

async function experimentPage(id) {
  const r = await api("/api/experiments/" + encodeURIComponent(id));
  const a = r.aggregates;
  const keys = Object.keys(a.scores || {}).sort();
  const stat = (label, value, bad) => h("div", { class: "card stat" },
    h("div", { class: "label" }, label), h("div", { class: "value" + (bad ? " bad" : "") }, value));
  let onlyFail = false;
  const tbody = h("tbody");
  const drawRows = () => {
    const rows = r.results.filter((x) => !onlyFail || x.error || x.scores.some((s) => s.score < 0.5));
    tbody.replaceChildren(...rows.map((x) => h("tr", {},
      h("td", { class: "num muted" }, x.index + 1, x.repetition ? h("div", { class: "small" }, "rep " + (x.repetition + 1)) : null),
      h("td", {}, cellJson(x.inputs)),
      h("td", {}, cellJson(x.reference)),
      h("td", {}, x.error ? h("div", { class: "status-err small" }, x.error) : cellJson(x.output)),
      keys.map((k) => scoreCell(x.scores.find((s) => s.key === k))),
      h("td", { class: "num" }, fmt.ms(x.latency_ms)))));
  };
  drawRows();
  const failToggle = h("input", { type: "checkbox", onchange: (e) => { onlyFail = e.target.checked; drawRows(); } });
  return h("div", {},
    h("div", { class: "page-head" }, h("div", {},
      h("div", { class: "crumbs" }, h("a", { href: "#/experiments" }, "Experiments"), " / ",
        h("a", { href: "#/dataset/" + encodeURIComponent(r.dataset) }, r.dataset)),
      h("h1", {}, r.name),
      h("div", { class: "muted" }, fmt.time(r.created_ms), " · ", h("span", { class: "mono" }, fmt.short(r.id))))),
    h("div", { class: "stats" },
      stat("Examples", a.count), stat("Errors", a.errors, a.errors > 0),
      stat("Mean latency", fmt.ms(a.mean_latency_ms)), stat("p50", fmt.ms(a.p50_latency_ms)), stat("p95", fmt.ms(a.p95_latency_ms)),
      keys.map((k) => h("div", { class: "card stat" }, h("div", { class: "label" }, k),
        h("div", { class: "value" }, scorePill(a.scores[k].mean)))),
      (r.summary_scores || []).map((sc) => h("div", { class: "card stat", title: sc.reason || "" }, h("div", { class: "label" }, sc.key + " (summary)"),
        h("div", { class: "value" }, scorePill(sc.score))))),
    r.repetitions > 1 ? h("p", { class: "muted small" }, `Each example ran ${r.repetitions} times; aggregates cover every repetition.`) : null,
    r.metadata && Object.keys(r.metadata).length ? h("div", { class: "section card pad-s" }, h("div", { class: "section-title" }, "Metadata"), jsonView(r.metadata)) : null,
    h("div", { class: "toolbar", style: { marginBottom: "8px" } }, h("label", { class: "toolbar" }, failToggle, "Only failures (error or score < 0.5)")),
    h("div", { class: "card table-wrap" }, h("table", {},
      h("thead", {}, h("tr", {}, h("th", { class: "num" }, "#"), h("th", {}, "Inputs"), h("th", {}, "Reference"), h("th", {}, "Output"),
        keys.map((k) => h("th", { class: "num" }, k)), h("th", { class: "num" }, "Latency"))),
      tbody)));
}

async function comparePage(aId, bId) {
  if (!aId || !bId) return h("div", { class: "card empty" }, "Pick two experiments on a dataset page to compare them.");
  const c = await api("/api/compare" + qs({ a: aId, b: bId }));
  let onlyDiff = false;
  const tbody = h("tbody");
  const drawRows = () => {
    const rows = c.rows.filter((row) => !onlyDiff || Object.values(row.deltas).some((d) => Math.abs(d) > 1e-9) || !row.a || !row.b ||
      JSON.stringify(row.a.output) !== JSON.stringify(row.b.output));
    tbody.replaceChildren(...rows.map((row) => h("tr", {},
      h("td", {}, cellJson(row.inputs)),
      h("td", {}, cellJson(row.reference)),
      h("td", {}, row.a ? (row.a.error ? h("div", { class: "status-err small" }, row.a.error) : cellJson(row.a.output)) : h("span", { class: "muted" }, "missing")),
      h("td", {}, row.b ? (row.b.error ? h("div", { class: "status-err small" }, row.b.error) : cellJson(row.b.output)) : h("span", { class: "muted" }, "missing")),
      c.keys.map((k) => {
        const sa = row.a && row.a.scores.find((s) => s.key === k);
        const sb = row.b && row.b.scores.find((s) => s.key === k);
        return h("td", { class: "num" }, scorePill(sa ? sa.score : null), " ", scorePill(sb ? sb.score : null), h("div", {}, deltaSpan(row.deltas[k])));
      }),
      h("td", { class: "num small" }, fmt.ms(row.a && row.a.latency_ms), " / ", fmt.ms(row.b && row.b.latency_ms)))));
  };
  drawRows();
  const head = (x, label) => h("div", { class: "card pad-s" },
    h("div", { class: "section-title" }, label),
    h("h2", {}, h("a", { href: "#/experiment/" + encodeURIComponent(x.id) }, x.name)),
    h("div", { class: "muted small" }, fmt.time(x.created_ms), ` · ${x.aggregates.count} examples · ${x.aggregates.errors} errors · p50 ${fmt.ms(x.aggregates.p50_latency_ms)}`));
  return h("div", {},
    h("div", { class: "page-head" }, h("div", {},
      h("div", { class: "crumbs" }, h("a", { href: "#/experiments" }, "Experiments"), " / ",
        h("a", { href: "#/dataset/" + encodeURIComponent(c.a.dataset) }, c.a.dataset)),
      h("h1", {}, "Compare experiments"))),
    h("div", { class: "grid-2 section" }, head(c.a, "A (baseline)"), head(c.b, "B (candidate)")),
    h("div", { class: "section card table-wrap" }, h("table", {},
      h("thead", {}, h("tr", {}, h("th", {}, "Score"), h("th", { class: "num" }, "A"), h("th", { class: "num" }, "B"), h("th", { class: "num" }, "B − A"))),
      h("tbody", {}, c.summary.map((s) => h("tr", {}, h("td", {}, s.key), h("td", { class: "num" }, scorePill(s.a)),
        h("td", { class: "num" }, scorePill(s.b)), h("td", { class: "num" }, deltaSpan(s.delta))))))),
    h("div", { class: "toolbar", style: { marginBottom: "8px" } },
      h("label", { class: "toolbar" }, h("input", { type: "checkbox", onchange: (e) => { onlyDiff = e.target.checked; drawRows(); } }), "Only rows that differ")),
    h("div", { class: "card table-wrap" }, h("table", {},
      h("thead", {}, h("tr", {}, h("th", {}, "Inputs"), h("th", {}, "Reference"), h("th", {}, "Output A"), h("th", {}, "Output B"),
        c.keys.map((k) => h("th", { class: "num" }, k + " (A B Δ)")), h("th", { class: "num" }, "Latency A / B"))),
      tbody)));
}

// ---------- chrome: projects, theme, token ----------

async function loadProjects() {
  try {
    const list = await api("/api/projects");
    const current = store.get("sinew.project") || "";
    projectSelect.replaceChildren(h("option", { value: "" }, "All projects"),
      ...list.map((p) => h("option", { value: p.project }, `${p.project} (${p.trace_count})`)));
    projectSelect.value = list.some((p) => p.project === current) ? current : "";
  } catch (_) { /* auth errors surface on the page itself */ }
}

projectSelect.addEventListener("change", () => {
  store.set("sinew.project", projectSelect.value || null);
  const { parts } = parseHash();
  if ((parts[0] || "traces") === "traces") go("/traces"); else route();
});

function applyTheme(t) {
  if (t) document.documentElement.setAttribute("data-theme", t);
  else document.documentElement.removeAttribute("data-theme");
}
applyTheme(store.get("sinew.theme"));
document.getElementById("theme-btn").addEventListener("click", () => {
  const dark = document.documentElement.getAttribute("data-theme")
    ? document.documentElement.getAttribute("data-theme") === "dark"
    : matchMedia("(prefers-color-scheme: dark)").matches;
  const next = dark ? "light" : "dark";
  store.set("sinew.theme", next);
  applyTheme(next);
});
document.getElementById("token-btn").addEventListener("click", () => {
  app.replaceChildren(h("div", { class: "card pad" }, h("h2", {}, "API token"),
    h("p", { class: "muted" }, "Only needed when the server runs with --token. Stored in this browser only."), tokenForm()));
});

window.addEventListener("hashchange", route);
loadProjects().then(route);
