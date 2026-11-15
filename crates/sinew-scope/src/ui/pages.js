// Sinew Scope web UI, part 2: monitoring, run search, prompt hub, feedback,
// annotation queues, rules & alerts, settings and pairwise comparisons.
// Loaded before app.js; uses its helpers (h, api, fmt, ...) at call time.
"use strict";

// ---------- shared bits ----------

function notice(kind, text) {
  return h("div", { class: kind === "ok" ? "ok-box" : "error-box" }, text);
}

/** Runs `fn`, showing errors (or a success message) in `box`. */
async function attempt(box, fn, okText) {
  box.replaceChildren();
  try {
    const r = await fn();
    if (okText) box.replaceChildren(notice("ok", typeof okText === "function" ? okText(r) : okText));
    return r;
  } catch (e) {
    box.replaceChildren(notice("err", String(e.message || e)));
    return undefined;
  }
}

function pageHead(title, sub, ...right) {
  return h("div", { class: "page-head" },
    h("div", {}, h("h1", {}, title), sub ? h("div", { class: "muted" }, sub) : null),
    right.length ? h("div", { class: "actions" }, right) : null);
}

function statCard(label, value, bad, title) {
  return h("div", { class: "card stat", title: title || null },
    h("div", { class: "label" }, label), h("div", { class: "value" + (bad ? " bad" : "") }, value));
}

function parseMeta(text) {
  const out = {};
  for (const part of (text || "").split(",")) {
    const [k, ...rest] = part.split("=");
    if (k && k.trim() && rest.length) out["meta." + k.trim()] = rest.join("=").trim();
  }
  return out;
}

function metaText(params) {
  const parts = [];
  for (const [k, v] of params.entries()) if (k.startsWith("meta.")) parts.push(k.slice(5) + "=" + v);
  return parts.join(", ");
}

function tagsOf(v) {
  return Array.isArray(v) ? v.map((t) => h("span", { class: "tag" }, String(t))) : null;
}

// ---------- SVG charts ----------

const SVGNS = "http://www.w3.org/2000/svg";
function svgEl(tag, attrs, ...children) {
  const el = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) if (v != null) el.setAttribute(k, v);
  for (const c of children.flat()) if (c != null) el.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
  return el;
}

/**
 * Line chart. `series` = [{name, color, points: [{x, y}]}] with x in ms.
 * Null y values break the line.
 */
function lineChart(title, series, { yfmt = (v) => fmt.num(v), height = 190, x0: dx0, x1: dx1 } = {}) {
  const W = 480, H = height, L = 56, R = 12, T = 10, B = 24;
  const xs = series.flatMap((se) => se.points.map((p) => p.x));
  const ys = series.flatMap((se) => se.points.map((p) => p.y)).filter((v) => v != null && isFinite(v));
  const card = h("div", { class: "card chart" }, h("h3", {}, title));
  if (!xs.length || !ys.length) {
    card.append(h("div", { class: "empty", style: { padding: "30px" } }, "No data in this range"));
    return card;
  }
  const x0 = dx0 != null ? Math.min(dx0, ...xs) : Math.min(...xs);
  const x1 = Math.max(dx1 != null ? dx1 : Math.max(...xs), x0 + 1);
  const yMax = Math.max(...ys, 0) || 1;
  const sx = (x) => L + ((x - x0) / (x1 - x0)) * (W - L - R);
  const sy = (y) => T + (1 - y / yMax) * (H - T - B);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title });
  for (let i = 0; i <= 3; i++) {
    const v = (yMax * i) / 3, y = sy(v);
    svg.append(svgEl("line", { class: i ? "grid" : "axis", x1: L, x2: W - R, y1: y, y2: y }),
      svgEl("text", { x: L - 6, y: y + 3, "text-anchor": "end" }, yfmt(v)));
  }
  const span = x1 - x0;
  const tfmt = (x) => {
    const d = new Date(x);
    return span > 2 * 86400000 ? d.toLocaleDateString(undefined, { month: "short", day: "numeric" })
      : d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
  };
  for (let i = 0; i <= 4; i++) {
    const x = x0 + (span * i) / 4;
    svg.append(svgEl("text", { x: sx(x), y: H - 6, "text-anchor": i === 0 ? "start" : i === 4 ? "end" : "middle" }, tfmt(x)));
  }
  for (const se of series) {
    let d = "", pen = false;
    for (const p of se.points) {
      if (p.y == null || !isFinite(p.y)) { pen = false; continue; }
      d += (pen ? "L" : "M") + sx(p.x).toFixed(1) + " " + sy(p.y).toFixed(1);
      pen = true;
    }
    if (d) svg.append(svgEl("path", { class: "line", d, stroke: se.color }));
    for (const p of se.points) {
      if (p.y == null || !isFinite(p.y)) continue;
      svg.append(svgEl("circle", { class: "dot", cx: sx(p.x), cy: sy(p.y), r: se.points.length > 40 ? 2 : 3, fill: se.color },
        svgEl("title", {}, `${se.name}: ${yfmt(p.y)} · ${new Date(p.x).toLocaleString()}`)));
    }
  }
  card.append(svg);
  if (series.length > 1) card.append(h("div", { class: "legend" }, series.map((se) => h("span", {}, h("i", { style: { background: se.color } }), se.name))));
  return card;
}

const CHART = ["var(--chart-1)", "var(--chart-2)", "var(--chart-3)", "var(--chart-4)"];

// ---------- monitor ----------

const RANGES = [["1h", 3600e3], ["6h", 6 * 3600e3], ["24h", 86400e3], ["7d", 7 * 86400e3], ["30d", 30 * 86400e3], ["all", 0]];

function groupTable(groups, label, { ttft } = {}) {
  if (!groups.length) return h("div", { class: "card empty" }, "No data");
  const maxCount = Math.max(...groups.map((g) => g.count), 1);
  return h("div", { class: "card table-wrap" }, h("table", {},
    h("thead", {}, h("tr", {}, h("th", {}, label), h("th", { class: "num" }, "Count"), h("th", { class: "num" }, "Error rate"),
      h("th", { class: "num" }, "p50"), h("th", { class: "num" }, "p95"), h("th", { class: "num" }, "p99"),
      ttft ? h("th", { class: "num" }, "TTFT p50 / p95") : null,
      h("th", { class: "num" }, "Tokens in / out"), h("th", { class: "num" }, "Cost"))),
    h("tbody", {}, groups.map((g) => h("tr", {},
      h("td", { class: "bar-cell" }, h("div", { class: "bar", style: { width: (g.count / maxCount) * 100 + "%" } }), h("span", {}, g.key)),
      h("td", { class: "num" }, fmt.num(g.count)),
      h("td", { class: "num" + (g.errors ? " status-err" : "") }, fmt.pct(g.error_rate)),
      h("td", { class: "num" }, fmt.ms(g.p50_latency_ms)),
      h("td", { class: "num" }, fmt.ms(g.p95_latency_ms)),
      h("td", { class: "num" }, fmt.ms(g.p99_latency_ms)),
      ttft ? h("td", { class: "num" }, `${fmt.ms(g.ttft_p50_ms)} / ${fmt.ms(g.ttft_p95_ms)}`) : null,
      h("td", { class: "num" }, `${fmt.num(g.input_tokens)} / ${fmt.num(g.output_tokens)}`),
      h("td", { class: "num cost" }, fmt.usd(g.cost_usd)))))));
}

async function monitorPage(params) {
  const project = projectSelect.value;
  const range = params.get("range") || "24h";
  const rangeMs = (RANGES.find((r) => r[0] === range) || RANGES[2])[1];
  const now = Date.now();
  const base = {
    project,
    from_ms: rangeMs ? now - rangeMs : "",
    to_ms: rangeMs ? now : "",
    tag: params.get("tag") || "",
    name: params.get("name") || "",
    ...parseMeta(params.get("meta") || ""),
  };
  const [traces, llm, fb] = await Promise.all([
    api("/api/metrics" + qs({ ...base, group_by: "name" })),
    api("/api/metrics" + qs({ ...base, name: "", level: "llm", group_by: "model" })),
    api("/api/feedback/stats" + qs({ project, from_ms: base.from_ms, to_ms: base.to_ms })),
  ]);
  const t = traces.totals, l = llm.totals;

  const rangeSel = h("select", {}, RANGES.map(([k]) => h("option", { value: k, selected: k === range }, k === "all" ? "all time" : "last " + k)));
  const tag = h("input", { type: "search", placeholder: "e.g. prod", value: base.tag, size: 10 });
  const name = h("input", { type: "search", placeholder: "trace name", value: base.name, size: 14 });
  const meta = h("input", { type: "search", placeholder: "key=value, key2=value2", value: params.get("meta") || "", size: 22 });
  const apply = (ev) => { if (ev) ev.preventDefault(); go("/monitor", { range: rangeSel.value, tag: tag.value.trim(), name: name.value.trim(), meta: meta.value.trim() }); };
  rangeSel.addEventListener("change", apply);
  const filters = h("form", { class: "card filters", onsubmit: apply },
    h("label", {}, "Time range", rangeSel), h("label", {}, "Trace name", name), h("label", {}, "Tag (root run)", tag),
    h("label", {}, "Metadata (root run)", meta),
    h("button", { class: "primary", type: "submit" }, "Apply"),
    h("button", { type: "button", onclick: () => go("/monitor") }, "Reset"),
    h("button", { type: "button", onclick: route }, "Refresh"));

  const pts = (m, f) => m.buckets.map((b) => ({ x: b.start_ms + m.bucket_ms / 2, y: f(b) }));
  const dom = { x0: traces.from_ms, x1: traces.to_ms };
  const charts = h("div", { class: "charts" },
    lineChart("Trace latency", [
      { name: "p50", color: CHART[0], points: pts(traces, (b) => b.p50_latency_ms) },
      { name: "p95", color: CHART[1], points: pts(traces, (b) => b.p95_latency_ms) },
      { name: "p99", color: CHART[3], points: pts(traces, (b) => b.p99_latency_ms) }], { yfmt: fmt.ms, ...dom }),
    lineChart("Error rate", [{ name: "error rate", color: "var(--err)", points: pts(traces, (b) => b.error_rate) }], { yfmt: (v) => (v * 100).toFixed(0) + "%", ...dom }),
    lineChart("Traces", [{ name: "traces", color: CHART[0], points: pts(traces, (b) => b.count) }], dom),
    lineChart("Tokens", [
      { name: "input", color: CHART[1], points: pts(traces, (b) => b.input_tokens) },
      { name: "output", color: CHART[2], points: pts(traces, (b) => b.output_tokens) }], dom),
    lineChart("Cost (USD)", [{ name: "cost", color: CHART[0], points: pts(traces, (b) => b.cost_usd) }], { yfmt: fmt.usd, ...dom }),
    lineChart("Time to first token (LLM)", [
      { name: "p50", color: CHART[0], points: pts(llm, (b) => b.ttft_p50_ms) },
      { name: "p95", color: CHART[1], points: pts(llm, (b) => b.ttft_p95_ms) }], { yfmt: fmt.ms, ...dom }));

  const fbTable = fb.length ? h("div", { class: "card table-wrap" }, h("table", {},
    h("thead", {}, h("tr", {}, h("th", {}, "Feedback key"), h("th", { class: "num" }, "Entries"), h("th", { class: "num" }, "Mean score"))),
    h("tbody", {}, fb.map((f) => h("tr", {}, h("td", {}, f.key), h("td", { class: "num" }, fmt.num(f.count)), h("td", { class: "num" }, scorePill(f.mean)))))))
    : h("div", { class: "card empty" }, "No feedback in this range.");

  return h("div", {},
    pageHead("Monitor", (project ? "Project " + project : "All projects") + (traces.truncated ? " · showing the newest 200,000 points" : "")),
    filters,
    h("div", { class: "stats" },
      statCard("Traces", fmt.num(t.count)),
      statCard("Error rate", fmt.pct(t.error_rate), t.errors > 0),
      statCard("Latency p50", fmt.ms(t.p50_latency_ms)),
      statCard("p95", fmt.ms(t.p95_latency_ms)),
      statCard("p99", fmt.ms(t.p99_latency_ms)),
      statCard("TTFT p50 / p95", `${fmt.ms(l.ttft_p50_ms)} / ${fmt.ms(l.ttft_p95_ms)}`),
      statCard("Tokens", fmt.num(t.input_tokens + t.output_tokens), false, `${fmt.num(t.input_tokens)} in · ${fmt.num(t.output_tokens)} out`),
      statCard("Cost", fmt.usd(t.cost_usd))),
    charts,
    h("div", { class: "section" }, h("h2", {}, "By trace name"), groupTable(traces.groups, "Trace")),
    h("div", { class: "section" }, h("h2", {}, "LLM calls by model"), groupTable(llm.groups, "Model", { ttft: true })),
    h("div", { class: "section" }, h("h2", {}, "Feedback"), fbTable));
}

// ---------- runs (search) ----------

async function runsPage(params) {
  const project = projectSelect.value;
  const q = {
    project,
    q: params.get("q") || "",
    kind: params.get("kind") || "",
    model: params.get("model") || "",
    tag: params.get("tag") || "",
    errors: params.get("errors") === "true",
    root: params.get("root") === "true",
    feedback_key: params.get("feedback_key") || "",
    limit: 50,
  };
  const meta = params.get("meta") || "";
  const textIn = h("input", { type: "search", placeholder: "full-text search", value: q.q, size: 24 });
  const kind = h("select", {}, KINDS.map((k) => h("option", { value: k, selected: k === q.kind }, k || "any")));
  const model = h("input", { type: "search", placeholder: "model prefix", value: q.model, size: 12 });
  const tag = h("input", { type: "search", placeholder: "tag", value: q.tag, size: 8 });
  const metaIn = h("input", { type: "search", placeholder: "key=value", value: meta, size: 14 });
  const fbKey = h("input", { type: "search", placeholder: "feedback key", value: q.feedback_key, size: 10 });
  const errors = h("input", { type: "checkbox", checked: q.errors });
  const root = h("input", { type: "checkbox", checked: q.root });
  const apply = (ev) => {
    if (ev) ev.preventDefault();
    go("/runs", { q: textIn.value.trim(), kind: kind.value, model: model.value.trim(), tag: tag.value.trim(), meta: metaIn.value.trim(),
      feedback_key: fbKey.value.trim(), errors: errors.checked ? "true" : "", root: root.checked ? "true" : "" });
  };
  [kind, errors, root].forEach((el) => el.addEventListener("change", apply));
  const filters = h("form", { class: "card filters", onsubmit: apply },
    h("label", {}, "Search", textIn), h("label", {}, "Kind", kind), h("label", {}, "Model", model), h("label", {}, "Tag", tag),
    h("label", {}, "Metadata", metaIn), h("label", {}, "Has feedback", fbKey),
    h("label", { class: "check" }, errors, "Errors"), h("label", { class: "check" }, root, "Root runs only"),
    h("button", { class: "primary", type: "submit" }, "Search"), h("button", { type: "button", onclick: () => go("/runs") }, "Reset"));

  const selected = new Set();
  const tbody = h("tbody");
  const moreBox = h("div", { class: "pager" });
  const msg = h("div");
  const queues = await api("/api/queues").catch(() => []);
  const queueSel = h("select", {}, h("option", { value: "" }, "Add selected to queue…"), queues.map((x) => h("option", { value: x.id }, x.name)));
  queueSel.addEventListener("change", async () => {
    const id = queueSel.value;
    queueSel.value = "";
    if (!id || !selected.size) return;
    await attempt(msg, () => apiPost(`/api/queues/${encodeURIComponent(id)}/items`, { run_ids: [...selected] }), (r) => `Added ${r.added} run(s) to the queue.`);
  });

  const load = async (cursor) => {
    const page = await api("/api/runs" + qs({ ...q, ...parseMeta(meta), cursor }));
    for (const r of page.runs) {
      const box = h("input", { type: "checkbox", "aria-label": "Select run", onclick: (e) => e.stopPropagation(),
        onchange: (e) => { e.target.checked ? selected.add(r.id) : selected.delete(r.id); } });
      tbody.append(h("tr", { class: "click", onclick: () => go("/trace/" + encodeURIComponent(r.trace_id), { run: r.id }) },
        h("td", {}, box),
        h("td", { class: "small muted", style: { whiteSpace: "nowrap" } }, fmt.time(r.start_ms)),
        h("td", {}, h("div", {}, kindBadge(r.kind), " ", h("b", {}, r.name), " ", tagsOf(r.tags)),
          h("div", { class: "small muted mono" }, fmt.short(r.id), project ? null : " · " + r.project, r.model && r.model !== r.name ? " · " + r.model : null)),
        h("td", {}, h("pre", { class: "cell-json" }, r.input_preview)),
        h("td", {}, r.error ? h("div", { class: "status-err small" }, r.error) : h("pre", { class: "cell-json" }, r.output_preview || "–")),
        h("td", { class: "num" }, fmt.ms(r.latency_ms)),
        h("td", { class: "num" }, r.input_tokens != null ? `${fmt.num(r.input_tokens)} → ${fmt.num(r.output_tokens)}` : "–"),
        h("td", { class: "num cost" }, r.cost_usd != null ? fmt.usd(r.cost_usd) : "–")));
    }
    moreBox.replaceChildren(
      h("span", { class: "muted small" }, `${tbody.children.length} runs`),
      page.next_cursor ? h("button", { onclick: () => load(page.next_cursor) }, "Load older →") : h("span", { class: "muted small" }, "end"));
  };
  await load("");
  return h("div", {},
    pageHead("Runs", "Search every run: full text over inputs, outputs, names and errors; filter by kind, model, tags and metadata.", queueSel),
    filters, msg,
    h("div", { class: "card" }, h("div", { class: "table-wrap" }, h("table", {},
      h("thead", {}, h("tr", {}, h("th", {}, ""), h("th", {}, "Started"), h("th", {}, "Run"), h("th", {}, "Input"), h("th", {}, "Output"),
        h("th", { class: "num" }, "Latency"), h("th", { class: "num" }, "Tokens"), h("th", { class: "num" }, "Cost"))),
      tbody)), moreBox));
}

// ---------- feedback + run actions (trace detail) ----------

function feedbackPanel(runId) {
  const list = h("div", { class: "fb-list" }, h("span", { class: "muted small" }, "Loading…"));
  const msg = h("div");
  const refresh = async () => {
    try {
      const items = await api(`/api/runs/${encodeURIComponent(runId)}/feedback`);
      list.replaceChildren(...(items.length ? items.map((f) => h("div", { class: "fb" },
        h("span", { class: "key" }, f.key),
        f.score != null ? scorePill(f.score) : null,
        f.value != null ? h("code", {}, fmt.text(f.value)) : null,
        f.comment ? h("span", {}, f.comment) : null,
        h("span", { class: "src" }, f.source + " · " + fmt.time(f.created_ms)),
        h("button", { class: "x", title: "Delete", "aria-label": "Delete feedback", onclick: async () => { await attempt(msg, () => apiDelete("/api/feedback/" + encodeURIComponent(f.id))); refresh(); } }, "×")))
        : [h("span", { class: "muted small" }, "No feedback yet.")]));
    } catch (e) { list.replaceChildren(h("span", { class: "muted small" }, String(e.message || e))); }
  };
  const key = h("input", { value: store.get("sinew.fbkey") || "user_score", size: 12, "aria-label": "Feedback key" });
  let score = null;
  const btns = h("div", { class: "score-btns" });
  const drawBtns = () => btns.replaceChildren(...[["0", 0], ["0.5", 0.5], ["1", 1]].map(([l, v]) =>
    h("button", { type: "button", class: "small" + (score === v ? " on" : ""), onclick: () => { score = score === v ? null : v; drawBtns(); } }, l)));
  drawBtns();
  const comment = h("input", { placeholder: "comment (optional)", size: 22 });
  const form = h("form", { class: "actions", onsubmit: async (ev) => {
    ev.preventDefault();
    store.set("sinew.fbkey", key.value.trim());
    const r = await attempt(msg, () => apiPost("/api/feedback", { run_id: runId, key: key.value.trim(), score, comment: comment.value.trim() || null, source: "ui" }));
    if (r) { comment.value = ""; score = null; drawBtns(); refresh(); }
  } }, key, btns, comment, h("button", { class: "primary small", type: "submit" }, "Add"));
  refresh();
  return h("div", {}, list, form, msg);
}

function runActions(r) {
  const msg = h("div");
  const ds = h("input", { placeholder: "dataset name", size: 14, list: "ds-list", "aria-label": "Dataset" });
  const dl = h("datalist", { id: "ds-list" });
  api("/api/datasets").then((l) => dl.replaceChildren(...l.map((d) => h("option", { value: d.name })))).catch(() => {});
  const qsel = h("select", { "aria-label": "Annotation queue" }, h("option", { value: "" }, "queue…"));
  api("/api/queues").then((l) => qsel.append(...l.map((x) => h("option", { value: x.id }, x.name)))).catch(() => {});
  return h("div", {},
    h("div", { class: "section-title" }, h("span", {}, "Actions")),
    h("div", { class: "actions" },
      h("form", { class: "actions", onsubmit: async (ev) => {
        ev.preventDefault();
        if (!ds.value.trim()) return;
        await attempt(msg, () => apiPost(`/api/datasets/${encodeURIComponent(ds.value.trim())}/examples`, { run_id: r.id }), (x) => `Added to dataset ${x.dataset} (${x.examples} examples).`);
      } }, ds, dl, h("button", { type: "submit", class: "small" }, "Add to dataset")),
      h("form", { class: "actions", onsubmit: async (ev) => {
        ev.preventDefault();
        if (!qsel.value) return;
        await attempt(msg, () => apiPost(`/api/queues/${encodeURIComponent(qsel.value)}/items`, { run_ids: [r.id] }), "Added to the annotation queue.");
      } }, qsel, h("button", { type: "submit", class: "small" }, "Add to queue"))),
    msg);
}

// ---------- prompt hub ----------

function promptForm(initial, onDone) {
  const msg = h("div");
  const name = h("input", { value: initial.name || "", placeholder: "support-answer", required: true, disabled: !!initial.name });
  const desc = h("input", { value: initial.description || "", placeholder: "what it is for" });
  const template = h("textarea", { rows: 8, placeholder: "Answer the {question} in a {tone} tone." });
  template.value = initial.template || "";
  const config = h("textarea", { rows: 3, placeholder: '{"model": "openai:gpt-4.1-mini", "temperature": 0.2}' });
  config.value = initial.model_config && Object.keys(initial.model_config).length ? JSON.stringify(initial.model_config, null, 2) : "";
  const tags = h("input", { placeholder: "staging, prod" });
  const message = h("input", { placeholder: "what changed" });
  const form = h("form", { class: "card pad-m section", onsubmit: async (ev) => {
    ev.preventDefault();
    let model_config = {};
    try { model_config = config.value.trim() ? JSON.parse(config.value) : {}; } catch (e) { msg.replaceChildren(notice("err", "Model config is not valid JSON: " + e.message)); return; }
    const body = { name: name.value.trim(), template: template.value, description: desc.value.trim() || null, model_config,
      message: message.value.trim() || null, tags: tags.value.split(",").map((t) => t.trim()).filter(Boolean) };
    if (initial.messages) body.messages = null;
    const r = await attempt(msg, () => apiPost("/api/prompts", body), (x) => x.created ? `Committed version ${x.version} (${x.commit}).` : `No change: same as version ${x.version}.`);
    if (r && onDone) onDone(r);
  } },
    h("h2", {}, initial.name ? "New version of " + initial.name : "New prompt"),
    h("div", { class: "form-grid" },
      h("label", {}, "Name", name), h("label", {}, "Description", desc),
      h("label", { class: "wide" }, "Template ({variable}; {{ and }} for literal braces)", template),
      h("label", { class: "wide" }, "Model config (JSON)", config),
      h("label", {}, "Tags", tags), h("label", {}, "Commit message", message)),
    h("div", { class: "actions", style: { marginTop: "10px" } }, h("button", { class: "primary", type: "submit" }, "Commit")),
    msg);
  return form;
}

async function promptsPage() {
  const list = await api("/api/prompts");
  const holder = h("div");
  const newBtn = h("button", { class: "primary", onclick: () => holder.replaceChildren(promptForm({}, (r) => go("/prompt/" + encodeURIComponent(r.name)))) }, "New prompt");
  const body = list.length ? h("div", { class: "card table-wrap" }, h("table", {},
    h("thead", {}, h("tr", {}, h("th", {}, "Prompt"), h("th", {}, "Description"), h("th", {}, "Tags"), h("th", { class: "num" }, "Versions"), h("th", {}, "Latest"), h("th", {}, "Updated"))),
    h("tbody", {}, list.map((p) => h("tr", { class: "click", onclick: () => go("/prompt/" + encodeURIComponent(p.name)) },
      h("td", {}, h("a", { href: "#/prompt/" + encodeURIComponent(p.name) }, p.name)),
      h("td", { class: "muted" }, p.description || "–"),
      h("td", {}, Object.entries(p.tags).map(([t, c]) => h("span", { class: "tag", title: c }, t))),
      h("td", { class: "num" }, p.version_count),
      h("td", { class: "mono small" }, p.latest_commit ? `v${p.latest_version} · ${p.latest_commit}` : "–"),
      h("td", { class: "small muted" }, fmt.time(p.updated_ms)))))))
    : h("div", { class: "card empty" }, "No prompts yet. Create one here or push with sinew_eval::PromptHub.");
  return h("div", {}, pageHead("Prompts", "Versioned prompt templates with tags (prod, staging…). Pull them with PromptHub::pull(\"name@prod\").", newBtn), holder, body);
}

function templateView(v) {
  if (v.messages && v.messages.length) return messagesView(v.messages);
  return h("pre", { class: "json" }, v.template);
}

async function promptPage(name, params) {
  const p = await api("/api/prompts/" + encodeURIComponent(name));
  const versions = p.versions;
  if (!versions.length) return h("div", { class: "card empty" }, "This prompt has no versions.");
  let selected = versions.find((v) => v.commit === params.get("v")) || versions[0];
  const compareWith = params.get("compare");
  const listBox = h("div", { class: "card list" });
  const detail = h("div");
  const editHolder = h("div");

  const drawList = () => listBox.replaceChildren(...versions.map((v) => h("div", {
    class: "list-item" + (v.commit === selected.commit ? " selected" : ""),
    onclick: () => go("/prompt/" + encodeURIComponent(name), { v: v.commit }),
  },
    h("div", { class: "top" }, h("b", {}, "v" + v.version), h("span", { class: "mono small muted" }, v.commit)),
    h("div", {}, tagsOf(v.tags)),
    v.message ? h("div", { class: "small" }, v.message) : null,
    h("div", { class: "small muted" }, fmt.time(v.created_ms)))));

  const drawDetail = async () => {
    const msg = h("div");
    const tagIn = h("input", { placeholder: "tag, e.g. prod", size: 12 });
    const cmpSel = h("select", {}, h("option", { value: "" }, "Compare with…"),
      versions.filter((v) => v.commit !== selected.commit).map((v) => h("option", { value: v.commit, selected: v.commit === compareWith }, `v${v.version} · ${v.commit}`)));
    cmpSel.addEventListener("change", () => go("/prompt/" + encodeURIComponent(name), { v: selected.commit, compare: cmpSel.value }));
    let diffBox = null;
    if (compareWith && compareWith !== selected.commit) {
      const d = await api(`/api/prompts/${encodeURIComponent(name)}/diff` + qs({ from: compareWith, to: selected.commit }));
      diffBox = h("div", { class: "card pad-m section" },
        h("div", { class: "section-title" }, h("span", {}, `Diff v${d.from.version} → v${d.to.version}`)),
        h("div", { class: "diff" }, d.template.map((l) => h("div", { class: l.op }, l.text || " "))),
        d.variables_added.length || d.variables_removed.length ? h("p", { class: "small" },
          d.variables_added.length ? ["Variables added: ", h("code", {}, d.variables_added.join(", ")), " "] : null,
          d.variables_removed.length ? ["Variables removed: ", h("code", {}, d.variables_removed.join(", "))] : null) : null,
        d.model_config.length ? h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "Model config"), h("th", {}, "From"), h("th", {}, "To"))),
          h("tbody", {}, d.model_config.map((c) => h("tr", {}, h("td", {}, c.key), h("td", {}, h("code", {}, fmt.text(c.from ?? null))), h("td", {}, h("code", {}, fmt.text(c.to ?? null))))))) : null);
    }
    detail.replaceChildren(
      h("div", { class: "card pad-m section" },
        h("div", { class: "top actions", style: { justifyContent: "space-between" } },
          h("h2", { style: { margin: 0 } }, `v${selected.version} `, h("span", { class: "mono muted small" }, selected.commit), " ", tagsOf(selected.tags)),
          h("div", { class: "actions" }, cmpSel,
            h("button", { onclick: () => editHolder.replaceChildren(promptForm({ ...selected, description: p.description }, (r) => go("/prompt/" + encodeURIComponent(name), { v: r.commit }))) }, "Edit as new version"))),
        selected.message ? h("p", { class: "muted" }, selected.message) : null,
        h("div", { class: "section-title" }, h("span", {}, "Template"), copyButton(() => selected.template)),
        templateView(selected),
        h("p", { class: "small" }, "Variables: ", selected.input_variables.length ? selected.input_variables.map((v) => h("code", { style: { marginRight: "6px" } }, v)) : "none"),
        selected.model_config && Object.keys(selected.model_config).length ? [h("div", { class: "section-title" }, "Model config"), jsonView(selected.model_config)] : null,
        h("div", { class: "section-title", style: { marginTop: "12px" } }, "Use it"),
        h("pre", { class: "json" }, `let t = PromptHub::new(url).pull("${name}@${selected.commit}").await?;`),
        h("form", { class: "actions", style: { marginTop: "12px" }, onsubmit: async (ev) => {
          ev.preventDefault();
          const t = tagIn.value.trim();
          if (!t) return;
          const r = await attempt(msg, () => apiPut(`/api/prompts/${encodeURIComponent(name)}/tags/${encodeURIComponent(t)}`, { commit: selected.commit }));
          if (r) route();
        } }, tagIn, h("button", { type: "submit" }, "Point tag here"),
          selected.tags.map((t) => h("button", { type: "button", class: "small danger", onclick: async () => { await attempt(msg, () => apiDelete(`/api/prompts/${encodeURIComponent(name)}/tags/${encodeURIComponent(t)}`)); route(); } }, "untag " + t))),
        msg),
      diffBox, editHolder);
  };
  drawList();
  await drawDetail();
  const delMsg = h("div");
  return h("div", {},
    h("div", { class: "page-head" }, h("div", {},
      h("div", { class: "crumbs" }, h("a", { href: "#/prompts" }, "Prompts")),
      h("h1", {}, name), h("div", { class: "muted" }, p.description || `${versions.length} versions`)),
      h("button", { class: "danger", onclick: async () => {
        if (!confirm(`Delete prompt ${name} and all its versions?`)) return;
        const r = await attempt(delMsg, () => apiDelete("/api/prompts/" + encodeURIComponent(name)));
        if (r) go("/prompts");
      } }, "Delete")),
    delMsg,
    h("div", { class: "split" }, listBox, detail));
}

// ---------- annotation queues ----------

async function queuesPage() {
  const list = await api("/api/queues");
  const msg = h("div");
  const name = h("input", { placeholder: "weekly review", required: true });
  const desc = h("input", { placeholder: "what reviewers should check" });
  const keys = h("input", { placeholder: "correctness, helpfulness", value: "correctness" });
  const form = h("form", { class: "card pad-m section", onsubmit: async (ev) => {
    ev.preventDefault();
    const r = await attempt(msg, () => apiPost("/api/queues", { name: name.value.trim(), description: desc.value.trim(), keys: keys.value.split(",").map((k) => k.trim()).filter(Boolean) }));
    if (r) go("/queue/" + encodeURIComponent(r.id));
  } }, h("h2", {}, "New queue"),
    h("div", { class: "form-grid" }, h("label", {}, "Name", name), h("label", {}, "Description", desc), h("label", {}, "Feedback keys (comma separated)", keys),
      h("button", { class: "primary", type: "submit" }, "Create")), msg);
  const body = list.length ? h("div", { class: "card table-wrap" }, h("table", {},
    h("thead", {}, h("tr", {}, h("th", {}, "Queue"), h("th", {}, "Keys"), h("th", { class: "num" }, "Pending"), h("th", { class: "num" }, "Done"), h("th", {}, "Created"))),
    h("tbody", {}, list.map((q) => h("tr", { class: "click", onclick: () => go("/queue/" + encodeURIComponent(q.id)) },
      h("td", {}, h("a", { href: "#/queue/" + encodeURIComponent(q.id) }, q.name), q.description ? h("div", { class: "small muted" }, q.description) : null),
      h("td", {}, q.keys.map((k) => h("span", { class: "tag" }, k))),
      h("td", { class: "num" }, q.pending), h("td", { class: "num" }, q.done),
      h("td", { class: "small muted" }, fmt.time(q.created_ms)))))))
    : h("div", { class: "card empty" }, "No annotation queues yet. Create one, then add runs from the Runs page or a trace.");
  return h("div", {}, pageHead("Annotation queues", "Human review: score queued runs and leave comments; results are stored as feedback."), body, form);
}

async function queuePage(id) {
  const [detail, next] = await Promise.all([api("/api/queues/" + encodeURIComponent(id)), api(`/api/queues/${encodeURIComponent(id)}/next`)]);
  const q = detail.queue;
  const msg = h("div");
  let review;
  if (!next.item) {
    review = h("div", { class: "card empty" }, detail.done ? "All done. Nothing left to review." : "The queue is empty. Add runs from the Runs page or a trace.");
  } else {
    const it = next.item, r = it.run || {};
    const scores = {};
    const comments = {};
    const keys = q.keys.length ? q.keys : ["score"];
    const keyRows = keys.map((k) => {
      const btns = h("div", { class: "score-btns" });
      const num = h("input", { type: "number", min: 0, max: 1, step: 0.1, size: 4, style: { width: "70px" }, "aria-label": k + " score",
        oninput: () => { scores[k] = num.value === "" ? null : Number(num.value); draw(); } });
      const draw = () => btns.replaceChildren(...[0, 0.5, 1].map((v) => h("button", { type: "button", class: scores[k] === v ? "on" : "",
        onclick: () => { scores[k] = scores[k] === v ? null : v; num.value = scores[k] ?? ""; draw(); } }, String(v))));
      draw();
      const c = h("textarea", { rows: 2, placeholder: "comment", oninput: () => { comments[k] = c.value; } });
      return h("div", { class: "field", style: { marginBottom: "12px" } }, h("b", { style: { color: "var(--text)" } }, k), h("div", { class: "actions" }, btns, num), c);
    });
    const submit = async (skip) => {
      const feedback = skip ? [] : keys.map((k) => ({ key: k, score: scores[k] ?? null, comment: (comments[k] || "").trim() || null }));
      const done = await attempt(msg, () => skip
        ? apiDelete(`/api/queues/${encodeURIComponent(id)}/items/${encodeURIComponent(it.run_id)}`)
        : apiPost(`/api/queues/${encodeURIComponent(id)}/items/${encodeURIComponent(it.run_id)}/done`, { feedback }));
      if (done) route();
    };
    review = h("div", { class: "review" },
      h("div", { class: "card" },
        h("div", { class: "detail-head" }, h("div", { class: "toolbar" }, r.kind ? kindBadge(r.kind) : null, h("h2", { style: { margin: 0 } }, r.name || it.run_id),
          r.trace_id ? h("a", { href: "#/trace/" + encodeURIComponent(r.trace_id) + "?run=" + encodeURIComponent(it.run_id), class: "small" }, "open trace →") : null),
          r.error ? h("div", { class: "error-box", style: { marginTop: "10px", marginBottom: 0 } }, r.error) : null),
        h("div", { class: "detail-section" }, h("div", { class: "section-title" }, "Inputs"), r.inputs && Array.isArray(r.inputs.messages) ? messagesView(r.inputs.messages) : jsonView(r.inputs)),
        h("div", { class: "detail-section" }, h("div", { class: "section-title" }, "Outputs"), r.outputs && typeof r.outputs === "object" && "role" in r.outputs ? messagesView([r.outputs]) : jsonView(r.outputs)),
        r.metadata && r.metadata.reference !== undefined ? h("div", { class: "detail-section" }, h("div", { class: "section-title" }, "Reference"), jsonView(r.metadata.reference)) : null,
        next.feedback && next.feedback.length ? h("div", { class: "detail-section" }, h("div", { class: "section-title" }, "Existing feedback"),
          next.feedback.map((f) => h("div", { class: "fb" }, h("span", { class: "key" }, f.key), f.score != null ? scorePill(f.score) : null, f.comment || "", h("span", { class: "src" }, f.source)))) : null),
      h("div", { class: "card pad-m detail" },
        h("h2", {}, "Your review"),
        keyRows,
        h("div", { class: "actions" }, h("button", { class: "primary", onclick: () => submit(false) }, "Submit & next"), h("button", { onclick: () => submit(true) }, "Remove from queue")),
        msg));
  }
  const items = detail.items;
  const itemTable = items.length ? h("div", { class: "card table-wrap" }, h("table", {},
    h("thead", {}, h("tr", {}, h("th", {}, "Run"), h("th", {}, "Status"), h("th", {}, "Added"), h("th", {}, "Done"))),
    h("tbody", {}, items.map((x) => h("tr", {},
      h("td", {}, x.run ? h("a", { href: "#/trace/" + encodeURIComponent(x.run.trace_id) + "?run=" + encodeURIComponent(x.run_id) }, x.run.name) : x.run_id, h("span", { class: "small muted mono" }, " " + fmt.short(x.run_id))),
      h("td", {}, x.status === "done" ? h("span", { class: "status-ok" }, "done") : h("span", { class: "muted" }, "pending")),
      h("td", { class: "small muted" }, fmt.time(x.added_ms)), h("td", { class: "small muted" }, fmt.time(x.done_ms)))))))
    : null;
  return h("div", {},
    h("div", { class: "page-head" }, h("div", {},
      h("div", { class: "crumbs" }, h("a", { href: "#/queues" }, "Annotation queues")),
      h("h1", {}, q.name), h("div", { class: "muted" }, `${detail.pending} pending · ${detail.done} done`, q.description ? " · " + q.description : "")),
      h("button", { class: "danger", onclick: async () => { if (confirm("Delete this queue? Feedback already given is kept.")) { await apiDelete("/api/queues/" + encodeURIComponent(id)); go("/queues"); } } }, "Delete queue")),
    review,
    itemTable ? h("div", { class: "section", style: { marginTop: "18px" } }, h("h2", {}, "Items"), itemTable) : null);
}

// ---------- rules & alerts ----------

const EVALUATOR_EXAMPLE = `[
  {"type": "not_empty"},
  {"type": "contains", "needle": "sorry", "key": "apologizes"},
  {"type": "llm_judge", "model": "openai:gpt-4.1-mini", "rubric": "Score 1 if the answer is helpful and polite.", "key": "helpful"}
]`;

async function rulesPage() {
  const [rules, alerts] = await Promise.all([api("/api/rules"), api("/api/alerts")]);
  const msg = h("div");

  const ruleRows = rules.map((r) => h("tr", {},
    h("td", {}, h("b", {}, r.name), r.last_error ? h("div", { class: "status-err small" }, r.last_error) : null),
    h("td", {}, h("label", { class: "switch" }, h("input", { type: "checkbox", checked: r.enabled, onchange: async (e) => {
      await attempt(msg, () => apiPut("/api/rules/" + encodeURIComponent(r.id), { ...r, enabled: e.target.checked }));
    } }), r.enabled ? "on" : "off")),
    h("td", { class: "small" }, Object.entries(r.filter).filter(([, v]) => v && (typeof v !== "object" || Object.keys(v).length)).map(([k, v]) => h("div", {}, `${k}: ${fmt.text(v)}`))),
    h("td", { class: "num" }, fmt.pct(r.sample_rate)),
    h("td", { class: "small" }, r.evaluators.map((e) => h("div", {}, h("code", {}, e.type), e.key ? " → " + e.key : ""))),
    h("td", { class: "num" }, fmt.num(r.runs_evaluated), h("div", { class: "small muted" }, fmt.time(r.last_run_ms))),
    h("td", {}, h("div", { class: "actions" },
      h("button", { class: "small", onclick: () => attempt(msg, () => apiPost(`/api/rules/${encodeURIComponent(r.id)}/run?limit=100`), (x) => `Scored the latest runs: ${x.feedback} feedback entries.`) }, "Run on latest 100"),
      h("button", { class: "small danger", onclick: async () => { if (confirm("Delete rule " + r.name + "?")) { await attempt(msg, () => apiDelete("/api/rules/" + encodeURIComponent(r.id))); route(); } } }, "Delete")))));

  const rn = h("input", { placeholder: "answer quality", required: true });
  const rproj = h("input", { placeholder: "any", value: projectSelect.value || "" });
  const rkind = h("select", {}, KINDS.map((k) => h("option", { value: k, selected: k === "" }, k || "any")));
  const rtag = h("input", { placeholder: "any" });
  const rroot = h("input", { type: "checkbox", checked: true });
  const rate = h("input", { type: "number", min: 0, max: 1, step: 0.05, value: 1 });
  const evals = h("textarea", { rows: 6 });
  evals.value = EVALUATOR_EXAMPLE;
  const ruleForm = h("form", { class: "card pad-m section", onsubmit: async (ev) => {
    ev.preventDefault();
    let evaluators;
    try { evaluators = JSON.parse(evals.value); } catch (e) { msg.replaceChildren(notice("err", "Evaluators must be a JSON array: " + e.message)); return; }
    const r = await attempt(msg, () => apiPost("/api/rules", { name: rn.value.trim(), enabled: true, sample_rate: Number(rate.value),
      filter: { project: rproj.value.trim() || null, kind: rkind.value || null, tag: rtag.value.trim() || null, root_only: rroot.checked }, evaluators }));
    if (r) route();
  } }, h("h2", {}, "New online evaluation rule"),
    h("p", { class: "hint" }, "Matching runs are sampled as they arrive, scored by the evaluators in the background, and the scores are stored as feedback (source rule:<name>). LLM judges use a provider:model spec and that provider's API key on the server. The reference for exact_match / contains / numeric comes from the run's metadata.reference."),
    h("div", { class: "form-grid" }, h("label", {}, "Name", rn), h("label", {}, "Project", rproj), h("label", {}, "Run kind", rkind), h("label", {}, "Tag", rtag),
      h("label", {}, "Sample rate (0–1)", rate), h("label", { class: "check filters" }, h("span", { class: "switch" }, rroot, "Root runs only")),
      h("label", { class: "wide" }, "Evaluators (JSON): exact_match, contains, regex, json_field, numeric, llm_judge, not_empty, is_json", evals)),
    h("div", { class: "actions", style: { marginTop: "10px" } }, h("button", { class: "primary", type: "submit" }, "Create rule")));

  const alertRows = alerts.map((a) => h("tr", {},
    h("td", {}, h("b", {}, a.name), a.last_error ? h("div", { class: "status-err small" }, a.last_error) : null),
    h("td", {}, h("code", {}, a.metric), " > ", String(a.threshold)),
    h("td", { class: "num" }, Math.round(a.window_ms / 60000) + " min"),
    h("td", { class: "small" }, a.filter.project || "all projects"),
    h("td", {}, a.last_status === "firing" ? h("span", { class: "status-err" }, "● firing") : a.last_status === "ok" ? h("span", { class: "status-ok" }, "● ok") : h("span", { class: "muted" }, a.last_status || "not checked"),
      h("div", { class: "small muted" }, a.last_value != null ? "value " + Number(a.last_value).toFixed(3) : "", a.last_fired_ms ? " · fired " + fmt.time(a.last_fired_ms) : "")),
    h("td", {}, h("div", { class: "actions" },
      h("button", { class: "small", onclick: async () => { await attempt(msg, () => apiPost(`/api/alerts/${encodeURIComponent(a.id)}/check`), (x) => `Status ${x.status}${x.value != null ? ", value " + Number(x.value).toFixed(3) : ""}${x.fired ? ", webhook sent" : ""}.`); } }, "Check now"),
      h("button", { class: "small danger", onclick: async () => { if (confirm("Delete alert " + a.name + "?")) { await attempt(msg, () => apiDelete("/api/alerts/" + encodeURIComponent(a.id))); route(); } } }, "Delete")))));

  const an = h("input", { placeholder: "error spike", required: true });
  const metric = h("select", {}, ["error_rate", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms", "avg_latency_ms", "ttft_p95_ms", "cost_usd", "count"].map((m) => h("option", { value: m }, m)));
  const thr = h("input", { type: "number", step: "any", value: 0.1 });
  const win = h("input", { type: "number", min: 1, value: 15 });
  const cool = h("input", { type: "number", min: 0, value: 60 });
  const aproj = h("input", { placeholder: "any", value: projectSelect.value || "" });
  const hook = h("input", { type: "url", placeholder: "https://hooks.slack.com/…", required: true });
  const alertForm = h("form", { class: "card pad-m section", onsubmit: async (ev) => {
    ev.preventDefault();
    const r = await attempt(msg, () => apiPost("/api/alerts", { name: an.value.trim(), enabled: true, metric: metric.value, threshold: Number(thr.value),
      window_ms: Number(win.value) * 60000, cooldown_ms: Number(cool.value) * 60000, webhook_url: hook.value.trim(), filter: { project: aproj.value.trim() || null } }));
    if (r) route();
  } }, h("h2", {}, "New alert"),
    h("p", { class: "hint" }, "Checked every 30 seconds over a sliding window (traces). When the metric goes above the threshold, a JSON payload with a text field is POSTed to the webhook (Slack-compatible), at most once per cooldown."),
    h("div", { class: "form-grid" }, h("label", {}, "Name", an), h("label", {}, "Metric", metric), h("label", {}, "Threshold", thr),
      h("label", {}, "Window (minutes)", win), h("label", {}, "Cooldown (minutes)", cool), h("label", {}, "Project", aproj), h("label", { class: "wide" }, "Webhook URL", hook)),
    h("div", { class: "actions", style: { marginTop: "10px" } }, h("button", { class: "primary", type: "submit" }, "Create alert")));

  return h("div", {},
    pageHead("Rules & alerts", "Online evaluators score incoming runs automatically; alerts watch error rate, latency and cost."),
    msg,
    h("div", { class: "section" }, h("h2", {}, `Online evaluation rules (${rules.length})`),
      rules.length ? h("div", { class: "card table-wrap" }, h("table", {},
        h("thead", {}, h("tr", {}, h("th", {}, "Rule"), h("th", {}, "Enabled"), h("th", {}, "Filter"), h("th", { class: "num" }, "Sample"), h("th", {}, "Evaluators"), h("th", { class: "num" }, "Evaluated"), h("th", {}, ""))),
        h("tbody", {}, ruleRows))) : h("div", { class: "card empty" }, "No rules yet.")),
    ruleForm,
    h("div", { class: "section" }, h("h2", {}, `Alerts (${alerts.length})`),
      alerts.length ? h("div", { class: "card table-wrap" }, h("table", {},
        h("thead", {}, h("tr", {}, h("th", {}, "Alert"), h("th", {}, "Condition"), h("th", { class: "num" }, "Window"), h("th", {}, "Scope"), h("th", {}, "Status"), h("th", {}, ""))),
        h("tbody", {}, alertRows))) : h("div", { class: "card empty" }, "No alerts yet.")),
    alertForm);
}

// ---------- settings ----------

async function settingsPage() {
  const [settings, pricing, health] = await Promise.all([api("/api/settings"), api("/api/pricing"), api("/api/health")]);
  const msg = h("div");
  const days = h("input", { type: "number", min: 1, placeholder: "keep forever", value: settings.retention_days ?? "" });
  const retention = h("form", { class: "card pad-m section", onsubmit: async (ev) => {
    ev.preventDefault();
    await attempt(msg, () => apiPut("/api/settings", { ...settings, retention_days: days.value ? Number(days.value) : null }), "Saved.");
  } }, h("h2", {}, "Data retention"),
    h("p", { class: "hint" }, "Traces older than this are deleted (with their runs and feedback) by an hourly job. Leave empty to keep everything."),
    h("div", { class: "actions" }, h("label", { class: "field" }, "Days", days), h("button", { class: "primary", type: "submit" }, "Save"),
      h("button", { type: "button", onclick: () => attempt(msg, () => apiPost("/api/retention/run"), (r) => `Deleted ${r.traces} traces, ${r.runs} runs, ${r.feedback} feedback entries.`) }, "Apply now")));

  const rows = h("tbody");
  const drawRows = (models) => rows.replaceChildren(...models.map((m) => h("tr", {},
    h("td", { class: "mono" }, m.model), h("td", { class: "muted" }, m.provider || ""),
    h("td", { class: "num" }, "$" + m.input_per_mtok), h("td", { class: "num" }, "$" + m.output_per_mtok))));
  drawRows(pricing.models);
  const editor = h("textarea", { rows: 14 });
  editor.value = JSON.stringify(pricing, null, 2);
  const pricingCard = h("div", { class: "card pad-m section" }, h("h2", {}, "Model pricing"),
    h("p", { class: "hint" }, "USD per million tokens. Models match by longest prefix after removing any provider: prefix. Costs are computed at ingest; saving recomputes every stored run. A run's metadata.cost_usd overrides the table."),
    h("div", { class: "table-wrap", style: { maxHeight: "320px", marginBottom: "10px" } }, h("table", {},
      h("thead", {}, h("tr", {}, h("th", {}, "Model (prefix)"), h("th", {}, "Provider"), h("th", { class: "num" }, "Input / 1M"), h("th", { class: "num" }, "Output / 1M"))), rows)),
    h("details", {}, h("summary", {}, "Edit as JSON"), editor,
      h("div", { class: "actions", style: { marginTop: "8px" } },
        h("button", { class: "primary", onclick: async () => {
          let p;
          try { p = JSON.parse(editor.value); } catch (e) { msg.replaceChildren(notice("err", "Invalid JSON: " + e.message)); return; }
          const r = await attempt(msg, () => apiPut("/api/pricing", p), (x) => `Saved; ${x.priced_runs} runs priced.`);
          if (r) drawRows(p.models || []);
        } }, "Save pricing"),
        h("button", { onclick: async () => { const r = await attempt(msg, () => apiDelete("/api/pricing"), "Restored the built-in table."); if (r) route(); } }, "Reset to built-in"),
        h("button", { onclick: () => attempt(msg, () => apiPost("/api/pricing/recompute"), (x) => `Recomputed; ${x.priced_runs} runs priced.`) }, "Recompute costs"))));

  return h("div", {},
    pageHead("Settings", `sinew-scope ${health.version} · full-text search ${health.fts ? "FTS5" : "LIKE fallback"} · background engine ${health.engine ? "on" : "off"} · OTLP protobuf ${health.otlp_protobuf ? "on" : "off"}`),
    msg, retention, pricingCard);
}

// ---------- pairwise ----------

function pairwiseTable(list) {
  return h("div", { class: "card table-wrap" }, h("table", {},
    h("thead", {}, h("tr", {}, h("th", {}, "Comparison"), h("th", {}, "Dataset"), h("th", {}, "Created"), h("th", {}, "Result"))),
    h("tbody", {}, list.map((p) => h("tr", { class: "click", onclick: () => go("/pairwise/" + encodeURIComponent(p.id)) },
      h("td", {}, h("a", { href: "#/pairwise/" + encodeURIComponent(p.id) }, p.name)),
      h("td", {}, p.dataset), h("td", { class: "small muted" }, fmt.time(p.created_ms)),
      h("td", { class: "small" }, Object.entries(p.summary || {}).map(([k, s]) => h("div", {}, `${k}: A ${s.wins_a} · B ${s.wins_b} · tie ${s.ties}`))))))));
}

async function pairwisePage(id) {
  const r = await api("/api/pairwise/" + encodeURIComponent(id));
  const keys = Object.keys(r.summary).sort();
  const winBar = (s) => {
    const total = s.wins_a + s.wins_b + s.ties || 1;
    return h("div", { style: { display: "flex", height: "10px", borderRadius: "5px", overflow: "hidden", minWidth: "160px", background: "var(--panel-2)" } },
      h("div", { title: "A wins", style: { width: (s.wins_a / total) * 100 + "%", background: "var(--chart-2)" } }),
      h("div", { title: "ties", style: { width: (s.ties / total) * 100 + "%", background: "var(--border)" } }),
      h("div", { title: "B wins", style: { width: (s.wins_b / total) * 100 + "%", background: "var(--chart-1)" } }));
  };
  const pref = (p) => p === "a" ? h("span", { class: "pill", style: { color: "var(--chart-2)" } }, "A") : p === "b" ? h("span", { class: "pill", style: { color: "var(--chart-1)" } }, "B")
    : p === "tie" ? h("span", { class: "pill muted" }, "tie") : h("span", { class: "muted" }, "–");
  return h("div", {},
    h("div", { class: "page-head" }, h("div", {},
      h("div", { class: "crumbs" }, h("a", { href: "#/experiments" }, "Experiments"), " / ", h("a", { href: "#/dataset/" + encodeURIComponent(r.dataset) }, r.dataset)),
      h("h1", {}, r.name), h("div", { class: "muted" }, fmt.time(r.created_ms), " · ",
        h("a", { href: "#/compare?a=" + encodeURIComponent(r.experiment_a.id) + "&b=" + encodeURIComponent(r.experiment_b.id) }, "score comparison")))),
    h("div", { class: "grid-2 section" },
      h("div", { class: "card pad-s" }, h("div", { class: "section-title" }, "A"), h("h2", {}, h("a", { href: "#/experiment/" + encodeURIComponent(r.experiment_a.id) }, r.experiment_a.name))),
      h("div", { class: "card pad-s" }, h("div", { class: "section-title" }, "B"), h("h2", {}, h("a", { href: "#/experiment/" + encodeURIComponent(r.experiment_b.id) }, r.experiment_b.name)))),
    h("div", { class: "section card table-wrap" }, h("table", {},
      h("thead", {}, h("tr", {}, h("th", {}, "Evaluator"), h("th", { class: "num" }, "A wins"), h("th", { class: "num" }, "B wins"), h("th", { class: "num" }, "Ties"),
        h("th", { class: "num" }, "Undecided"), h("th", { class: "num" }, "Win rate A / B"), h("th", {}, ""))),
      h("tbody", {}, keys.map((k) => { const s = r.summary[k]; return h("tr", {}, h("td", {}, k), h("td", { class: "num" }, s.wins_a), h("td", { class: "num" }, s.wins_b),
        h("td", { class: "num" }, s.ties), h("td", { class: "num" }, s.undecided), h("td", { class: "num" }, `${fmt.pct(s.win_rate_a)} / ${fmt.pct(s.win_rate_b)}`), h("td", {}, winBar(s))); })))),
    h("div", { class: "card table-wrap" }, h("table", {},
      h("thead", {}, h("tr", {}, h("th", {}, "Inputs"), h("th", {}, "Reference"), h("th", {}, "Output A"), h("th", {}, "Output B"), keys.map((k) => h("th", {}, k)))),
      h("tbody", {}, r.results.map((x) => h("tr", {},
        h("td", {}, cellJson(x.inputs), x.repetition ? h("div", { class: "small muted" }, "rep " + (x.repetition + 1)) : null),
        h("td", {}, cellJson(x.reference)),
        h("td", {}, x.error ? h("div", { class: "status-err small" }, x.error) : cellJson(x.output_a)),
        h("td", {}, cellJson(x.output_b)),
        keys.map((k) => { const sc = x.scores.find((y) => y.key === k); return h("td", {}, sc ? pref(sc.preference) : "–", sc && sc.reason ? h("div", { class: "reason", title: sc.reason }, sc.reason) : null); })))))));
}
