//! Graph diagrams: the shape of a compiled graph as data, Mermaid or DOT.

use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use crate::{
    graph::{CompiledGraph, Edge},
    END, START,
};

/// The static shape of a [`CompiledGraph`]: nodes, edges and the policies
/// attached to them. Serializable, so UIs can draw it; also renders to
/// [Mermaid](GraphView::to_mermaid) and [Graphviz DOT](GraphView::to_dot).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphView {
    /// Nodes in the order they were added, without `__start__`/`__end__`.
    pub nodes: Vec<GraphNode>,
    /// Edges, including the entry edge from [`START`] and edges to [`END`].
    pub edges: Vec<GraphEdge>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphNode {
    pub name: String,
    pub kind: NodeKind,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub interrupt_before: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub interrupt_after: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub retry: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub cache: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub timeout: bool,
    /// The nested graph of a subgraph node.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subgraph: Option<Box<GraphView>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    /// Returns a new state.
    Node,
    /// Returns a [`Command`](crate::Command) and may jump anywhere it declared.
    Command,
    /// A compiled graph used as a node.
    Subgraph,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphEdge {
    pub from: String,
    pub to: String,
    pub kind: EdgeKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeKind {
    /// Always taken (several static edges from one node run in parallel).
    Static,
    /// One possible target of a router.
    Conditional,
    /// One possible target of a `Send` fan-out (map step).
    Send,
    /// A destination a command node declared for `goto`.
    Command,
}

impl<S> CompiledGraph<S> {
    /// The graph's shape, for diagrams and tooling.
    ///
    /// ```
    /// use sinew_graph::{StateGraph, END};
    ///
    /// let mut g = StateGraph::<u32>::new();
    /// g.add_node("plan", |s: u32| async move { Ok(s + 1) });
    /// g.add_node("act", |s: u32| async move { Ok(s * 2) });
    /// g.set_entry_point("plan");
    /// g.add_edge("plan", "act");
    /// g.add_conditional_edges("act", |s: &u32| if *s > 10 { END.into() } else { "plan".into() }, ["plan", END]);
    /// let graph = g.compile().unwrap();
    ///
    /// let mermaid = graph.to_mermaid();
    /// assert!(mermaid.contains("plan --> act"));
    /// assert!(mermaid.contains("act -.-> plan"));
    /// ```
    pub fn view(&self) -> GraphView {
        let mut nodes = Vec::with_capacity(self.order.len());
        for name in self.order.iter() {
            let Some(spec) = self.nodes.get(name) else {
                continue;
            };
            let kind = if spec.subgraph.is_some() {
                NodeKind::Subgraph
            } else if spec.command {
                NodeKind::Command
            } else {
                NodeKind::Node
            };
            nodes.push(GraphNode {
                name: name.clone(),
                kind,
                interrupt_before: self.interrupt_before.contains(name),
                interrupt_after: self.interrupt_after.contains(name),
                retry: spec.retry.is_some(),
                cache: spec.cache.is_some(),
                timeout: spec.timeout.is_some(),
                subgraph: spec.subgraph.clone().map(Box::new),
            });
        }

        let mut edges = Vec::new();
        let push_edge = |edges: &mut Vec<GraphEdge>, from: &str, e: &Edge<S>| {
            let kind = match e {
                Edge::Static(_) => EdgeKind::Static,
                Edge::Router { .. } => EdgeKind::Conditional,
                Edge::Send { .. } => EdgeKind::Send,
            };
            for t in e.targets() {
                edges.push(GraphEdge {
                    from: from.to_string(),
                    to: t.to_string(),
                    kind,
                });
            }
        };
        push_edge(&mut edges, START, &self.entry);
        for name in self.order.iter() {
            if let Some(e) = self.edges.get(name) {
                push_edge(&mut edges, name, e);
            }
            if let Some(spec) = self.nodes.get(name) {
                for d in &spec.destinations {
                    let dup = edges.iter().any(|x| &x.from == name && &x.to == d);
                    if !dup {
                        edges.push(GraphEdge {
                            from: name.clone(),
                            to: d.clone(),
                            kind: EdgeKind::Command,
                        });
                    }
                }
            }
        }
        GraphView { nodes, edges }
    }

    /// Mermaid flowchart (`flowchart TD`), ready for GitHub Markdown.
    pub fn to_mermaid(&self) -> String {
        self.view().to_mermaid()
    }

    /// Graphviz DOT (`dot -Tpng graph.dot -o graph.png`).
    pub fn to_dot(&self) -> String {
        self.view().to_dot()
    }
}

/// Mermaid node ids must be simple; names may contain anything.
fn mermaid_id(prefix: &str, name: &str) -> String {
    let mut id = String::from(prefix);
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            id.push(c);
        } else {
            let _ = write!(id, "_{:x}_", c as u32);
        }
    }
    id
}

fn mermaid_label(s: &str) -> String {
    s.replace('"', "#quot;")
}

impl GraphView {
    /// Mermaid flowchart. Static edges are solid, conditional and `Send`
    /// edges dotted, command destinations dashed with a `goto` label;
    /// interrupts are marked with ⏸ and subgraphs drawn nested.
    pub fn to_mermaid(&self) -> String {
        let mut out = String::from("flowchart TD\n");
        out.push_str("    __start__([\"START\"]):::terminal\n");
        out.push_str("    __end__([\"END\"]):::terminal\n");
        self.mermaid_body(&mut out, "", "    ");
        out.push_str("    classDef terminal fill:#eee,stroke:#999,color:#333\n");
        out.push_str("    classDef paused stroke:#d97706,stroke-width:2px\n");
        out
    }

    fn mermaid_body(&self, out: &mut String, prefix: &str, indent: &str) {
        let id = |name: &str| match name {
            START if prefix.is_empty() => "__start__".to_string(),
            END if prefix.is_empty() => "__end__".to_string(),
            START => format!("{prefix}__start__"),
            END => format!("{prefix}__end__"),
            n => mermaid_id(prefix, n),
        };
        if !prefix.is_empty() {
            let _ = writeln!(out, "{indent}{}([\"start\"]):::terminal", id(START));
            let _ = writeln!(out, "{indent}{}([\"end\"]):::terminal", id(END));
        }
        for n in &self.nodes {
            let mut label = mermaid_label(&n.name);
            if n.interrupt_before {
                label = format!("⏸ {label}");
            }
            if n.interrupt_after {
                label = format!("{label} ⏸");
            }
            let mut tags = Vec::new();
            if n.retry {
                tags.push("retry");
            }
            if n.cache {
                tags.push("cache");
            }
            if n.timeout {
                tags.push("timeout");
            }
            if !tags.is_empty() {
                label = format!("{label}<br/><small>{}</small>", tags.join(" · "));
            }
            let paused = if n.interrupt_before || n.interrupt_after {
                ":::paused"
            } else {
                ""
            };
            match (&n.subgraph, n.kind) {
                (Some(sub), _) => {
                    let _ = writeln!(out, "{indent}subgraph {}[\"{label}\"]", id(&n.name));
                    let inner = format!("{}__", id(&n.name));
                    sub.mermaid_body(out, &inner, &format!("{indent}    "));
                    let _ = writeln!(out, "{indent}end");
                }
                (None, NodeKind::Command) => {
                    let _ = writeln!(out, "{indent}{}{{{{\"{label}\"}}}}{paused}", id(&n.name));
                }
                (None, _) => {
                    let _ = writeln!(out, "{indent}{}[\"{label}\"]{paused}", id(&n.name));
                }
            }
        }
        for e in &self.edges {
            let arrow = match e.kind {
                EdgeKind::Static => "-->",
                EdgeKind::Conditional => "-.->",
                EdgeKind::Send => "-. send .->",
                EdgeKind::Command => "-. goto .->",
            };
            let _ = writeln!(out, "{indent}{} {arrow} {}", id(&e.from), id(&e.to));
        }
    }

    /// Graphviz DOT digraph.
    pub fn to_dot(&self) -> String {
        let mut out = String::from("digraph G {\n    rankdir=TB;\n    node [shape=box, style=rounded, fontname=\"Helvetica\"];\n");
        out.push_str("    \"__start__\" [label=\"START\", shape=oval, style=filled, fillcolor=\"#eeeeee\"];\n");
        out.push_str(
            "    \"__end__\" [label=\"END\", shape=oval, style=filled, fillcolor=\"#eeeeee\"];\n",
        );
        self.dot_body(&mut out, "", "    ");
        out.push_str("}\n");
        out
    }

    fn dot_body(&self, out: &mut String, prefix: &str, indent: &str) {
        let q = |s: &str| format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""));
        let id = |name: &str| match name {
            START | END if prefix.is_empty() => q(name),
            n => q(&format!("{prefix}{n}")),
        };
        if !prefix.is_empty() {
            let _ = writeln!(out, "{indent}{} [label=\"start\", shape=oval];", id(START));
            let _ = writeln!(out, "{indent}{} [label=\"end\", shape=oval];", id(END));
        }
        for n in &self.nodes {
            let mut label = n.name.clone();
            if n.interrupt_before {
                label = format!("⏸ {label}");
            }
            if n.interrupt_after {
                label = format!("{label} ⏸");
            }
            if let Some(sub) = &n.subgraph {
                let _ = writeln!(
                    out,
                    "{indent}subgraph {} {{\n{indent}    label={};\n{indent}    style=rounded;",
                    q(&format!("cluster_{prefix}{}", n.name)),
                    q(&label)
                );
                sub.dot_body(
                    out,
                    &format!("{prefix}{}/", n.name),
                    &format!("{indent}    "),
                );
                let _ = writeln!(out, "{indent}}}");
                continue;
            }
            let shape = if n.kind == NodeKind::Command {
                ", shape=hexagon"
            } else {
                ""
            };
            let color = if n.interrupt_before || n.interrupt_after {
                ", color=\"#d97706\", penwidth=2"
            } else {
                ""
            };
            let _ = writeln!(
                out,
                "{indent}{} [label={}{shape}{color}];",
                id(&n.name),
                q(&label)
            );
        }
        for e in &self.edges {
            let (from, to) = (
                self.dot_endpoint(&e.from, prefix, true),
                self.dot_endpoint(&e.to, prefix, false),
            );
            let style = match e.kind {
                EdgeKind::Static => "",
                EdgeKind::Conditional => " [style=dashed]",
                EdgeKind::Send => " [style=dashed, label=\"send\"]",
                EdgeKind::Command => " [style=dotted, label=\"goto\"]",
            };
            let _ = writeln!(out, "{indent}{} -> {}{style};", q(&from), q(&to));
        }
    }

    /// DOT cannot point an edge at a cluster: use its start or end node.
    fn dot_endpoint(&self, name: &str, prefix: &str, from: bool) -> String {
        if (name == START || name == END) && prefix.is_empty() {
            return name.to_string();
        }
        let is_sub = self
            .nodes
            .iter()
            .any(|n| n.name == name && n.subgraph.is_some());
        if is_sub {
            let inner = if from { END } else { START };
            format!("{prefix}{name}/{inner}")
        } else {
            format!("{prefix}{name}")
        }
    }
}
