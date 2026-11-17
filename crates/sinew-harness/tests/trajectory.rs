//! Grading harness runs by their trajectory with `sinew-eval` (offline,
//! `ScriptedModel`).

use std::{fs, sync::Arc};

use serde_json::{json, Value};
use sinew_core::{mock::ScriptedModel, Message, ToolCall};
use sinew_eval::{
    ArgumentMatch, Dataset, Example, Experiment, NoRepeatedCalls, PassRate, RelativeEfficiency,
    ToolUsage, Trajectory, TrajectoryBudget, TrajectoryJudge, TrajectoryMatch,
};
use sinew_harness::{Agent, AgentOutcome, Pricing};
use sinew_trace::{MemorySink, Tracer};

fn call(id: &str, name: &str, args: Value) -> ToolCall {
    ToolCall {
        id: id.into(),
        name: name.into(),
        arguments: args,
    }
}

fn tool_msg(calls: Vec<ToolCall>) -> Message {
    Message::assistant_with_tools("", calls)
}

fn workspace() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("notes.md"),
        "# Notes\nThe launch is on Friday.\nBudget: 42k.\n",
    )
    .unwrap();
    dir
}

/// Scripted behaviours: a direct agent, one that loops on the same read,
/// and one that searches with grep instead of reading the file.
fn script(style: &str) -> Vec<Message> {
    let read = || call("r", "read_file", json!({"path": "notes.md"}));
    match style {
        "direct" => vec![
            tool_msg(vec![read()]),
            Message::assistant("The launch is on Friday."),
        ],
        "looping" => vec![
            tool_msg(vec![read()]),
            tool_msg(vec![call("r2", "read_file", json!({"path": "notes.md"}))]),
            Message::assistant("The launch is on Friday."),
        ],
        _ => vec![
            tool_msg(vec![call(
                "g",
                "grep",
                json!({"pattern": "launch", "path": "."}),
            )]),
            tool_msg(vec![read()]),
            Message::assistant("Friday."),
        ],
    }
}

/// Runs one harness turn and returns the trajectory of that turn with the
/// harness usage totals.
async fn run_agent(style: &str, question: &str) -> Result<Value, String> {
    let ws = workspace();
    let agent = Agent::builder()
        .model_arc(Arc::new(ScriptedModel::new(script(style))))
        .workspace(ws.path())
        .filesystem_read_only()
        .pricing(Pricing::new(1.0, 2.0))
        .loop_guard(None)
        .build()
        .map_err(|e| e.to_string())?;
    let AgentOutcome::Completed(r) = agent.run("t", question).await.map_err(|e| e.to_string())?
    else {
        return Err("approval required".into());
    };
    let state = agent
        .state("t")
        .await
        .map_err(|e| e.to_string())?
        .ok_or("no state")?;
    let trajectory = Trajectory::from_last_turn(&state.messages)
        .with_usage(r.usage.input_tokens, r.usage.output_tokens)
        .with_cost(r.usage.cost_usd);
    assert_eq!(trajectory.model_calls, r.steps);
    assert_eq!(trajectory.final_answer.as_deref(), Some(r.text.as_str()));
    Ok(trajectory.into_output())
}

#[tokio::test]
async fn harness_runs_are_graded_by_trajectory() {
    let expected = Trajectory::new().tool("read_file", json!({"path": "notes.md"}));
    let ds = Dataset::new("launch-date").with_example(
        Example::new(json!({"question": "When is the launch?"}))
            .with_id("launch")
            .with_expected_trajectory(expected),
    );

    let mut reports = Vec::new();
    for style in ["direct", "looping", "grep-first"] {
        let judge = Arc::new(ScriptedModel::new([Message::assistant(
            r#"{"score": 0.5, "reasoning": "scripted judge"}"#,
        )]));
        let report = Experiment::new(&ds)
            .trajectory_evaluator(TrajectoryMatch::exact().arguments(ArgumentMatch::Subset))
            .trajectory_evaluator(TrajectoryMatch::subsequence())
            .trajectory_evaluator(ToolUsage::new().require("read_file").forbid("shell"))
            .trajectory_evaluator(NoRepeatedCalls::new())
            .trajectory_evaluator(TrajectoryBudget::new().max_steps(2).max_cost(1.0))
            .trajectory_evaluator(RelativeEfficiency::new())
            .trajectory_evaluator(TrajectoryJudge::new(judge, "Was the path efficient?"))
            .summary_evaluator(PassRate::new("trajectory_exact", 1.0))
            .name(style)
            .run(move |inputs: Value| async move {
                run_agent(style, inputs["question"].as_str().unwrap_or_default()).await
            })
            .await;
        assert_eq!(report.aggregates.errors, 0, "{style}: {:?}", report.results);
        reports.push(report);
    }
    let score = |i: usize, key: &str| reports[i].results[0].score(key).unwrap();

    // The direct agent passes everything.
    for key in [
        "trajectory_exact",
        "trajectory_subsequence",
        "tool_usage",
        "no_repeated_calls",
        "trajectory_budget",
        "trajectory_efficiency",
    ] {
        assert_eq!(score(0, key), 1.0, "direct: {key}");
    }
    assert_eq!(score(0, "trajectory_judge"), 0.5);
    assert_eq!(
        reports[0].summary_score("pass_rate:trajectory_exact"),
        Some(1.0)
    );

    // The looping agent reads twice: right tool, wrong shape.
    assert_eq!(score(1, "trajectory_exact"), 0.0);
    assert_eq!(score(1, "trajectory_subsequence"), 1.0);
    assert_eq!(score(1, "no_repeated_calls"), 0.0);
    assert_eq!(score(1, "trajectory_budget"), 0.0);
    assert_eq!(score(1, "trajectory_efficiency"), 0.5);

    // grep first, then the read: extra step, but no repeats.
    assert_eq!(score(2, "trajectory_exact"), 0.0);
    assert_eq!(score(2, "trajectory_subsequence"), 1.0);
    assert_eq!(score(2, "no_repeated_calls"), 1.0);
    assert_eq!(score(2, "tool_usage"), 1.0);
    assert_eq!(score(2, "trajectory_efficiency"), 0.5);
}

#[tokio::test]
async fn trajectory_from_harness_trace_matches_messages() {
    let ws = workspace();
    let sink = Arc::new(MemorySink::new());
    let agent = Agent::builder()
        .model_arc(Arc::new(ScriptedModel::new(script("grep-first"))))
        .workspace(ws.path())
        .filesystem_read_only()
        .tracer(Tracer::new().with_shared_sink(sink.clone()))
        .build()
        .unwrap();
    let r = agent.ask("When is the launch?").await.unwrap();

    let from_runs = Trajectory::from_runs(&sink.runs());
    assert_eq!(from_runs.tool_names(), vec!["grep", "read_file"]);
    assert_eq!(
        from_runs.tool_calls[1].arguments,
        json!({"path": "notes.md"})
    );
    assert!(from_runs.tool_calls[1]
        .result
        .as_deref()
        .unwrap()
        .contains("Friday"));
    assert_eq!(from_runs.model_calls, r.steps);
    assert!(from_runs.total_tokens().is_some_and(|n| n > 0));
    assert_eq!(from_runs.final_answer.as_deref(), Some("Friday."));
}
