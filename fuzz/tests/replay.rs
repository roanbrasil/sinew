//! Replays the committed seed corpus (`fuzz/seeds/<target>/`) and the
//! minimized crash reproducers (`fuzz/regressions/<target>/`) through every
//! fuzz target's checks. Runs on stable with `cargo test` in `fuzz/`, so the
//! inputs that once crashed keep being checked without nightly or libFuzzer.

use std::{
    fs,
    path::{Path, PathBuf},
};

use sinew_fuzz::TARGETS;

fn inputs(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file())
        .collect();
    files.sort();
    files
}

fn replay(kind: &str) -> usize {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join(kind);
    let mut count = 0;
    for target in TARGETS {
        for path in inputs(&root.join(target.name)) {
            let data = fs::read(&path).expect("read input");
            let result = std::panic::catch_unwind(|| (target.run)(&data));
            assert!(
                result.is_ok(),
                "{} failed on {}",
                target.name,
                path.display()
            );
            count += 1;
        }
    }
    count
}

#[test]
fn seeds_pass_every_target() {
    let n = replay("seeds");
    assert!(
        n >= TARGETS.len(),
        "expected seeds for every target, found {n}"
    );
}

#[test]
fn regressions_stay_fixed() {
    replay("regressions");
}

#[test]
fn every_target_has_seeds_and_a_binary() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for target in TARGETS {
        assert!(
            !inputs(&root.join("seeds").join(target.name)).is_empty(),
            "no seeds for {}",
            target.name
        );
        assert!(
            root.join("fuzz_targets")
                .join(format!("{}.rs", target.name))
                .is_file(),
            "no fuzz_targets/{}.rs",
            target.name
        );
    }
}

#[test]
fn empty_and_tiny_inputs() {
    for target in TARGETS {
        for data in [&b""[..], b"\0", b"\n", b"{", b"\xff\xfe"] {
            (target.run)(data);
        }
    }
}
