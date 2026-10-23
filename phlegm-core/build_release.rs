//! Embed provenance in the binary: running an old binary inside a tagged checkout
//! must never make it eligible for publication.
use std::{env, path::Path, process::Command};

fn git(root: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_owned())
}

pub fn emit() {
    let manifest = env::var("CARGO_MANIFEST_DIR").unwrap();
    let root = Path::new(&manifest).parent().unwrap();
    // A release guard must notice edits, commits, tag changes, and worktrees.
    for path in [
        "Cargo.toml",
        "Cargo.lock",
        "phlegm-core",
        "phlegm-cli",
        "benchmarks/reference",
    ] {
        println!("cargo:rerun-if-changed={}", root.join(path).display());
    }
    for name in ["HEAD", "index", "refs", "packed-refs"] {
        if let Some(path) = git(root, &["rev-parse", "--git-path", name]) {
            println!("cargo:rerun-if-changed={}", root.join(path).display());
        }
    }
    let version = env::var("CARGO_PKG_VERSION").unwrap();
    let source = inspect(root, &version);
    let rustc = Command::new(env::var("RUSTC").unwrap_or_else(|_| "rustc".into()))
        .arg("--version")
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_owned())
        .unwrap_or_default();
    let opt_level = env::var("OPT_LEVEL").unwrap_or_default();
    for (key, value) in [
        ("PHLEGM_BUILD_TAG", source.tag),
        ("PHLEGM_BUILD_COMMIT", source.commit),
        ("PHLEGM_BUILD_DIRTY", source.dirty.to_string()),
        ("PHLEGM_BUILD_OPT_LEVEL", opt_level),
        ("PHLEGM_BUILD_TARGET", env::var("TARGET").unwrap()),
        ("PHLEGM_BUILD_RUSTC", rustc),
    ] {
        println!("cargo:rustc-env={key}={value}");
    }
}

pub struct SourceProvenance {
    pub tag: String,
    pub commit: String,
    pub dirty: bool,
}

pub fn inspect(root: &Path, version: &str) -> SourceProvenance {
    let expected = format!("v{version}");
    let tags = git(root, &["tag", "--points-at", "HEAD"]).unwrap_or_default();
    let tag = tags
        .lines()
        .find(|tag| *tag == expected)
        .unwrap_or("")
        .to_owned();
    let commit = git(root, &["rev-parse", "HEAD"]).unwrap_or_default();
    let tracked = git(root, &["status", "--porcelain", "--untracked-files=no"]);
    let untracked = git(
        root,
        &[
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            "phlegm-core",
            "phlegm-cli",
        ],
    );
    let clean = tracked.as_deref() == Some("") && untracked.as_deref() == Some("");
    SourceProvenance {
        tag,
        commit,
        dirty: !clean,
    }
}
