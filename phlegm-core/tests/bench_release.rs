#[allow(dead_code)]
#[path = "../build_release.rs"]
mod build_release;

use std::{
    fs,
    path::Path,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

fn git(root: &Path, args: &[&str]) {
    let out = Command::new("git")
        .args([
            "-c",
            "user.name=Bench Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=disabled-hooks",
            "-c",
            "core.autocrlf=false",
        ])
        .args(args)
        .current_dir(root)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn release_provenance_requires_exact_tag_and_clean_source() {
    // A dedicated ignored fixture repository; never modify the working repo's
    // history, tags or configuration.
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../target/bench-release-tests")
        .join(format!(
            "{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
    fs::create_dir_all(root.join("phlegm-core/src")).unwrap();
    fs::write(root.join("phlegm-core/src/lib.rs"), "// fixture\n").unwrap();
    git(&root, &["init", "--quiet"]);
    git(&root, &["add", "."]);
    git(&root, &["commit", "--quiet", "-m", "fixture"]);
    let info = build_release::inspect(&root, "0.1.0");
    assert!(info.tag.is_empty());
    assert!(!info.dirty);
    assert_eq!(info.commit.len(), 40);

    git(&root, &["tag", "v9.9.9"]);
    assert!(build_release::inspect(&root, "0.1.0").tag.is_empty());
    git(&root, &["tag", "-a", "v0.1.0", "-m", "fixture release"]);
    assert_eq!(build_release::inspect(&root, "0.1.0").tag, "v0.1.0");

    // Unrelated scratch files don't invalidate the compiled engine.
    fs::write(root.join("scratch.txt"), "research").unwrap();
    assert!(!build_release::inspect(&root, "0.1.0").dirty);
    fs::write(root.join("phlegm-core/src/untracked.rs"), "// new source").unwrap();
    assert!(build_release::inspect(&root, "0.1.0").dirty);
    git(&root, &["add", "phlegm-core/src/untracked.rs"]);
    assert!(build_release::inspect(&root, "0.1.0").dirty);
    git(&root, &["commit", "--quiet", "-m", "next version"]);
    assert!(build_release::inspect(&root, "0.1.0").tag.is_empty());
    fs::write(root.join("phlegm-core/src/lib.rs"), "// modified\n").unwrap();
    assert!(build_release::inspect(&root, "0.1.0").dirty);
}
