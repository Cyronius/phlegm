//! Phlegm-only integration; keep the imported C++ engine pristine.
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, env, fs, path::{Path, PathBuf}, process::Command};

fn collect(base: &Path, prefix: &str, files: &mut BTreeMap<String, String>) {
    for entry in fs::read_dir(base).expect("read engine inputs") {
        let path = entry.expect("engine input").path();
        let name = format!("{prefix}/{}", path.file_name().unwrap().to_str().unwrap());
        if path.is_dir() {
            collect(&path, &name, files);
        } else if matches!(path.extension().and_then(|s| s.to_str()), Some("cpp" | "hpp" | "h")) {
            files.insert(name, format!("{:x}", Sha256::digest(fs::read(path).unwrap())));
        }
    }
}

pub fn engine_hash(root: &Path) -> String {
    let mut files = BTreeMap::new();
    collect(&root.join("src/open_qwen36"), "engine", &mut files);
    collect(&root.join("src/include/nlohmann"), "json", &mut files);
    for (name, path) in [
        ("bridge/bridge.cpp", "benchmarks/reference/bridge.cpp"),
        ("harness/stream_patch.hpp", "open_kernels/harness/stream_patch.hpp"),
    ] {
        files.insert(name.into(), format!("{:x}", Sha256::digest(fs::read(root.join(path)).unwrap())));
    }
    let mut hash = Sha256::new();
    for (name, digest) in files {
        hash.update(format!("{name}\0{digest}\n"));
    }
    format!("{:x}", hash.finalize())
}

pub fn emit(root: &Path) {
    for path in ["src/open_qwen36", "src/include/nlohmann", "open_kernels/harness/stream_patch.hpp"] {
        println!("cargo:rerun-if-changed={}", root.join(path).display());
    }
    println!("cargo:rustc-env=PHLEGM_ENGINE_TREE_SHA256={}", engine_hash(root));
}

pub fn build(root: &Path, out: &Path, include: &Path, lib: &Path) {
    println!("cargo:rerun-if-env-changed=PYTHON");
    let bridge_out = out.join("bridge");
    let result = Command::new(env::var_os("PYTHON").unwrap_or_else(|| "python3".into()))
        .arg(root.join("benchmarks/reference/build.py"))
        .arg("--reference-root").arg(root)
        .arg("--out-dir").arg(&bridge_out)
        .arg("--xrt-include").arg(include)
        .arg("--xrt-lib").arg(lib.join("libxrt_coreutil.so.2"))
        .arg("--expected-engine-tree").arg(engine_hash(root))
        .output().expect("launch Python 3.11+ for the in-tree C++ bridge build (set PYTHON)");
    if !result.status.success() {
        panic!("in-tree C++ bridge build failed:\n{}\n{}", String::from_utf8_lossy(&result.stdout), String::from_utf8_lossy(&result.stderr));
    }
    // OUT_DIR is <target>/<profile>/build/<crate-id>/out, also with --target.
    let profile = out.ancestors().nth(3).expect("Cargo profile output directory");
    for name in ["phlegm-reference-bench", "phlegm-reference-bench.build.json"] {
        let dest = profile.join(name);
        let temp: PathBuf = profile.join(format!(".{name}.{}", std::process::id()));
        fs::copy(bridge_out.join(name), &temp).expect("stage adjacent bridge");
        fs::rename(temp, dest).expect("install adjacent bridge");
    }
    println!("cargo:warning=in-tree bridge: {}", profile.join("phlegm-reference-bench").display());
    println!("cargo:rustc-env=PHLEGM_IN_TREE_BRIDGE={}", profile.join("phlegm-reference-bench").display());
}
