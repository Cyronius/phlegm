//! Resolve the resident reference bridge for normal generation.
use std::path::{Path, PathBuf};

const INDEX: &str = include_str!("../../bridges/index.json");
const BINARY: &str = if cfg!(windows) { "phlegm-reference-bench.exe" } else { "phlegm-reference-bench" };

/// Explicit paths take precedence; automatic family selection reads the selected kernel manifest.
pub fn resolve_bridge(explicit: Option<&Path>, kernels: &Path) -> Result<PathBuf, String> {
    let exe = std::env::current_exe().map_err(|e| format!("cannot locate executable: {e}"))?;
    let env = std::env::var_os("PHLEGM_REFERENCE_BRIDGE").map(PathBuf::from);
    // Cargo builds the in-tree bridge beside phlegm; tests live one directory deeper.
    let in_tree = if cfg!(debug_assertions) { option_env!("PHLEGM_IN_TREE_BRIDGE").map(Path::new) } else { None };
    // Stopgap: the bridges built in this workspace's `target/npu-dev/bridges`. The
    // workspace root is the parent of this crate's directory at compile time. Only a last
    // resort; installed builds use the adjacent binary or `bridges/index.json` beside the exe.
    let dev = Path::new(env!("CARGO_MANIFEST_DIR")).parent().map(|root| root.join("target/npu-dev/bridges"));
    resolve(explicit, env.as_deref(), exe.parent().unwrap_or(Path::new(".")), kernels, in_tree, dev.as_deref())
}

fn resolve(explicit: Option<&Path>, env: Option<&Path>, exe_dir: &Path, kernels: &Path, in_tree: Option<&Path>, dev: Option<&Path>) -> Result<PathBuf, String> {
    if let Some(path) = explicit.or(env) { return checked(path.to_path_buf()); }
    let manifest = kernels.join("manifest.json");
    let bytes = std::fs::read(&manifest).map_err(|e| format!("cannot read kernel manifest {}: {e}", manifest.display()))?;
    let value: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| format!("invalid kernel manifest {}: {e}", manifest.display()))?;
    let family = value.get("family").and_then(|v| v.as_str()).ok_or_else(|| format!("missing family in kernel manifest {}", manifest.display()))?;
    let index_path = exe_dir.join("bridges/index.json");
    let index = if index_path.is_file() {
        std::fs::read_to_string(&index_path).map_err(|e| format!("cannot read bridge index {}: {e}", index_path.display()))?
    } else { INDEX.to_owned() };
    let map: serde_json::Value = serde_json::from_str(&index).map_err(|e| format!("invalid bridge index: {e}"))?;
    // Stopgap (P1 S2-S7): a family listed in the index still needs its per-family bridge; the
    // in-tree engine does not carry its patches yet. Never hand such a family to the in-tree bridge.
    if let Some(name) = map.get(family).and_then(|v| v.as_str()) {
        let relative = Path::new(name);
        if relative.components().any(|c| !matches!(c, std::path::Component::Normal(_))) { return Err(format!("invalid bridge index entry for {family}: {name}")); }
        let installed = exe_dir.join("bridges").join(relative).join(BINARY);
        if installed.is_file() { return Ok(installed); }
        if let Some(root) = dev { let path = root.join(relative).join(BINARY); if path.is_file() { return Ok(path); } }
        return Err(format!("missing bridge binary for family {family}: {} (the in-tree engine does not serve this family yet)", installed.display()));
    }
    // Every other family: the in-tree bridge, installed beside phlegm or built by Cargo.
    let adjacent = exe_dir.join(BINARY);
    if adjacent.is_file() { return Ok(adjacent); }
    if let Some(path) = in_tree.filter(|path| path.is_file()) { return Ok(path.to_path_buf()); }
    Err(format!("unknown kernel family {family}: no per-family bridge and no in-tree bridge at {}", adjacent.display()))
}
fn checked(path: PathBuf) -> Result<PathBuf, String> {
    if path.is_file() { Ok(path) } else { Err(format!("missing bridge binary: {}", path.display())) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn resolution_order_and_errors() {
        let dir = std::env::temp_dir().join(format!("phlegm-resolve-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let exe = dir.join("exe"); let kernels = dir.join("kernels"); let dev = dir.join("dev");
        std::fs::create_dir_all(&exe).unwrap(); std::fs::create_dir_all(&kernels).unwrap();
        let flag = dir.join("flag"); let env = dir.join("env"); std::fs::write(&flag, "").unwrap(); std::fs::write(&env, "").unwrap();
        let in_tree = dir.join(BINARY);
        let run = |f: Option<&Path>, e: Option<&Path>| resolve(f, e, &exe, &kernels, Some(&in_tree), Some(&dev));
        assert_eq!(run(Some(&flag), Some(&env)).unwrap(), flag); assert_eq!(run(None, Some(&env)).unwrap(), env);
        assert!(run(Some(&dir.join("missing")), None).unwrap_err().contains("missing bridge binary:"));
        assert!(run(None, None).unwrap_err().contains("cannot read kernel manifest"));
        // A family outside the index uses the in-tree bridge: adjacent first, then Cargo's.
        std::fs::write(kernels.join("manifest.json"), r#"{"family":"unknown"}"#).unwrap();
        assert!(run(None,None).unwrap_err().starts_with("unknown kernel family unknown"));
        std::fs::write(&in_tree, "").unwrap(); assert_eq!(run(None,None).unwrap(), in_tree);
        let adjacent = exe.join(BINARY); std::fs::write(&adjacent, "").unwrap(); assert_eq!(run(None, None).unwrap(), adjacent);
        // An indexed family never falls back to the in-tree bridge, even when one sits beside phlegm.
        std::fs::write(kernels.join("manifest.json"), r#"{"family":"lfm3"}"#).unwrap();
        let err = run(None,None).unwrap_err();
        assert!(err.contains("missing bridge binary for family lfm3") && err.contains("does not serve this family"));
        let fallback = dev.join("lfm3").join(BINARY); std::fs::create_dir_all(fallback.parent().unwrap()).unwrap(); std::fs::write(&fallback, "").unwrap(); assert_eq!(run(None,None).unwrap(), fallback);
        let installed = exe.join("bridges/custom").join(BINARY); std::fs::create_dir_all(installed.parent().unwrap()).unwrap(); std::fs::write(&installed, "").unwrap(); std::fs::write(exe.join("bridges/index.json"), r#"{"lfm3":"custom"}"#).unwrap(); assert_eq!(run(None,None).unwrap(), installed);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
