//! Explicit selection of a complete, lower-precision kernel bundle.
use super::hash_file;
use serde::Deserialize;
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Deserialize)]
struct Metadata {
    schema_version: u32,
    precision: String,
    experimental: bool,
    base_manifest_sha256: String,
    turbo_manifest_sha256: String,
    arithmetic: String,
}

pub(super) fn select(base: &Path, route: &str) -> Result<PathBuf, String> {
    if !["auto", "block"].contains(&route) {
        return Err("--turbo requires block prefill; sequential prefill is unsupported".into());
    }
    let check = || -> Result<PathBuf, String> {
        let root = base.canonicalize().map_err(|e| e.to_string())?;
        let selected = root
            .join("turbo")
            .canonicalize()
            .map_err(|e| e.to_string())?;
        if selected == root || !selected.starts_with(&root) {
            return Err("turbo bundle escapes its base kernel directory".into());
        }
        let metadata: Metadata = serde_json::from_slice(
            &fs::read(selected.join("turbo.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| format!("invalid turbo.json: {e}"))?;
        if metadata.schema_version != 1
            || metadata.precision != "bfp16-hh"
            || !metadata.experimental
            || metadata.arithmetic != "HH only; low limbs ignored"
        {
            return Err("unsupported turbo precision or metadata schema".into());
        }
        if metadata.base_manifest_sha256 != hash_file(&root.join("manifest.json"))? {
            return Err("turbo bundle does not match the current base manifest".into());
        }
        if metadata.turbo_manifest_sha256 != hash_file(&selected.join("manifest.json"))? {
            return Err("turbo metadata does not match the selected manifest".into());
        }
        let manifest: serde_json::Value = serde_json::from_slice(
            &fs::read(selected.join("manifest.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| format!("invalid turbo manifest: {e}"))?;
        let block = &manifest["gguf"]["layer_types"]["dense"]["gemm_block"];
        let program = block["program"]
            .as_array()
            .ok_or("turbo requires a dense GGUF block program")?;
        if program.is_empty()
            || program
                .iter()
                .any(|step| step["fmt"].as_str() != Some("bfp16_hilo"))
            || block["attn_fa"]["psplit"].as_bool() != Some(true)
        {
            return Err(
                "turbo requires bfp16_hilo block projections and paired flash probabilities".into(),
            );
        }
        Ok(selected)
    };
    check()
        .map_err(|e| format!("--turbo unavailable: {e}; install a matching complete turbo bundle"))
}

pub(super) fn validate_loaded_route(route: Option<&str>) -> Result<(), String> {
    match route {
        Some("layer-major" | "gemm-block") => Ok(()),
        _ => Err("--turbo requires a confirmed batched prefill route; bridge returned an unsupported route".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn turbo_accepts_batched_scheduling_and_refuses_sequential_fallback() {
        assert!(validate_loaded_route(Some("layer-major")).is_ok());
        assert!(validate_loaded_route(Some("gemm-block")).is_ok());
        for route in [None, Some("sequential"), Some("auto"), Some("unknown")] {
            assert!(validate_loaded_route(route).is_err());
        }
    }

    #[test]
    fn bundle_is_explicit_and_bound_to_base() {
        let root = std::env::temp_dir().join(format!("phlegm-turbo-{}", std::process::id()));
        fs::create_dir_all(root.join("turbo")).unwrap();
        fs::write(root.join("manifest.json"), "{\"version\":1}").unwrap();
        assert!(select(&root, "auto").is_err());
        let mut metadata = json!({"schema_version":1,"precision":"bfp16-hh",
            "experimental":true,"arithmetic":"HH only; low limbs ignored",
            "turbo_manifest_sha256":"missing",
            "base_manifest_sha256":hash_file(&root.join("manifest.json")).unwrap()});
        let write = |v: &serde_json::Value| {
            fs::write(root.join("turbo/turbo.json"), v.to_string()).unwrap()
        };
        write(&metadata);
        assert!(select(&root, "block").is_err());
        let manifest = json!({"gguf":{"layer_types":{"dense":{"gemm_block":{
            "program":[{"op":"run","kernel":"gemm_n3072_k2048_f32","fmt":"bfp16_hilo",
                "args":["w","x","y"]}],"attn_fa":{"psplit":true}}}}}});
        let write_manifest = |value: &serde_json::Value| {
            fs::write(root.join("turbo/manifest.json"), value.to_string()).unwrap();
        };
        write_manifest(&manifest);
        assert!(select(&root, "auto").is_err());
        metadata["turbo_manifest_sha256"] =
            json!(hash_file(&root.join("turbo/manifest.json")).unwrap());
        write(&metadata);
        assert_eq!(
            select(&root, "auto").unwrap(),
            root.join("turbo").canonicalize().unwrap()
        );
        assert!(select(&root, "sequential").is_err());
        for bad in [
            json!({}),
            json!({"gguf":{"layer_types":{"dense":{"gemm_block":{
            "program":[],"attn_fa":{"psplit":true}}}}}}),
            json!({"gguf":{"layer_types":{"dense":{"gemm_block":{
                "program":[{"fmt":"bfp16"}],"attn_fa":{"psplit":true}}}}}}),
            json!({"gguf":{"layer_types":{"dense":{"gemm_block":{
                "program":[{"fmt":"bfp16_hilo"}],"attn_fa":{"psplit":false}}}}}}),
        ] {
            write_manifest(&bad);
            metadata["turbo_manifest_sha256"] =
                json!(hash_file(&root.join("turbo/manifest.json")).unwrap());
            write(&metadata);
            assert!(select(&root, "block").is_err());
        }
        write_manifest(&manifest);
        metadata["turbo_manifest_sha256"] =
            json!(hash_file(&root.join("turbo/manifest.json")).unwrap());
        metadata["precision"] = json!("standard");
        write(&metadata);
        assert!(select(&root, "auto").is_err());
        metadata["precision"] = json!("bfp16-hh");
        write(&metadata);
        fs::write(root.join("manifest.json"), "{\"version\":2}").unwrap();
        assert!(select(&root, "block").is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
