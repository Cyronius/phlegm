//! Per-model runtime settings, in OpenFlowLM's two-file format (openflowlm-next #214,
//! `docs/config.md` there): a shipped registry, `config/model_list.json` (the models and the
//! shipped defaults), under the user's sparse overlay `settings.json`, with the settings
//! themselves defined in `config/settings.json`. Specified by SERVER-SETTINGS in
//! `specs/server/spec.md`.
//!
//! Precedence, lowest to highest (OpenFlowLM's order, plus the model directory's own
//! `generation_config.json` just above the built-in default):
//!   1. the setting's built-in `default_value`
//!   2. the model dir's `generation_config.json` (the sampling settings, HF semantics)
//!   3. the model's registry entry, at the setting's `registry_key`
//!   4. the shipped `defaults`
//!   5. the user's `defaults` (every model)
//!   6. the user's `settings.<family>.<size>` (one model)
//!   7. the setting's `PHLEGM_*` environment variable
//!   8. the command-line flag
//!   9. per-request API fields (the server applies those itself)
//!
//! Files: the registry is built into the binary (`PHLEGM_CONFIG_PATH` reads another); the user
//! file is `PHLEGM_SETTINGS_PATH`, else `$XDG_CONFIG_HOME/phlegm/settings.json`, else
//! `~/.config/phlegm/settings.json` (`%USERPROFILE%\.phlegm\settings.json` on Windows). Nothing
//! phlegm ships writes the user file; `phlegm config set/unset` does.

use serde::Deserialize;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

const SCHEMA: &str = include_str!("../config/settings.json");
const SHIPPED: &str = include_str!("../config/model_list.json");

#[derive(Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Enum,
    Int,
    Float,
    Bool,
    String,
}

#[derive(Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Scope {
    Global,
    PerModel,
}

#[derive(Deserialize, Clone, Debug)]
pub struct OptionDef {
    pub value: String,
    #[serde(default)]
    pub description: String,
}

/// One entry of `config/settings.json`.
#[derive(Deserialize, Clone, Debug)]
pub struct SettingDef {
    pub key: String,
    #[serde(default)]
    pub cli_flag: Option<String>,
    #[serde(default)]
    pub env_var: Option<String>,
    #[serde(default)]
    pub description: String,
    #[serde(rename = "type")]
    pub kind: Kind,
    #[serde(default)]
    pub options: Vec<OptionDef>,
    #[serde(default)]
    pub min_value: Option<f64>,
    #[serde(default)]
    pub min_exclusive: bool,
    #[serde(default)]
    pub max_value: Option<f64>,
    #[serde(default)]
    pub presets: Vec<Value>,
    pub scope: Scope,
    pub delivery: String,
    pub category: String,
    #[serde(default)]
    pub reload_required: bool,
    /// Where a registry entry keeps the model's value ("details.think", "sampling.top_k").
    #[serde(default)]
    pub registry_key: Option<String>,
    /// The `generation_config.json` field a model directory supplies the value from.
    #[serde(default)]
    pub generation_config: Option<String>,
    #[serde(default)]
    pub default_value: Option<Value>,
}

impl SettingDef {
    /// A value given as text (a flag, an env var, `config set`), checked.
    pub fn parse(&self, text: &str) -> Result<Value, String> {
        let t = text.trim();
        let v = match self.kind {
            Kind::Enum | Kind::String => Value::String(t.to_string()),
            Kind::Int => t
                .parse::<i64>()
                .map(Value::from)
                .map_err(|_| format!("{}: expected an integer, got {text:?}", self.key))?,
            Kind::Float => t
                .parse::<f64>()
                .map(Value::from)
                .map_err(|_| format!("{}: expected a number, got {text:?}", self.key))?,
            Kind::Bool => match t.to_ascii_lowercase().as_str() {
                "true" | "1" | "yes" | "on" => Value::Bool(true),
                "false" | "0" | "no" | "off" => Value::Bool(false),
                _ => return Err(format!("{}: expected true or false, got {text:?}", self.key)),
            },
        };
        self.check(&v)
    }

    /// A JSON value (a settings file, a registry entry), checked and normalized to the type.
    pub fn check(&self, v: &Value) -> Result<Value, String> {
        let key = &self.key;
        let range = |x: f64| -> Result<(), String> {
            if let Some(lo) = self.min_value {
                if x < lo || (self.min_exclusive && x == lo) {
                    let op = if self.min_exclusive { ">" } else { ">=" };
                    return Err(format!("{key}: {x} is out of range (must be {op} {lo})"));
                }
            }
            match self.max_value {
                Some(hi) if x > hi => Err(format!("{key}: {x} is out of range (must be <= {hi})")),
                _ => Ok(()),
            }
        };
        match self.kind {
            Kind::Enum => {
                let s = v.as_str().ok_or_else(|| format!("{key}: expected a string"))?;
                if self.options.iter().any(|o| o.value == s) {
                    Ok(Value::from(s))
                } else {
                    let all: Vec<_> = self.options.iter().map(|o| o.value.as_str()).collect();
                    Err(format!("{key}: {s:?} is not one of {}", all.join(", ")))
                }
            }
            Kind::String => v
                .as_str()
                .map(Value::from)
                .ok_or_else(|| format!("{key}: expected a string")),
            Kind::Bool => v
                .as_bool()
                .map(Value::from)
                .ok_or_else(|| format!("{key}: expected true or false")),
            Kind::Int => {
                let n = v
                    .as_i64()
                    .or_else(|| v.as_f64().filter(|f| f.fract() == 0.0).map(|f| f as i64))
                    .ok_or_else(|| format!("{key}: expected an integer"))?;
                range(n as f64)?;
                Ok(Value::from(n))
            }
            Kind::Float => {
                let x = v
                    .as_f64()
                    .filter(|x| x.is_finite())
                    .ok_or_else(|| format!("{key}: expected a finite number"))?;
                range(x)?;
                Ok(Value::from(x))
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// The setting's built-in default, the registry entry or the shipped defaults.
    Shipped,
    /// The model directory's generation_config.json.
    GenerationConfig,
    User,
    Env,
    Cli,
}

impl Source {
    pub fn name(self) -> &'static str {
        match self {
            Source::Shipped => "shipped",
            Source::GenerationConfig => "generation_config",
            Source::User => "user",
            Source::Env => "env",
            Source::Cli => "cli",
        }
    }
}

/// Every setting's effective value for one model, with where it came from.
#[derive(Clone, Debug)]
pub struct Resolved {
    pub tag: Option<String>,
    pub values: BTreeMap<String, (Value, Source)>,
}

impl Resolved {
    fn value(&self, key: &str) -> &Value {
        &self
            .values
            .get(key)
            .unwrap_or_else(|| panic!("setting {key} is not in config/settings.json"))
            .0
    }
    pub fn str(&self, key: &str) -> String {
        self.value(key).as_str().unwrap_or_default().to_string()
    }
    pub fn usize(&self, key: &str) -> usize {
        self.value(key).as_u64().unwrap_or(0) as usize
    }
    pub fn f32(&self, key: &str) -> f32 {
        self.value(key).as_f64().unwrap_or(0.0) as f32
    }
    pub fn bool(&self, key: &str) -> bool {
        self.value(key).as_bool().unwrap_or(false)
    }
}

pub struct Config {
    pub schema: Vec<SettingDef>,
    pub shipped: Value,
    /// None: the registry built into the binary.
    pub shipped_path: Option<PathBuf>,
    pub user: Value,
    pub user_path: PathBuf,
    /// Problems found in the user file (reported, not fatal), and default-change notices.
    pub warnings: Vec<String>,
}

/// Objects merge recursively; anything else in the overlay replaces; a null never overwrites.
pub fn deep_merge(base: &Value, overlay: &Value) -> Value {
    match (base, overlay) {
        (Value::Object(b), Value::Object(o)) => {
            let mut out = b.clone();
            for (k, v) in o {
                if v.is_null() {
                    continue;
                }
                let merged = match out.get(k) {
                    Some(prev) => deep_merge(prev, v),
                    None => v.clone(),
                };
                out.insert(k.clone(), merged);
            }
            Value::Object(out)
        }
        (_, Value::Null) => base.clone(),
        _ => overlay.clone(),
    }
}

/// "lfm3:1.7b" -> ("lfm3", "1.7b"); no colon -> (tag, "").
pub fn split_tag(tag: &str) -> (&str, &str) {
    tag.split_once(':').unwrap_or((tag, ""))
}

/// JSON with sorted keys and no spaces (what Python's `json.dumps(sort_keys=True,
/// separators=(",", ":"))` prints), whatever map order serde_json was built with.
fn canonical(v: &Value) -> String {
    match v {
        Value::Object(m) => {
            let mut keys: Vec<_> = m.keys().collect();
            keys.sort();
            let body: Vec<String> = keys
                .iter()
                .map(|k| format!("{}:{}", Value::from(k.as_str()), canonical(&m[*k])))
                .collect();
            format!("{{{}}}", body.join(","))
        }
        Value::Array(a) => format!("[{}]", a.iter().map(canonical).collect::<Vec<_>>().join(",")),
        _ => v.to_string(),
    }
}

/// The first 16 hex digits of sha256(canonical(defaults)): the registry's `defaults_version`.
pub fn defaults_version(defaults: &Value) -> String {
    let digest = Sha256::digest(canonical(defaults).as_bytes());
    digest.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

/// Where the user's settings.json lives.
pub fn user_settings_path() -> PathBuf {
    if let Some(p) = std::env::var_os("PHLEGM_SETTINGS_PATH") {
        return PathBuf::from(p);
    }
    if cfg!(windows) {
        let home = std::env::var_os("USERPROFILE").unwrap_or_default();
        return PathBuf::from(home).join(".phlegm").join("settings.json");
    }
    if let Some(x) = std::env::var_os("XDG_CONFIG_HOME").filter(|x| !x.is_empty()) {
        return PathBuf::from(x).join("phlegm").join("settings.json");
    }
    let home = std::env::var_os("HOME").unwrap_or_default();
    PathBuf::from(home).join(".config").join("phlegm").join("settings.json")
}

fn schema() -> Vec<SettingDef> {
    let doc: Value = serde_json::from_str(SCHEMA).expect("config/settings.json is JSON");
    serde_json::from_value(doc["settings"].clone()).expect("config/settings.json's settings")
}

/// A field of a JSON object by dotted path ("details.think").
fn pointer<'a>(v: &'a Value, dotted: &str) -> Option<&'a Value> {
    dotted.split('.').try_fold(v, |at, part| at.get(part))
}

/// What a model directory's generation_config.json says for a setting, with HF
/// `GenerationConfig` semantics: without `do_sample: true` it decodes greedily (temperature 0,
/// top-k and top-p unused) and applies only `repetition_penalty`; with it, a field it leaves out
/// takes HF's default (temperature 1, top_k 50, top_p 1).
fn generation_config_value(gc: &Value, field: &str) -> Option<Value> {
    let sampling = gc.get("do_sample").and_then(Value::as_bool).unwrap_or(false);
    let given = gc.get(field).filter(|v| !v.is_null()).cloned();
    match field {
        "repetition_penalty" => given,
        _ if !sampling => (field == "temperature").then(|| json!(0.0)),
        "temperature" => Some(given.unwrap_or(json!(1.0))),
        "top_k" => Some(given.unwrap_or(json!(50))),
        "top_p" => Some(given.unwrap_or(json!(1.0))),
        _ => given,
    }
}

impl Config {
    /// The shipped registry (`PHLEGM_CONFIG_PATH`, else built in) under the user's settings.
    pub fn load() -> Result<Config, String> {
        let (shipped, shipped_path) = match std::env::var_os("PHLEGM_CONFIG_PATH") {
            Some(p) => {
                let p = PathBuf::from(p);
                let text = std::fs::read_to_string(&p)
                    .map_err(|e| format!("reading {}: {e}", p.display()))?;
                let v = serde_json::from_str(&text)
                    .map_err(|e| format!("{} is not valid JSON: {e}", p.display()))?;
                (v, Some(p))
            }
            None => (serde_json::from_str(SHIPPED).expect("config/model_list.json is JSON"), None),
        };
        let user_path = user_settings_path();
        let mut warnings = Vec::new();
        let user = match std::fs::read_to_string(&user_path) {
            Ok(text) => match serde_json::from_str::<Value>(&text) {
                Ok(v) if v.is_object() => v,
                Ok(_) | Err(_) => {
                    warnings.push(format!(
                        "{} is not a JSON object; using the shipped defaults only (fix it, or move \
                         it aside to start fresh)",
                        user_path.display()
                    ));
                    json!({})
                }
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => json!({}),
            Err(e) => return Err(format!("reading {}: {e}", user_path.display())),
        };
        let mut cfg = Config { schema: schema(), shipped, shipped_path, user, user_path, warnings };
        cfg.check_user();
        Ok(cfg)
    }

    /// The built-in registry and the given user overlay (tests, and a known starting point).
    pub fn with_user(user: Value, user_path: PathBuf) -> Config {
        let mut cfg = Config {
            schema: schema(),
            shipped: serde_json::from_str(SHIPPED).expect("config/model_list.json is JSON"),
            shipped_path: None,
            user,
            user_path,
            warnings: Vec::new(),
        };
        cfg.check_user();
        cfg
    }

    pub fn def(&self, key: &str) -> Option<&SettingDef> {
        self.schema.iter().find(|d| d.key == key)
    }

    /// Unknown keys and bad values in the user file are reported, and dropped from resolution.
    /// A shipped default that changed since the user file was written is reported too.
    fn check_user(&mut self) {
        let mut warnings = Vec::new();
        let mut clean = |scope: &str, obj: &mut Map<String, Value>| {
            obj.retain(|k, v| match self.schema.iter().find(|d| &d.key == k) {
                None => {
                    warnings.push(format!("settings.json {scope}: unknown setting {k:?}, ignored"));
                    false
                }
                Some(def) => match def.check(v) {
                    Ok(n) => {
                        *v = n;
                        true
                    }
                    Err(e) => {
                        warnings.push(format!("settings.json {scope}: {e}; ignored"));
                        false
                    }
                },
            });
        };
        if let Some(d) = self.user.get_mut("defaults").and_then(Value::as_object_mut) {
            clean("defaults", d);
        }
        if let Some(families) = self.user.get_mut("settings").and_then(Value::as_object_mut) {
            for (family, sizes) in families.iter_mut() {
                for (size, s) in sizes.as_object_mut().into_iter().flatten() {
                    if let Some(s) = s.as_object_mut() {
                        clean(&format!("settings.{family}.{size}"), s);
                    }
                }
            }
        }
        let shipped_defaults = self.shipped.get("defaults").cloned().unwrap_or(json!({}));
        if let Some(theirs) = self.user.get("defaults_version").and_then(Value::as_str) {
            if theirs != defaults_version(&shipped_defaults) {
                let changed: Vec<String> = self
                    .user
                    .get("defaults")
                    .and_then(Value::as_object)
                    .into_iter()
                    .flat_map(|m| m.keys())
                    .filter(|k| shipped_defaults.get(k.as_str()).is_some())
                    .cloned()
                    .collect();
                if !changed.is_empty() {
                    warnings.push(format!(
                        "the shipped defaults changed since settings.json was written; you override: {}",
                        changed.join(", ")
                    ));
                }
            }
        }
        self.warnings.extend(warnings);
    }

    /// The registry's models with the user's added or changed entries merged in.
    pub fn models(&self) -> Value {
        deep_merge(
            self.shipped.get("models").unwrap_or(&json!({})),
            self.user.get("models").unwrap_or(&json!({})),
        )
    }

    pub fn entry(&self, tag: &str) -> Option<Value> {
        let (family, size) = split_tag(tag);
        self.models().get(family)?.get(size).cloned()
    }

    /// Every (tag, entry), sorted by tag.
    pub fn tags(&self) -> Vec<(String, Value)> {
        let mut out = Vec::new();
        for (family, sizes) in self.models().as_object().into_iter().flatten() {
            for (size, entry) in sizes.as_object().into_iter().flatten() {
                out.push((format!("{family}:{size}"), entry.clone()));
            }
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// The tag whose entry's `name` is this directory's name (case-insensitive).
    pub fn tag_for_dir(&self, dir: &Path) -> Option<String> {
        let name = dir.file_name()?.to_str()?;
        self.tags()
            .into_iter()
            .find(|(_, e)| e.get("name").and_then(Value::as_str).is_some_and(|n| n.eq_ignore_ascii_case(name)))
            .map(|(t, _)| t)
    }

    /// The model a command means: `--config-model TAG`, else `--model-dir DIR` (its registry
    /// tag, if it has one). Errors on an unknown tag.
    pub fn model_ref(
        &self,
        tag: Option<&str>,
        dir: Option<&Path>,
    ) -> Result<Option<String>, String> {
        match (tag, dir) {
            (Some(t), _) if self.entry(t).is_none() => {
                Err(format!("no model {t:?} in the registry (phlegm config list shows them)"))
            }
            (Some(t), _) => Ok(Some(t.to_string())),
            (None, Some(d)) => Ok(self.tag_for_dir(d)),
            (None, None) => Ok(None),
        }
    }

    /// Every setting's value for one model. `cli` maps flags ("--temperature") to their text;
    /// `env` reads an environment variable.
    pub fn resolve_with(
        &self,
        tag: Option<&str>,
        dir: Option<&Path>,
        cli: &HashMap<String, String>,
        env: &dyn Fn(&str) -> Option<String>,
    ) -> Result<Resolved, String> {
        let entry = tag.and_then(|t| self.entry(t));
        let user_model = tag.and_then(|t| {
            let (family, size) = split_tag(t);
            self.user.get("settings")?.get(family)?.get(size).cloned()
        });
        let gen_config: Option<Value> = dir
            .map(|d| d.join("generation_config.json"))
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|t| serde_json::from_str(&t).ok());
        let mut values = BTreeMap::new();
        for def in &self.schema {
            let key = def.key.as_str();
            let per_model = def.scope == Scope::PerModel;
            let checked = |v: &Value, what: &str| def.check(v).map_err(|e| format!("{what}: {e}"));
            let found: Option<(Value, Source)> = if let Some(text) =
                def.cli_flag.as_ref().and_then(|f| cli.get(f))
            {
                Some((def.parse(text)?, Source::Cli))
            } else if let Some(text) = def.env_var.as_ref().and_then(|v| env(v)).filter(|t| !t.is_empty()) {
                let name = def.env_var.as_deref().unwrap_or_default();
                Some((def.parse(&text).map_err(|e| format!("{name}: {e}"))?, Source::Env))
            } else if let Some(v) = user_model.as_ref().filter(|_| per_model).and_then(|m| m.get(key)) {
                Some((v.clone(), Source::User))
            } else if let Some(v) = self.user.get("defaults").and_then(|d| d.get(key)) {
                Some((v.clone(), Source::User))
            } else if let Some(v) = self.shipped.get("defaults").and_then(|d| d.get(key)) {
                Some((checked(v, "shipped defaults")?, Source::Shipped))
            } else if let Some(v) = entry
                .as_ref()
                .filter(|_| per_model)
                .zip(def.registry_key.as_deref())
                .and_then(|(e, rk)| pointer(e, rk))
            {
                Some((checked(v, &format!("registry entry {}", tag.unwrap_or_default()))?, Source::Shipped))
            } else if let Some(v) = gen_config
                .as_ref()
                .filter(|_| per_model)
                .zip(def.generation_config.as_deref())
                .and_then(|(g, f)| generation_config_value(g, f))
            {
                Some((checked(&v, "generation_config.json")?, Source::GenerationConfig))
            } else {
                def.default_value
                    .as_ref()
                    .map(|v| checked(v, "settings.json").map(|v| (v, Source::Shipped)))
                    .transpose()?
            };
            let found = found.ok_or_else(|| format!("setting {key} has no value and no default"))?;
            values.insert(key.to_string(), found);
        }
        Ok(Resolved { tag: tag.map(str::to_string), values })
    }

    /// `resolve_with` reading the process environment.
    pub fn resolve(
        &self,
        tag: Option<&str>,
        dir: Option<&Path>,
        cli: &HashMap<String, String>,
    ) -> Result<Resolved, String> {
        self.resolve_with(tag, dir, cli, &|k| std::env::var(k).ok())
    }

    /// Set one user override: a model's (`tag`) or every model's (`None`).
    pub fn set_user(&mut self, tag: Option<&str>, key: &str, value: Value) -> Result<(), String> {
        let def = self.def(key).ok_or_else(|| format!("unknown setting {key:?}"))?.clone();
        let value = def.check(&value)?;
        let root = self.user_object();
        match tag {
            None => {
                obj(root, "defaults").insert(key.into(), value);
            }
            Some(_) if def.scope == Scope::Global => {
                return Err(format!("{key} applies to every model; set it without a model"))
            }
            Some(t) => {
                let (family, size) = split_tag(t);
                let at = obj(obj(obj(root, "settings"), family), size);
                at.insert(key.into(), value);
            }
        }
        Ok(())
    }

    /// Remove one user override; false if there was none.
    pub fn unset_user(&mut self, tag: Option<&str>, key: &str) -> Result<bool, String> {
        self.def(key).ok_or_else(|| format!("unknown setting {key:?}"))?;
        let removed = match tag {
            None => self
                .user
                .get_mut("defaults")
                .and_then(Value::as_object_mut)
                .and_then(|d| d.remove(key))
                .is_some(),
            Some(t) => {
                let (family, size) = split_tag(t);
                let removed = self
                    .user
                    .get("settings")
                    .and_then(|s| s.get(family))
                    .and_then(|s| s.get(size))
                    .is_some_and(|m| m.get(key).is_some());
                if removed {
                    let sizes = self.user["settings"][family].as_object_mut().unwrap();
                    sizes[size].as_object_mut().unwrap().remove(key);
                    if sizes[size].as_object().is_some_and(Map::is_empty) {
                        sizes.remove(size);
                    }
                    if sizes.is_empty() {
                        self.user["settings"].as_object_mut().unwrap().remove(family);
                    }
                }
                removed
            }
        };
        Ok(removed)
    }

    fn user_object(&mut self) -> &mut Map<String, Value> {
        if !self.user.is_object() {
            self.user = json!({});
        }
        self.user.as_object_mut().unwrap()
    }

    /// Write the user file atomically (a temporary file in the same directory, then a rename),
    /// stamped with the format version and the shipped defaults it was written against.
    pub fn save_user(&mut self) -> Result<(), String> {
        let version = defaults_version(self.shipped.get("defaults").unwrap_or(&json!({})));
        let root = self.user_object();
        root.insert("version".into(), json!(1));
        root.insert("defaults_version".into(), json!(version));
        let path = self.user_path.clone();
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("creating {}: {e}", dir.display()))?;
        }
        let tmp = path.with_extension(format!("json.tmp{}", std::process::id()));
        let text = serde_json::to_string_pretty(&self.user).map_err(|e| e.to_string())? + "\n";
        std::fs::write(&tmp, text).map_err(|e| format!("writing {}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, &path).map_err(|e| format!("replacing {}: {e}", path.display()))
    }
}

/// The object at `key` (created, or replacing a non-object).
fn obj<'a>(m: &'a mut Map<String, Value>, key: &str) -> &'a mut Map<String, Value> {
    let slot = m.entry(key.to_string()).or_insert_with(|| json!({}));
    if !slot.is_object() {
        *slot = json!({});
    }
    slot.as_object_mut().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_env(_: &str) -> Option<String> {
        None
    }

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("phlegm-settings-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn the_shipped_files_are_consistent() {
        let cfg = Config::with_user(json!({}), PathBuf::from("/nonexistent/settings.json"));
        let mut flags = std::collections::HashSet::new();
        for def in &cfg.schema {
            let d = def.default_value.as_ref().expect("every setting has a built-in default");
            def.check(d).unwrap();
            if let Some(f) = &def.cli_flag {
                assert!(flags.insert(f.clone()), "{f} is two settings' flag");
            }
            if def.kind == Kind::Enum {
                assert!(!def.options.is_empty());
            }
        }
        let defaults = &cfg.shipped["defaults"];
        assert_eq!(cfg.shipped["defaults_version"], json!(defaults_version(defaults)),
                   "model_list.json's defaults_version is stale");
        for (k, v) in defaults.as_object().unwrap() {
            cfg.def(k).expect("a shipped default names a setting").check(v).unwrap();
        }
        // Every registry entry's values pass their setting's checks and its name is unique.
        let mut names = std::collections::HashSet::new();
        for (tag, entry) in cfg.tags() {
            assert!(names.insert(entry["name"].as_str().unwrap().to_ascii_lowercase()), "{tag}");
            for def in &cfg.schema {
                if let Some(v) = def.registry_key.as_deref().and_then(|rk| pointer(&entry, rk)) {
                    def.check(v).unwrap_or_else(|e| panic!("{tag}: {e}"));
                }
            }
            cfg.resolve_with(Some(tag.as_str()), None, &HashMap::new(), &no_env).unwrap();
        }
        assert!(cfg.warnings.is_empty(), "{:?}", cfg.warnings);
    }

    #[test]
    fn defaults_version_matches_python_canonical_json() {
        // python3 -c 'import json,hashlib; print(hashlib.sha256(json.dumps({"b":1,"a":"x"},
        //   sort_keys=True, separators=(",",":")).encode()).hexdigest()[:16])'
        assert_eq!(canonical(&json!({"b": 1, "a": "x"})), r#"{"a":"x","b":1}"#);
        assert_eq!(defaults_version(&json!({"b": 1, "a": "x"})), "cdab067e9f3beb32");
    }

    #[test]
    fn precedence_runs_from_built_in_to_the_command_line() {
        let dir = scratch("prec").join("LFM3-1.7B-NPU2");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("generation_config.json"),
                       r#"{"do_sample": true, "temperature": 0.7, "top_k": 9}"#).unwrap();
        let mut cfg = Config::with_user(json!({}), dir.join("settings.json"));
        let tag = cfg.tag_for_dir(&dir);
        assert_eq!(tag.as_deref(), Some("lfm3:1.7b"));
        let r = |cfg: &Config, cli: &[(&str, &str)], env: &dyn Fn(&str) -> Option<String>| {
            let cli: HashMap<String, String> =
                cli.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
            cfg.resolve_with(tag.as_deref(), Some(&dir), &cli, env).unwrap()
        };
        // The registry entry beats the model dir's generation_config, which beats the built-in.
        let base = r(&cfg, &[], &no_env);
        assert_eq!(base.values["temperature"], (json!(0.1), Source::Shipped));
        assert_eq!(base.values["top_p"], (json!(1.0), Source::GenerationConfig));
        assert_eq!(base.values["max_tokens"], (json!(0), Source::Shipped));
        assert_eq!(base.values["ctx_len"], (json!(4096), Source::Shipped));
        // User defaults, then the user's per-model setting, then env, then the flag.
        cfg.set_user(None, "temperature", json!(0.5)).unwrap();
        assert_eq!(r(&cfg, &[], &no_env).values["temperature"], (json!(0.5), Source::User));
        cfg.set_user(tag.as_deref(), "temperature", json!(0.3)).unwrap();
        assert_eq!(r(&cfg, &[], &no_env).values["temperature"], (json!(0.3), Source::User));
        cfg.set_user(None, "max_tokens", json!(100)).unwrap();
        let env = |k: &str| (k == "PHLEGM_MAX_TOKENS").then(|| "200".to_string());
        assert_eq!(r(&cfg, &[], &env).values["max_tokens"], (json!(200), Source::Env));
        assert_eq!(r(&cfg, &[("--max-tokens", "300")], &env).values["max_tokens"], (json!(300), Source::Cli));
        assert_eq!(r(&cfg, &[("--temperature", "0")], &no_env).values["temperature"], (json!(0.0), Source::Cli));
        // A bad flag value is an error naming the setting, not a silent default.
        let cli = HashMap::from([("--top-p".to_string(), "0".to_string())]);
        assert!(cfg.resolve_with(None, None, &cli, &no_env).unwrap_err().contains("top_p"));
        // Unset falls back a level; the file round-trips.
        assert!(cfg.unset_user(tag.as_deref(), "temperature").unwrap());
        assert!(!cfg.unset_user(tag.as_deref(), "temperature").unwrap());
        assert_eq!(cfg.user.get("settings").and_then(|s| s.get("lfm3")), None);
        cfg.save_user().unwrap();
        let back: Value = serde_json::from_str(&std::fs::read_to_string(&cfg.user_path).unwrap()).unwrap();
        assert_eq!(back["defaults"], json!({"temperature": 0.5, "max_tokens": 100}));
        assert_eq!(back["defaults_version"], cfg.shipped["defaults_version"]);
        std::fs::remove_dir_all(dir.parent().unwrap()).unwrap();
    }

    #[test]
    fn a_model_outside_the_registry_takes_its_generation_config_with_hf_semantics() {
        let dir = scratch("gc");
        let cfg = Config::with_user(json!({}), dir.join("settings.json"));
        assert_eq!(cfg.tag_for_dir(&dir), None);
        let get = |gc: &str| {
            std::fs::write(dir.join("generation_config.json"), gc).unwrap();
            let r = cfg.resolve_with(None, Some(&dir), &HashMap::new(), &no_env).unwrap();
            ["temperature", "top_k", "top_p", "repetition_penalty"].map(|k| r.values[k].0.clone())
        };
        // do_sample alone: HF's defaults.
        assert_eq!(get(r#"{"do_sample": true}"#), [json!(1.0), json!(50), json!(1.0), json!(1.0)]);
        // No do_sample: greedy, whatever temperature says; the repetition penalty still applies.
        assert_eq!(get(r#"{"temperature": 0.6, "top_k": 5, "repetition_penalty": 1.2}"#),
                   [json!(0.0), json!(0), json!(1.0), json!(1.2)]);
        assert_eq!(get(r#"{"do_sample": true, "temperature": 0.6, "top_p": 0.9}"#),
                   [json!(0.6), json!(50), json!(0.9), json!(1.0)]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_user_file_overlays_models_and_bad_entries_are_reported_not_fatal() {
        let user = json!({
            "defaults": {"temperature": -1, "nonsense": 3, "top_k": 7},
            "settings": {"lfm3": {"1.7b": {"top_p": 0.9}}},
            "models": {"mine": {"7b": {"name": "My-7B-NPU2", "sampling": {"temperature": 0.2}}},
                       "lfm3": {"1.7b": {"sampling": {"top_k": 40}}}},
            "defaults_version": "0000000000000000"
        });
        let cfg = Config::with_user(user, PathBuf::from("/nonexistent/settings.json"));
        assert_eq!(cfg.warnings.len(), 2, "{:?}", cfg.warnings);   // temperature -1, nonsense
        assert_eq!(cfg.tag_for_dir(Path::new("/m/my-7b-npu2")).as_deref(), Some("mine:7b"));
        let r = cfg.resolve_with(Some("lfm3:1.7b"), None, &HashMap::new(), &no_env).unwrap();
        assert_eq!(r.values["top_k"], (json!(7), Source::User));         // user defaults beat the entry
        assert_eq!(r.values["top_p"], (json!(0.9), Source::User));
        assert_eq!(r.values["temperature"], (json!(0.1), Source::Shipped));
        assert_eq!(cfg.entry("lfm3:1.7b").unwrap()["sampling"]["top_k"], json!(40));
        let mine = cfg.resolve_with(Some("mine:7b"), None, &HashMap::new(), &no_env).unwrap();
        assert_eq!(mine.f32("temperature"), 0.2);
        assert!(cfg.model_ref(Some("nope:1b"), None).is_err());
    }

    #[test]
    fn deep_merge_merges_objects_replaces_the_rest_and_ignores_nulls() {
        let a = json!({"x": {"y": 1, "z": [1, 2]}, "k": 1});
        let b = json!({"x": {"z": [3], "w": 2}, "k": null});
        assert_eq!(deep_merge(&a, &b), json!({"x": {"y": 1, "z": [3], "w": 2}, "k": 1}));
        assert_eq!(split_tag("lfm3:1.7b"), ("lfm3", "1.7b"));
        assert_eq!(split_tag("solo"), ("solo", ""));
    }
}
