//! Reasoning effort (SERVER-REASONING-EFFORT): a request's or the model's thinking choice,
//! translated into whatever its chat template reads.
//!
//! Clients speak OpenAI's `reasoning_effort` (none ... xhigh) or a plain on/off switch
//! (`chat_template_kwargs.enable_thinking`, OpenFlowLM's `think`). Templates differ: Qwen, LFM
//! and Gemma read `enable_thinking`; GPT-OSS, Nex and Hy-Memory read `reasoning_effort`, each with
//! its own values (Hy-Memory: no_think / low / high; Nex: none / medium / high). A registry
//! entry's `details.reasoning_efforts` maps OpenAI's names to the template's.

use serde_json::{Map, Value};

/// OpenAI's reasoning_effort values, least to most.
pub const LADDER: [&str; 6] = ["none", "minimal", "low", "medium", "high", "xhigh"];

/// What the model should do before it answers.
#[derive(Clone, Debug, PartialEq)]
pub enum Thinking {
    Off,
    /// The model's usual thinking (OpenAI's default effort, medium).
    On,
    /// A named effort other than none.
    Effort(String),
}

impl Thinking {
    /// One of OpenAI's names; "none" is Off.
    pub fn effort(name: &str) -> Result<Thinking, String> {
        match name {
            "none" => Ok(Thinking::Off),
            _ if LADDER.contains(&name) => Ok(Thinking::Effort(name.to_string())),
            _ => Err(format!("reasoning_effort must be one of {}", LADDER.join(", "))),
        }
    }

    /// The model's settings: `reasoning_effort`, where "default" defers to `think`.
    pub fn from_settings(effort: &str, think: bool) -> Result<Thinking, String> {
        match effort {
            "" | "default" => Ok(if think { Thinking::On } else { Thinking::Off }),
            name => Thinking::effort(name),
        }
    }

    /// A request's choice over the model's: its effort, else its on/off switch. Switching on a
    /// model whose settings name an effort keeps that effort.
    pub fn for_request(&self, effort: Option<&str>, think: Option<bool>) -> Result<Thinking, String> {
        if let Some(name) = effort {
            return Thinking::effort(name);
        }
        Ok(match (think, self) {
            (Some(false), _) => Thinking::Off,
            (Some(true), Thinking::Effort(e)) => Thinking::Effort(e.clone()),
            (Some(true), _) => Thinking::On,
            (None, model) => model.clone(),
        })
    }

    fn name(&self) -> &str {
        match self {
            Thinking::Off => "none",
            Thinking::On => "medium",
            Thinking::Effort(e) => e,
        }
    }
}

/// A template's own effort values by OpenAI name (`details.reasoning_efforts`).
#[derive(Clone, Debug, PartialEq)]
pub struct Levels(Vec<(usize, String)>);

impl Levels {
    /// The registry entry's table, if it has one.
    pub fn from_entry(entry: Option<&Value>) -> Result<Option<Levels>, String> {
        let Some(table) = entry.and_then(|e| e.pointer("/details/reasoning_efforts")) else {
            return Ok(None);
        };
        let bad = || "details.reasoning_efforts: expected an object of OpenAI effort names to strings".to_string();
        let mut levels = Vec::new();
        for (name, value) in table.as_object().ok_or_else(bad)? {
            let at = LADDER.iter().position(|l| l == name).ok_or_else(bad)?;
            levels.push((at, value.as_str().ok_or_else(bad)?.to_string()));
        }
        if levels.is_empty() {
            return Err(bad());
        }
        levels.sort();
        Ok(Some(Levels(levels)))
    }

    /// The template value nearest the named effort; between two, the higher.
    pub fn pick(&self, name: &str) -> &str {
        let want = LADDER.iter().position(|l| *l == name).unwrap_or(3) as isize;
        let mut best = &self.0[0];
        for level in &self.0 {
            if (level.0 as isize - want).abs() <= (best.0 as isize - want).abs() {
                best = level;
            }
        }
        &best.1
    }
}

/// Which switches a chat template reads.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TemplateVars {
    pub enable_thinking: bool,
    pub reasoning_effort: bool,
}

impl TemplateVars {
    pub fn of(template_src: &str) -> TemplateVars {
        TemplateVars {
            enable_thinking: template_src.contains("enable_thinking"),
            reasoning_effort: template_src.contains("reasoning_effort"),
        }
    }
}

/// The template variables for a choice. `enable_thinking` follows on/off. `reasoning_effort`
/// takes the level table's nearest value; without a table, a named effort passes through as
/// written, and Off becomes "none" only where the template has no `enable_thinking` to use.
pub fn template_kwargs(thinking: &Thinking, vars: TemplateVars, levels: Option<&Levels>) -> Map<String, Value> {
    let mut kwargs = Map::new();
    if vars.enable_thinking {
        kwargs.insert("enable_thinking".into(), Value::Bool(*thinking != Thinking::Off));
    }
    if vars.reasoning_effort {
        let value = match (levels, thinking) {
            (Some(levels), t) => Some(levels.pick(t.name()).to_string()),
            (None, Thinking::Effort(e)) => Some(e.clone()),
            (None, Thinking::Off) if !vars.enable_thinking => Some("none".to_string()),
            (None, _) => None,
        };
        if let Some(v) = value {
            kwargs.insert("reasoning_effort".into(), Value::String(v));
        }
    }
    kwargs
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn levels(table: Value) -> Levels {
        Levels::from_entry(Some(&json!({"details": {"reasoning_efforts": table}}))).unwrap().unwrap()
    }

    const EFFORT: TemplateVars = TemplateVars { enable_thinking: false, reasoning_effort: true };
    const SWITCH: TemplateVars = TemplateVars { enable_thinking: true, reasoning_effort: false };
    const BOTH: TemplateVars = TemplateVars { enable_thinking: true, reasoning_effort: true };

    fn effort(t: &Thinking, vars: TemplateVars, l: Option<&Levels>) -> Value {
        Value::Object(template_kwargs(t, vars, l))
    }

    #[test]
    fn hy_memory_and_nex_get_their_own_values() {
        let hy = levels(json!({"none": "no_think", "low": "low", "high": "high"}));
        let nex = levels(json!({"none": "none", "medium": "medium", "high": "high"}));
        let cases = [
            ("none", "no_think", "none"),
            ("minimal", "low", "none"), // Hy-Memory: no_think and low equally near; the higher wins
            ("low", "low", "medium"),   // Nex: none and medium equally near
            ("medium", "high", "medium"), // Hy-Memory: low and high equally near
            ("high", "high", "high"),
            ("xhigh", "high", "high"),
        ];
        for (name, h, n) in cases {
            let t = Thinking::effort(name).unwrap();
            assert_eq!(effort(&t, EFFORT, Some(&hy)), json!({"reasoning_effort": h}), "{name}");
            assert_eq!(effort(&t, EFFORT, Some(&nex)), json!({"reasoning_effort": n}), "{name}");
        }
        assert_eq!(effort(&Thinking::On, EFFORT, Some(&hy)), json!({"reasoning_effort": "high"}));
        assert_eq!(effort(&Thinking::Off, EFFORT, Some(&nex)), json!({"reasoning_effort": "none"}));
    }

    #[test]
    fn a_model_that_cannot_stop_thinking_gets_its_lowest_effort() {
        let gpt_oss = levels(json!({"low": "low", "medium": "medium", "high": "high"}));
        assert_eq!(effort(&Thinking::Off, EFFORT, Some(&gpt_oss)), json!({"reasoning_effort": "low"}));
        assert_eq!(effort(&Thinking::On, EFFORT, Some(&gpt_oss)), json!({"reasoning_effort": "medium"}));
    }

    #[test]
    fn switch_templates_and_templates_without_a_table() {
        let high = Thinking::effort("high").unwrap();
        assert_eq!(effort(&Thinking::Off, SWITCH, None), json!({"enable_thinking": false}));
        assert_eq!(effort(&high, SWITCH, None), json!({"enable_thinking": true}));
        // Granite reads both: on/off through enable_thinking, the effort as written.
        assert_eq!(effort(&Thinking::Off, BOTH, None), json!({"enable_thinking": false}));
        assert_eq!(effort(&high, BOTH, None), json!({"enable_thinking": true, "reasoning_effort": "high"}));
        assert_eq!(effort(&Thinking::On, BOTH, None), json!({"enable_thinking": true}));
        // Effort-only, no table: as written; on leaves the template's default.
        assert_eq!(effort(&high, EFFORT, None), json!({"reasoning_effort": "high"}));
        assert_eq!(effort(&Thinking::Off, EFFORT, None), json!({"reasoning_effort": "none"}));
        assert_eq!(effort(&Thinking::On, EFFORT, None), json!({}));
        assert_eq!(effort(&high, TemplateVars::default(), None), json!({}));
    }

    #[test]
    fn requests_override_settings() {
        let low = Thinking::from_settings("low", true).unwrap();
        assert_eq!(low, Thinking::Effort("low".into()));
        assert_eq!(Thinking::from_settings("default", false).unwrap(), Thinking::Off);
        assert_eq!(Thinking::from_settings("none", true).unwrap(), Thinking::Off);
        assert_eq!(low.for_request(None, None).unwrap(), low);
        assert_eq!(low.for_request(None, Some(true)).unwrap(), low);
        assert_eq!(low.for_request(None, Some(false)).unwrap(), Thinking::Off);
        assert_eq!(Thinking::Off.for_request(None, Some(true)).unwrap(), Thinking::On);
        assert_eq!(low.for_request(Some("high"), Some(false)).unwrap(), Thinking::Effort("high".into()));
        assert!(low.for_request(Some("max"), None).is_err());
    }

    #[test]
    fn bad_tables_are_errors() {
        for table in [json!({"medium": 3}), json!({"maximum": "x"}), json!({}), json!("high")] {
            let entry = json!({"details": {"reasoning_efforts": table}});
            assert!(Levels::from_entry(Some(&entry)).is_err(), "{entry}");
        }
        assert_eq!(Levels::from_entry(Some(&json!({"details": {}}))).unwrap(), None);
    }
}
