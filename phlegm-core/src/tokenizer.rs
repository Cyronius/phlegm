//! Hugging Face tokenizer and chat template for the selected open-kernel model.
//! The token IDs and end tokens are read from the model's tokenizer files;
//! templates may be an external Jinja file or embedded in tokenizer_config.json.

use crate::reasoning_effort::TemplateVars;
use minijinja::Environment;
use serde_json::{Map, Value};
use std::path::Path;
use tokenizers::Tokenizer as HfTokenizer;

#[path = "gguf_tokenizer.rs"]
mod gguf_tokenizer;

/// Padded logit width emitted by the lm_head (tokenizer.json vocab is 248070).
pub const MODEL_VOCAB_SIZE: usize = 248320;

pub const ENDOFTEXT: u32 = 248044; // <|endoftext|> (also pad)
pub const IM_START: u32 = 248045; // <|im_start|>
pub const IM_END: u32 = 248046; // <|im_end|> (assistant-turn stop)
pub const OBJECT_REF_END: u32 = 248048;

/// Any of these ends generation (tokenizer_config.json: eos_token_id).
pub const EOS_TOKEN_IDS: [u32; 3] = [ENDOFTEXT, IM_END, OBJECT_REF_END];

/// `strftime` over a UTC time for the chat templates (`strftime_now`): %Y %m %d %H %M %S %b %B
/// %a %A %y %j %e %%; anything else is copied through as written.
pub(crate) fn strftime_utc(fmt: &str, unix_secs: i64) -> String {
    let days = unix_secs.div_euclid(86_400);
    let tod = unix_secs.rem_euclid(86_400);
    // civil_from_days (Howard Hinnant): proleptic Gregorian date of a day count
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    const MONTHS: [&str; 12] = ["January", "February", "March", "April", "May", "June", "July",
                                "August", "September", "October", "November", "December"];
    const DAYS: [&str; 7] = ["Thursday", "Friday", "Saturday", "Sunday", "Monday", "Tuesday", "Wednesday"];
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let first_of_month = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334][(m - 1) as usize];
    let yday = first_of_month + d + i64::from(leap && m > 2);
    let wd = DAYS[days.rem_euclid(7) as usize];
    let mn = MONTHS[(m - 1) as usize];
    let mut out = String::new();
    let mut chars = fmt.chars();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('Y') => out += &y.to_string(),
            Some('y') => out += &format!("{:02}", y.rem_euclid(100)),
            Some('m') => out += &format!("{m:02}"),
            Some('d') => out += &format!("{d:02}"),
            Some('e') => out += &format!("{d:2}"),
            Some('j') => out += &format!("{yday:03}"),
            Some('H') => out += &format!("{:02}", tod / 3600),
            Some('M') => out += &format!("{:02}", tod / 60 % 60),
            Some('S') => out += &format!("{:02}", tod % 60),
            Some('b') => out += &mn[..3],
            Some('B') => out += mn,
            Some('a') => out += &wd[..3],
            Some('A') => out += wd,
            Some('%') => out.push('%'),
            Some(other) => {
                out.push('%');
                out.push(other);
            }
            None => out.push('%'),
        }
    }
    out
}

/// GPT-OSS's harmony output, `<|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final
/// <|message|>...<|return|>`, split into (analysis, final). None when the text carries no final
/// channel marker (another family, or a harmony turn cut short before its answer).
pub fn harmony_split(text: &str) -> Option<(String, String)> {
    const FINAL: &str = "<|channel|>final<|message|>";
    const ANALYSIS: &str = "<|channel|>analysis<|message|>";
    let at = text.rfind(FINAL)?;
    let strip = |t: &str| {
        let mut t = t;
        for end in ["<|return|>", "<|end|>", "<|call|>", "<|endoftext|>"] {
            if let Some(i) = t.find(end) {
                t = &t[..i];
            }
        }
        t.trim().to_owned()
    };
    let analysis = text[..at].find(ANALYSIS).map(|i| strip(&text[i + ANALYSIS.len()..at])).unwrap_or_default();
    Some((analysis, strip(&text[at + FINAL.len()..])))
}

/// A harmony reply so far as (reasoning, answer): the analysis message's text and the final
/// message's, each without its markers and cut at its end marker. The streaming server re-derives
/// both after every token and sends each one's new suffix (`reasoning_content` / `content`).
pub fn harmony_progress(text: &str) -> (String, String) {
    const FINAL: &str = "<|channel|>final<|message|>";
    const ANALYSIS: &str = "<|channel|>analysis<|message|>";
    let cut = |t: &str| {
        let mut t = t;
        for end in ["<|end|>", "<|return|>", "<|call|>", "<|start|>", "<|endoftext|>"] {
            if let Some(i) = t.find(end) {
                t = &t[..i];
            }
        }
        t.to_owned()
    };
    let fin = text.rfind(FINAL);
    let reasoning = text
        .find(ANALYSIS)
        .filter(|i| fin.map_or(true, |f| *i < f))
        .map(|i| cut(&text[i + ANALYSIS.len()..fin.unwrap_or(text.len())]))
        .unwrap_or_default();
    let answer = fin.map(|f| cut(&text[f + FINAL.len()..])).unwrap_or_default();
    (reasoning, answer)
}

/// How a model's chat template marks reasoning apart from the answer, read from the model's own
/// files at load (`Tokenizer::load`), never from its name.
#[derive(Clone, Debug, PartialEq)]
pub enum ReasoningFormat {
    /// GPT-OSS's harmony channels (`harmony_progress` / `harmony_split`).
    Harmony,
    /// A reasoning block between two markers ahead of the answer: Gemma 4's
    /// `<|channel>thought\n ... <channel|>` (its tokenizer_config `response_template`), Qwen3.5's
    /// `<think> ... </think>` (its chat template).
    Tagged { open: String, close: String },
}

/// One reply's reasoning/answer splitter: the format, plus whether the rendered prompt already
/// opened the reasoning block (Qwen's generation prompt ends in `<think>\n`, so its reply starts
/// inside the block and never repeats the opener).
#[derive(Clone, Debug, PartialEq)]
pub struct Reasoning {
    pub format: ReasoningFormat,
    pub opened_by_prompt: bool,
}

impl Reasoning {
    pub fn for_prompt(format: ReasoningFormat, prompt: &str) -> Reasoning {
        let opened_by_prompt = match &format {
            ReasoningFormat::Tagged { open, .. } => prompt.trim_end().ends_with(open.trim_end()),
            ReasoningFormat::Harmony => false,
        };
        Reasoning { format, opened_by_prompt }
    }

    /// The reply so far as (reasoning, answer), for streaming: each side only grows as text is
    /// appended, and a partly decoded marker is held back until it resolves.
    pub fn progress(&self, text: &str) -> (String, String) {
        self.derive(text, true)
    }

    /// `progress` once the reply is complete: nothing more is held back.
    pub fn progress_done(&self, text: &str) -> (String, String) {
        self.derive(text, false)
    }

    /// A finished reply as (reasoning, answer), trimmed. A reply cut short inside its reasoning
    /// has an empty answer.
    pub fn split(&self, text: &str) -> (String, String) {
        if self.format == ReasoningFormat::Harmony {
            if let Some(parts) = harmony_split(text) {
                return parts;
            }
        }
        let (r, a) = self.derive(text, false);
        (r.trim().to_owned(), a.trim().to_owned())
    }

    fn derive(&self, text: &str, partial: bool) -> (String, String) {
        let (open, close) = match &self.format {
            ReasoningFormat::Harmony => return harmony_progress(text),
            ReasoningFormat::Tagged { open, close } => (open.trim_end(), close.as_str()),
        };
        // Drop a trailing piece of text that may be the start of `marker`.
        let hold = |t: &'_ str, marker: &str| -> usize {
            if !partial {
                return t.len();
            }
            (1..marker.len().min(t.len() + 1))
                .rev()
                .find(|&n| t.is_char_boundary(t.len() - n) && marker.starts_with(&t[t.len() - n..]))
                .map_or(t.len(), |n| t.len() - n)
        };
        let lead = text.trim_start();
        let inside = if self.opened_by_prompt {
            text
        } else if let Some(rest) = lead.strip_prefix(open) {
            rest
        } else if partial && open.starts_with(lead) {
            return (String::new(), String::new());
        } else {
            return (String::new(), lead.to_owned());
        };
        match inside.find(close) {
            Some(i) => (
                inside[..i].trim_start().to_owned(),
                inside[i + close.len()..].trim_start().to_owned(),
            ),
            None => (inside[..hold(inside, close)].trim_start().to_owned(), String::new()),
        }
    }
}

#[derive(Default, serde::Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokenizers::{
        decoders::byte_fallback::ByteFallback, models::bpe::BPE,
        processors::template::TemplateProcessing, AddedToken,
    };

    static NEXT: AtomicUsize = AtomicUsize::new(0);

    pub(crate) fn fixture(add_bos: bool, object_tokens: bool) -> (std::path::PathBuf, Tokenizer) {
        let dir = std::env::temp_dir().join(format!(
            "phlegm-tokenizer-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let vocab: tokenizers::models::bpe::Vocab = [
            "<bos>", "<eos>", "A", "<0xF0>", "<0x9F>", "<0x99>", "<0x82>",
        ]
        .into_iter()
        .enumerate()
        .map(|(id, text)| (text.to_owned(), id as u32))
        .collect();
        let mut tk = HfTokenizer::new(
            BPE::builder()
                .vocab_and_merges(vocab, vec![])
                .byte_fallback(true)
                .build()
                .unwrap(),
        );
        tk.add_special_tokens([
            AddedToken::from("<bos>", true),
            AddedToken::from("<eos>", true),
        ])
        .unwrap();
        tk.with_decoder(Some(ByteFallback::default()));
        tk.with_post_processor(Some(
            TemplateProcessing::builder()
                .try_single("<bos> $A")
                .unwrap()
                .special_tokens(vec![("<bos>", 0)])
                .build()
                .unwrap(),
        ));
        tk.save(dir.join("tokenizer.json"), false).unwrap();
        let mut config = json!({"add_bos_token":add_bos,
            "chat_template":"{{ bos_token }}{{ messages[0].content }}{{ eos_token }}"});
        if object_tokens {
            std::fs::write(
                dir.join("special_tokens_map.json"),
                json!({"bos_token":{"content":"<bos>","special":true},
                       "eos_token":{"content":"<eos>","special":true}})
                .to_string(),
            )
            .unwrap();
        } else {
            config["bos_token"] = json!("<bos>");
            config["eos_token"] = json!("<eos>");
        }
        std::fs::write(dir.join("tokenizer_config.json"), config.to_string()).unwrap();
        let tokenizer = Tokenizer::load(&dir).unwrap();
        (dir, tokenizer)
    }

    #[test]
    fn template_special_tokens_render_without_double_bos() {
        for add_bos in [false, true] {
            for object_tokens in [false, true] {
                let (dir, tokenizer) = fixture(add_bos, object_tokens);
                let text = tokenizer
                    .apply_chat_template(
                        &[ChatMessage {
                            role: "user".into(),
                            content: "A".into(),
                            ..Default::default()
                        }],
                        true,
                        true,
                    )
                    .unwrap();
                assert_eq!(text, "<bos>A<eos>");
                assert_eq!(tokenizer.encode(&text).unwrap(), [0, 2, 1]);
                assert_eq!(
                    tokenizer.encode("A").unwrap(),
                    if add_bos { vec![0, 2] } else { vec![2] }
                );
                assert!(tokenizer.is_eos(1));
                std::fs::remove_dir_all(dir).unwrap();
            }
        }
    }

    #[test]
    fn none_tools_is_not_iterable() {
        // NVIDIA Nemotron-H's guard, reduced: only `tools is defined` resets it, so with tools =
        // none the template relies on jinja2's `none is iterable` being false.
        let (dir, _) = fixture(false, false);
        std::fs::write(
            dir.join("chat_template.jinja"),
            "{%- if not tools is defined %}{%- set tools = [] %}{%- endif %}\
             {%- if tools is iterable and tools | length > 0 %}T{%- endif %}\
             {{ messages[0].content }}|{{ [] is iterable }}{{ 'ab' is iterable }}{{ 3 is iterable }}",
        )
        .unwrap();
        let tokenizer = Tokenizer::load(&dir).unwrap();
        let msg = [ChatMessage {
            role: "user".into(),
            content: "A".into(),
            ..Default::default()
        }];
        assert_eq!(
            tokenizer.render_chat(&msg, true, &Map::new()).unwrap(),
            "A|TrueTrueFalse"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn effort_templates_get_their_own_values() {
        use crate::reasoning_effort::{template_kwargs, Levels, TemplateVars, Thinking};
        let (dir, _) = fixture(false, false);
        // Hy-Memory's generation prompt, reduced: no_think closes the block, low/high open it,
        // and anything else falls back to no_think.
        std::fs::write(
            dir.join("chat_template.jinja"),
            "{%- if reasoning_effort is not defined or reasoning_effort not in ['high', 'low', 'no_think'] %}\
             {%- set reasoning_effort = 'no_think' %}{%- endif %}\
             {{ messages[0].content }}|{{ reasoning_effort }}{{ '|' ~ preserve_thinking if preserve_thinking is defined }}",
        )
        .unwrap();
        let tokenizer = Tokenizer::load(&dir).unwrap();
        assert_eq!(tokenizer.template_vars(), TemplateVars { enable_thinking: false, reasoning_effort: true });
        let levels = Levels::from_entry(Some(&serde_json::json!(
            {"details": {"reasoning_efforts": {"none": "no_think", "low": "low", "high": "high"}}}
        )))
        .unwrap();
        let msg = [ChatMessage {
            role: "user".into(),
            content: "A".into(),
            ..Default::default()
        }];
        let render = |t: Thinking| {
            let kwargs = template_kwargs(&t, tokenizer.template_vars(), levels.as_ref());
            tokenizer.render_chat(&msg, true, &kwargs).unwrap()
        };
        assert_eq!(render(Thinking::Off), "A|no_think");
        assert_eq!(render(Thinking::On), "A|high");
        assert_eq!(render(Thinking::effort("minimal").unwrap()), "A|low");
        // Without the table the OpenAI name goes through, and this template ignores "medium".
        let kwargs = template_kwargs(&Thinking::effort("medium").unwrap(), tokenizer.template_vars(), None);
        assert_eq!(tokenizer.render_chat(&msg, true, &kwargs).unwrap(), "A|no_think");
        // Other template variables pass through.
        let mut kwargs = Map::new();
        kwargs.insert("preserve_thinking".into(), Value::Bool(true));
        assert_eq!(tokenizer.render_chat(&msg, true, &kwargs).unwrap(), "A|no_think|True"); // as Python prints it
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn harmony_split_takes_the_final_channel() {
        let t = "<|channel|>analysis<|message|>Think.<|end|><|start|>assistant<|channel|>final<|message|>Paris.<|return|>";
        assert_eq!(harmony_split(t), Some(("Think.".into(), "Paris.".into())));
        assert_eq!(harmony_split("<|channel|>final<|message|> Hi "), Some((String::new(), "Hi".into())));
        assert_eq!(harmony_split("<|channel|>analysis<|message|>cut short"), None);
        assert_eq!(harmony_split("plain text"), None);
    }

    #[test]
    fn harmony_progress_grows_reasoning_then_answer() {
        let steps = ["<|channel|>analysis<|message|>Th", "<|channel|>analysis<|message|>Think.<|end|>",
                     "<|channel|>analysis<|message|>Think.<|end|><|start|>assistant<|channel|>final<|message|>Pa",
                     "<|channel|>analysis<|message|>Think.<|end|><|start|>assistant<|channel|>final<|message|>Paris."];
        let got: Vec<_> = steps.iter().map(|t| harmony_progress(t)).collect();
        assert_eq!(got[0], ("Th".into(), String::new()));
        assert_eq!(got[1], ("Think.".into(), String::new()));
        assert_eq!(got[2], ("Think.".into(), "Pa".into()));
        assert_eq!(got[3], ("Think.".into(), "Paris.".into()));
    }

    #[test]
    fn tagged_reasoning_streams_monotonically_and_splits() {
        let gemma = Reasoning::for_prompt(
            ReasoningFormat::Tagged { open: "<|channel>thought\n".into(), close: "<channel|>".into() },
            "<|turn>user\nHi<turn|>\n<|turn>model\n",
        );
        assert!(!gemma.opened_by_prompt);
        let reply = "<|channel>thought\nThe user greets.<channel|>Hello!";
        let mut last = (String::new(), String::new());
        for end in (0..=reply.len()).filter(|&i| reply.is_char_boundary(i)) {
            let now = gemma.progress(&reply[..end]);
            assert!(now.0.starts_with(&last.0) && now.1.starts_with(&last.1), "{end}: {last:?} -> {now:?}");
            last = now;
        }
        assert_eq!(last, ("The user greets.".into(), "Hello!".into()));
        assert_eq!(gemma.progress("<|channel>thought\nHm <chan"), ("Hm ".into(), String::new()));
        assert_eq!(gemma.split("Hello!"), (String::new(), "Hello!".into()));
        assert_eq!(gemma.split("<|channel>thought\ncut"), ("cut".into(), String::new()));

        let qwen = Reasoning::for_prompt(
            ReasoningFormat::Tagged { open: "<think>".into(), close: "</think>".into() },
            "<|im_start|>assistant\n<think>\n",
        );
        assert!(qwen.opened_by_prompt);
        assert_eq!(qwen.split("Plan.\n</think>\n\nParis."), ("Plan.".into(), "Paris.".into()));
        assert_eq!(qwen.progress("Plan.\n</thi"), ("Plan.\n".into(), String::new()));
        let closed = Reasoning::for_prompt(qwen.format.clone(), "<think>\n\n</think>\n\n");
        assert_eq!(closed.split("Paris."), (String::new(), "Paris.".into()));
        let harmony = Reasoning::for_prompt(ReasoningFormat::Harmony, "");
        assert_eq!(harmony.split("<|channel|>analysis<|message|>cut short"), ("cut short".into(), String::new()));
    }

    #[test]
    fn strftime_utc_matches_known_dates() {
        assert_eq!(strftime_utc("%Y-%m-%d", 0), "1970-01-01");
        assert_eq!(strftime_utc("%Y-%m-%d %H:%M:%S", 1_793_404_800 + 3723), "2026-10-31 01:02:03");
        assert_eq!(strftime_utc("%d %b %Y, %A (%j)", 1_709_164_800), "29 Feb 2024, Thursday (060)");
        assert_eq!(strftime_utc("100%% %Q", 0), "100% %Q");
    }

    #[test]
    fn generation_config_eos_replaces_the_fallback_names() {
        // GPT-OSS: <|end|> closes the analysis message mid-turn; only generation_config's ids stop
        let (dir, _) = fixture(false, false);
        let mut tk = HfTokenizer::from_file(dir.join("tokenizer.json")).unwrap();
        tk.add_special_tokens([AddedToken::from("<|end|>", true), AddedToken::from("<|return|>", true)])
            .unwrap();
        tk.save(dir.join("tokenizer.json"), false).unwrap();
        let end = tk.token_to_id("<|end|>").unwrap();
        let ret = tk.token_to_id("<|return|>").unwrap();
        let before = Tokenizer::load(&dir).unwrap();
        assert!(before.is_eos(end), "without generation_config.json the fallback names apply");
        std::fs::write(dir.join("generation_config.json"), json!({"eos_token_id": [ret]}).to_string()).unwrap();
        let after = Tokenizer::load(&dir).unwrap();
        assert!(after.is_eos(ret) && after.is_eos(1) && !after.is_eos(end));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn native_stream_buffers_split_utf8_and_keeps_exact_complete_decode() {
        let (dir, tokenizer) = fixture(false, true);
        let mut stream = tokenizer.decode_stream();
        for id in [3, 4, 5] {
            assert_eq!(stream(id).unwrap(), None);
        }
        assert_eq!(stream(6).unwrap(), Some("🙂".into()));
        assert_eq!(stream(2).unwrap(), Some("A".into()));
        assert_eq!(tokenizer.decode(&[3, 4, 5, 6, 2]).unwrap(), "🙂A");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn native_stream_preserves_metaspace_decoder_context() {
        let (dir, mut tokenizer) = fixture(false, false);
        let vocab: tokenizers::models::bpe::Vocab = [("▁This".to_owned(), 0)].into_iter().collect();
        let mut tk = HfTokenizer::new(
            BPE::builder()
                .vocab_and_merges(vocab, vec![])
                .build()
                .unwrap(),
        );
        tk.with_decoder(Some(tokenizers::decoders::metaspace::Metaspace::default()));
        tokenizer.tk = tk;
        let mut stream = tokenizer.decode_stream();
        assert_eq!(stream(0).unwrap(), Some("This".into()));
        assert_eq!(stream(0).unwrap(), Some(" This".into()));
        assert_eq!(tokenizer.decode(&[0, 0]).unwrap(), "This This");
        std::fs::remove_dir_all(dir).unwrap();
    }
}

pub struct Tokenizer {
    constrained_vocab:
        std::sync::OnceLock<Result<std::sync::Arc<crate::structured_output::Vocabulary>, String>>,
    tool_format: Option<crate::tool_calls::Format>,
    reasoning: Option<ReasoningFormat>,
    template_vars: TemplateVars,
    tk: HfTokenizer,
    env: Environment<'static>,
    add_bos_token: bool,
    gguf_add_eos: Option<u32>,
    bos_token: Option<String>,
    eos_ids: std::collections::HashSet<u32>,
}

impl Tokenizer {
    /// Load tokenizer files and the model chat template from a model directory.
    pub fn load(model_dir: &Path) -> Result<Tokenizer, String> {
        if !model_dir.join("tokenizer.json").try_exists().map_err(|e| e.to_string())? {
            let (tk, config) = gguf_tokenizer::load(model_dir)?;
            return Self::from_parts(model_dir, tk, config, Value::Null);
        }
        let tk = HfTokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| format!("loading tokenizer.json: {e}"))?;

        let config_text = std::fs::read_to_string(model_dir.join("tokenizer_config.json"))
            .map_err(|e| format!("reading tokenizer_config.json: {e}"))?;
        let config: Value = serde_json::from_str(&config_text)
            .map_err(|e| format!("parsing tokenizer_config.json: {e}"))?;
        let special_map = match std::fs::read_to_string(model_dir.join("special_tokens_map.json")) {
            Ok(text) => serde_json::from_str::<Value>(&text)
                .map_err(|e| format!("parsing special_tokens_map.json: {e}"))?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Value::Null,
            Err(e) => return Err(format!("reading special_tokens_map.json: {e}")),
        };
        Self::from_parts(model_dir, tk, config, special_map)
    }

    fn from_parts(model_dir: &Path, tk: HfTokenizer, config: Value, special_map: Value) -> Result<Tokenizer, String> {
        let special_token = |name: &str| {
            [
                config.get(name),
                config.get("special_tokens_dict").and_then(|v| v.get(name)),
                special_map.get(name),
            ]
            .into_iter()
            .flatten()
            .find_map(|value| {
                value
                    .as_str()
                    .or_else(|| value.get("content").and_then(Value::as_str))
            })
            .map(str::to_owned)
        };
        let bos_token = special_token("bos_token");
        let eos_token = special_token("eos_token");
        let add_bos_token = config
            .get("add_bos_token")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let gguf_add_eos = if config.get("_gguf").and_then(Value::as_bool) == Some(true)
            && config.get("add_eos_token").and_then(Value::as_bool) == Some(true) {
            eos_token.as_deref().and_then(|t| tk.token_to_id(t))
        } else { None };

        let template_src = match std::fs::read_to_string(model_dir.join("chat_template.jinja")) {
            Ok(text) => text,
            Err(_) => config
                .get("chat_template")
                .and_then(Value::as_str)
                .ok_or("model needs chat_template.jinja or tokenizer_config.json chat_template")?
                .to_owned(),
        };
        let mut env = Environment::new();
        if let Some(token) = &bos_token {
            env.add_global("bos_token", token.clone());
        }
        if let Some(token) = &eos_token {
            env.add_global("eos_token", token.clone());
        }
        // Mirror transformers' jinja2 Environment(trim_blocks=True, lstrip_blocks=True). Most
        // templates use `{%- -%}` everywhere and render the same either way; HY V3's opens with
        // a plain `{# ... #}` comment and `{% if %}` blocks, whose newlines transformers drops.
        // Publisher templates use Python json.dumps' ensure_ascii keyword. The native filter
        // already writes Unicode; consume the explicit false before delegating its other options.
        env.add_filter(
            "tojson",
            |v: minijinja::Value,
             indent: Option<minijinja::Value>,
             args: minijinja::value::Kwargs| {
                let _: Option<bool> = args.get("ensure_ascii")?;
                minijinja::filters::tojson(&v, indent, args)
            },
        );
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        // jinja2's `iterable` is `iter(value)` succeeding, so `none is iterable` is false there;
        // minijinja iterates none as empty and says true. We pass `tools` = none as transformers
        // does, so Nemotron-H's `tools is iterable and tools | length > 0` failed on `length`.
        env.add_test("iterable", |v: &minijinja::Value| !v.is_none() && v.try_iter().is_ok());
        env.add_function(
            "raise_exception",
            |msg: String| -> Result<String, minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    msg,
                ))
            },
        );
        // The template is written for Python's jinja2 and calls Python string
        // methods (str.startswith/.endswith/...) that minijinja doesn't define
        // natively; minijinja-contrib's pycompat shim adds them.
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        // transformers passes `strftime_now` into the template's globals; GPT-OSS's harmony
        // template stamps "Current date:" with it. UTC: the server has no timezone of its own.
        env.add_function("strftime_now", |fmt: String| -> String {
            let secs = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);
            strftime_utc(&fmt, secs)
        });
        // Where the reply's reasoning ends, from the model's own files: harmony channels in the
        // template; else the tokenizer_config `response_template` thinking field (Gemma 4); else
        // `<think>`/`</think>` in the template (Qwen3.5).
        let tagged = |open: &str, close: &str| ReasoningFormat::Tagged { open: open.into(), close: close.into() };
        let thinking = config.pointer("/response_template/fields/thinking");
        let reasoning = if template_src.contains("<|channel|>") {
            Some(ReasoningFormat::Harmony)
        } else if let (Some(open), Some(close)) = (
            thinking.and_then(|t| t.get("open")).and_then(Value::as_str),
            thinking.and_then(|t| t.get("close")).and_then(Value::as_str),
        ) {
            Some(tagged(open, close))
        } else if template_src.contains("<think>") && template_src.contains("</think>") {
            Some(tagged("<think>", "</think>"))
        } else {
            None
        };
        let tool_format = crate::tool_calls::Format::of(&template_src);
        let template_vars = TemplateVars::of(&template_src);
        env.add_template_owned("chat", template_src)
            .map_err(|e| format!("compiling chat_template.jinja: {e}"))?;

        // generation_config.json's eos_token_id is what transformers stops on. Where a model ships
        // one it is the whole list: the fallback names below include `<|end|>`, which GPT-OSS's
        // harmony format writes after its analysis message and BEFORE the final answer.
        let generation_eos: Option<Vec<u32>> =
            std::fs::read_to_string(model_dir.join("generation_config.json"))
                .ok()
                .and_then(|text| serde_json::from_str::<Value>(&text).ok())
                .and_then(|g| match g.get("eos_token_id") {
                    Some(Value::Number(n)) => n.as_u64().map(|id| vec![id]),
                    Some(Value::Array(ids)) => Some(ids.iter().filter_map(Value::as_u64).collect()),
                    _ => None,
                })
                .map(|ids| ids.into_iter().filter_map(|id| u32::try_from(id).ok()).collect());
        let mut eos_ids: std::collections::HashSet<u32> = match config.get("eos_token_id") {
            Some(Value::Number(n)) => n
                .as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .into_iter()
                .collect(),
            Some(Value::Array(ids)) => ids
                .iter()
                .filter_map(|id| id.as_u64())
                .filter_map(|id| u32::try_from(id).ok())
                .collect(),
            _ => std::collections::HashSet::new(),
        };
        if let Some(name) = eos_token.as_deref() {
            if let Some(id) = tk.token_to_id(name) {
                eos_ids.insert(id);
            }
        }
        if let Some(ids) = &generation_eos {
            eos_ids.extend(ids.iter().copied());
        }
        for name in if generation_eos.is_some() { &[][..] } else { &[
            "<|im_end|>",
            "<|eot_id|>",
            "<end_of_turn>",
            "<|end|>",
            "<|endoftext|>",
            "<|end_of_text|>",
            "<｜end▁of▁sentence｜>",
        ][..] } {
            if let Some(id) = tk.token_to_id(name) {
                eos_ids.insert(id);
            }
        }
        if eos_ids.is_empty() {
            return Err("tokenizer declares no recognized end token".into());
        }
        Ok(Tokenizer {
            tk,
            env,
            tool_format,
            constrained_vocab: std::sync::OnceLock::new(),
            reasoning,
            template_vars,
            add_bos_token,
            gguf_add_eos,
            bos_token,
            eos_ids,
        })
    }

    // ---- core encode / decode -------------------------------------------
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        // A rendered template already owns its special tokens. In particular,
        // do not let the tokenizer's postprocessor add a second leading BOS.
        let has_bos = self
            .bos_token
            .as_ref()
            .is_some_and(|bos| text.starts_with(bos));
        let mut ids = self.encode_with(text, (self.add_bos_token || self.gguf_add_eos.is_some()) && !has_bos)?;
        if has_bos {
            if let Some(eos) = self.gguf_add_eos { ids.push(eos); }
        }
        Ok(ids)
    }

    /// `add_special_tokens` controls only the model's automatic BOS/EOS
    /// (Qwen3.6 adds none); special tokens already present in the string
    /// (e.g. `<|im_start|>` from the chat template) are always recognized.
    pub fn encode_with(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, String> {
        self.tk
            .encode(text, add_special_tokens)
            .map_err(|e| format!("encode: {e}"))
            .map(|enc| enc.get_ids().to_vec())
    }

    /// Token ids -> text. Byte-level BPE, so decode the whole sequence rather
    /// than per-token (a single token can be a partial UTF-8 byte).
    pub fn decode(&self, ids: &[u32]) -> Result<String, String> {
        self.tk
            .decode(ids, false)
            .map_err(|e| format!("decode: {e}"))
    }

    /// The tokenizer's native stateful decoder buffers partial UTF-8 and
    /// preserves decoder context (e.g. SentencePiece's initial-space rule).
    pub fn decode_stream(&self) -> impl FnMut(u32) -> Result<Option<String>, String> + '_ {
        let mut stream = self.tk.decode_stream(false);
        move |id| stream.step(id).map_err(|e| format!("stream decode: {e}"))
    }

    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.tk.id_to_token(id)
    }

    pub fn is_eos(&self, tok: u32) -> bool {
        self.eos_ids.contains(&tok)
    }

    pub fn stop_ids(&self) -> std::collections::HashSet<u32> {
        self.eos_ids.clone()
    }

    /// The token that closes a tagged reasoning block (`</think>`), when the marker is one token:
    /// what `think_end_bias` raises (SERVER-SETTINGS).
    pub fn think_end_id(&self) -> Option<u32> {
        match &self.reasoning {
            Some(ReasoningFormat::Tagged { close, .. }) => self.tk.token_to_id(close),
            _ => None,
        }
    }

    /// Lazily build and share exact token bytes for requests with a JSON constraint.
    pub fn structured_vocabulary(
        &self,
    ) -> Result<std::sync::Arc<crate::structured_output::Vocabulary>, String> {
        self.constrained_vocab
            .get_or_init(|| {
                crate::structured_output::Vocabulary::from_tokenizer(&self.tk, self.stop_ids())
                    .map(std::sync::Arc::new)
            })
            .clone()
    }

    pub fn tool_format(&self) -> Option<crate::tool_calls::Format> {
        self.tool_format
    }

    /// Whether the chat template uses OpenAI harmony channels (GPT-OSS).
    pub fn is_harmony(&self) -> bool {
        self.reasoning == Some(ReasoningFormat::Harmony)
    }

    /// The splitter for a reply to `prompt` (the rendered chat template), or None when the model's
    /// files mark no reasoning block.
    pub fn reasoning_for(&self, prompt: &str) -> Option<Reasoning> {
        self.reasoning.clone().map(|f| Reasoning::for_prompt(f, prompt))
    }

    // ---- chat template ----------------------------------------------------
    /// The thinking switches the chat template reads (`enable_thinking`, `reasoning_effort`).
    pub fn template_vars(&self) -> TemplateVars {
        self.template_vars
    }

    /// Render messages through the model's chat template with only `enable_thinking` set.
    pub fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> Result<String, String> {
        let mut kwargs = Map::new();
        kwargs.insert("enable_thinking".into(), Value::Bool(enable_thinking));
        self.render_chat(messages, add_generation_prompt, &kwargs)
    }

    /// Render messages through the model's chat template with extra template variables, the way
    /// transformers and vLLM pass `chat_template_kwargs` (`enable_thinking`, `reasoning_effort`,
    /// `preserve_thinking`, ...). `messages` and `add_generation_prompt` remain reserved;
    /// the server supplies `tools` through these variables after processing request options.
    pub fn render_chat(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        kwargs: &Map<String, Value>,
    ) -> Result<String, String> {
        let tmpl = self.env.get_template("chat").map_err(|e| e.to_string())?;
        let mut msgs = Vec::new();
        for m in messages {
            let mut msg = serde_json::to_value(m).map_err(|e| e.to_string())?;
            if msg
                .get("tool_calls")
                .is_some_and(|v| v.is_null() || v.as_array().is_some_and(Vec::is_empty))
            {
                msg.as_object_mut().unwrap().remove("tool_calls");
            }
            if msg.get("tool_calls").is_some_and(|v| !v.is_array()) {
                return Err("tool_calls must be an array".into());
            }
            if let Some(calls) = msg.get_mut("tool_calls").and_then(Value::as_array_mut) {
                for call in calls {
                    if let Some(args) = call.pointer_mut("/function/arguments") {
                        if let Some(text) = args.as_str() {
                            *args = serde_json::from_str(text)
                                .map_err(|e| format!("tool_calls arguments must be JSON: {e}"))?;
                        }
                        if !args.is_object() {
                            return Err("tool_calls arguments must be a JSON object".into());
                        }
                    }
                }
            }
            // GPT-OSS names its reasoning field `thinking`.
            if let Some(r) = msg.get("reasoning_content").cloned() {
                msg["thinking"] = r;
            }
            msgs.push(msg);
        }
        let mut ctx: std::collections::BTreeMap<String, minijinja::Value> = kwargs
            .iter()
            .map(|(k, v)| (k.clone(), minijinja::Value::from_serialize(v)))
            .collect();
        ctx.insert("messages".into(), minijinja::Value::from_serialize(msgs));
        ctx.insert("add_generation_prompt".into(), add_generation_prompt.into());
        ctx.entry("tools".into())
            .or_insert_with(|| minijinja::Value::from(()));
        tmpl.render(ctx)
            .map_err(|e| format!("rendering chat template: {e:#}"))
    }
}
