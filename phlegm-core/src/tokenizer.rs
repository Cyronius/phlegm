//! Hugging Face tokenizer and chat template for the selected open-kernel model.
//! The token IDs and end tokens are read from the model's tokenizer files;
//! templates may be an external Jinja file or embedded in tokenizer_config.json.

use minijinja::{context, Environment};
use serde_json::Value;
use std::path::Path;
use tokenizers::Tokenizer as HfTokenizer;

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

pub struct ChatMessage {
    pub role: String,
    pub content: String,
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
    harmony: bool,
    tk: HfTokenizer,
    env: Environment<'static>,
    add_bos_token: bool,
    bos_token: Option<String>,
    eos_ids: std::collections::HashSet<u32>,
}

impl Tokenizer {
    /// Load tokenizer files and the model chat template from a model directory.
    pub fn load(model_dir: &Path) -> Result<Tokenizer, String> {
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
        // Mirror transformers' jinja2 Environment(trim_blocks=True, lstrip_blocks=True):
        // minijinja's default whitespace handling already strips `{%- -%}` control
        // markers; the template exclusively uses those, so no extra config needed.
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
        let harmony = template_src.contains("<|channel|>");
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
            harmony,
            add_bos_token,
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
        self.encode_with(text, self.add_bos_token && !has_bos)
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

    /// The chat template speaks OpenAI's harmony format (GPT-OSS): replies arrive as an analysis
    /// channel then a final one, which the CLI and server separate (`harmony_split`).
    pub fn is_harmony(&self) -> bool {
        self.harmony
    }

    // ---- chat template ----------------------------------------------------
    /// Render messages through the packaged chat_template.jinja. With
    /// `add_generation_prompt=true` the render ends at
    /// `<|im_start|>assistant\n` plus a `<think>\n` opener (Qwen3.6 is a
    /// thinking model); pass `enable_thinking=false` to emit an empty
    /// `<think>\n\n</think>` block.
    pub fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> Result<String, String> {
        let tmpl = self.env.get_template("chat").map_err(|e| e.to_string())?;
        let msgs: Vec<_> = messages
            .iter()
            .map(|m| context! { role => m.role.clone(), content => m.content.clone() })
            .collect();
        tmpl.render(context! {
            messages => msgs,
            add_generation_prompt => add_generation_prompt,
            enable_thinking => enable_thinking,
            tools => Option::<Value>::None,
        })
        .map_err(|e| format!("rendering chat template: {e:#}"))
    }
}
