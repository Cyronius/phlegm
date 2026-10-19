//! Real tokenizer for Qwen3.6 (Qwen2Tokenizer / byte-level BPE), ported from
//! `tools/kernel-interp/tokenizer.py`.
//!
//! That Python module's own docstring says it loads `tokenizer.json` via "the
//! HuggingFace `tokenizers` crate binding (pure-Rust BPE, no torch/transformers
//! dependency)" — i.e. the Python `tokenizers` package IS a binding over this
//! same Rust crate. So this isn't a port, it's cutting out the middleman: load
//! the same `tokenizer.json` directly with the `tokenizers` crate. Only the
//! chat-template rendering (Python: jinja2) needs a real port, to `minijinja`.
//!
//! Facts (from tokenizer_config.json, mirrored from tokenizer.py's docstring):
//! vocab in tokenizer.json = 248070 real tokens; the model's lm_head is padded
//! to vocab_size = 248320. add_bos_token is false, so encode() adds no BOS by
//! default. Generation stops on any of EOS_TOKEN_IDS.

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

pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

pub struct Tokenizer {
    tk: HfTokenizer,
    env: Environment<'static>,
    add_bos_token: bool,
}

impl Tokenizer {
    /// Load `tokenizer.json` + `tokenizer_config.json` + `chat_template.jinja`
    /// from a model directory (mirrors `Qwen36Tokenizer.__init__`).
    pub fn load(model_dir: &Path) -> Result<Tokenizer, String> {
        let tk = HfTokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| format!("loading tokenizer.json: {e}"))?;

        let config_text = std::fs::read_to_string(model_dir.join("tokenizer_config.json"))
            .map_err(|e| format!("reading tokenizer_config.json: {e}"))?;
        let config: Value = serde_json::from_str(&config_text)
            .map_err(|e| format!("parsing tokenizer_config.json: {e}"))?;
        let add_bos_token = config
            .get("add_bos_token")
            .and_then(Value::as_bool)
            .unwrap_or(false);

        let template_src = std::fs::read_to_string(model_dir.join("chat_template.jinja"))
            .map_err(|e| format!("reading chat_template.jinja: {e}"))?;
        let mut env = Environment::new();
        // Mirror transformers' jinja2 Environment(trim_blocks=True, lstrip_blocks=True):
        // minijinja's default whitespace handling already strips `{%- -%}` control
        // markers; the template exclusively uses those, so no extra config needed.
        env.add_function("raise_exception", |msg: String| -> Result<String, minijinja::Error> {
            Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, msg))
        });
        // The template is written for Python's jinja2 and calls Python string
        // methods (str.startswith/.endswith/...) that minijinja doesn't define
        // natively; minijinja-contrib's pycompat shim adds them.
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_template_owned("chat", template_src)
            .map_err(|e| format!("compiling chat_template.jinja: {e}"))?;

        Ok(Tokenizer { tk, env, add_bos_token })
    }

    // ---- core encode / decode -------------------------------------------
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        self.encode_with(text, self.add_bos_token)
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
        self.tk.decode(ids, false).map_err(|e| format!("decode: {e}"))
    }

    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.tk.id_to_token(id)
    }

    pub fn is_eos(&self, tok: u32) -> bool {
        EOS_TOKEN_IDS.contains(&tok)
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
