//! GGUF vocabulary -> the native tokenizer pipeline. No tensor data is read.
use crate::gguf::{self, Metadata};
use serde_json::{json, Map, Value};
use std::{collections::HashMap, path::Path};
use tokenizers::Tokenizer;

type Meta = HashMap<String, Metadata>;

// llama.cpp infers these EOG attributes even when a converter marks the
// spelling NORMAL (notably Qwen3's ordinary-vocabulary `</s>`).
const EOG_SPELLINGS: &[&str] = &[
    "<|eot_id|>",
    "<|eom_id|>",
    "<|end_of_text|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<turn|>",
    "<|tool_response>",
    "<|fim_pad|>",
    "<|repo_name|>",
    "<|file_sep|>",
    "</s>",
];

fn string<'a>(m: &'a Meta, key: &str) -> Result<&'a str, String> {
    match m.get(key) {
        Some(Metadata::String(s)) => Ok(s),
        _ => Err(format!("GGUF needs string {key}")),
    }
}

fn array<'a>(m: &'a Meta, key: &str) -> Result<&'a [Metadata], String> {
    match m.get(key) {
        Some(Metadata::Array(a)) => Ok(a),
        _ => Err(format!("GGUF needs array {key}")),
    }
}

fn flag(m: &Meta, key: &str, default: bool) -> Result<bool, String> {
    match m.get(key) {
        None => Ok(default),
        Some(Metadata::Bool(v)) => Ok(*v),
        _ => Err(format!("GGUF {key} must be a bool")),
    }
}

fn id(m: &Meta, key: &str, len: usize) -> Result<Option<u32>, String> {
    match m.get(key) {
        None => Ok(None),
        Some(Metadata::Unsigned(n)) if *n < len as u64 => Ok(Some(*n as u32)),
        _ => Err(format!("GGUF {key} is not a valid vocabulary id")),
    }
}

pub(super) fn load(dir: &Path) -> Result<(Tokenizer, Value), String> {
    let path = gguf::model_file(dir)?;
    from_metadata(&gguf::read_metadata_file(&path)?).map_err(|e| format!("{}: {e}", path.display()))
}

fn from_metadata(m: &Meta) -> Result<(Tokenizer, Value), String> {
    let model = string(m, "tokenizer.ggml.model")?;
    let pre = match m.get("tokenizer.ggml.pre") {
        Some(Metadata::String(s)) => s.as_str(),
        None if model == "gemma4" => "gemma4",
        _ => return Err("GGUF needs tokenizer.ggml.pre".into()),
    };
    let (gemma, llama, pattern) = match (model, pre) {
        ("gpt2", "llama-bpe" | "llama3" | "llama-v3") => (
            false,
            true,
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        ),
        ("gpt2", "qwen2") => (
            false,
            false,
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        ),
        ("gpt2", "qwen35") => (
            false,
            false,
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        ),
        ("gemma4", "gemma4") => (true, false, r"[^\n]+|[\n]+"),
        _ => {
            return Err(format!(
                "unsupported GGUF tokenizer model {model:?}, pre-tokenizer {pre:?}"
            ))
        }
    };
    // These flags change the text before BPE. Do not silently approximate them.
    for key in [
        "tokenizer.ggml.add_space_prefix",
        "tokenizer.ggml.remove_extra_whitespaces",
        "tokenizer.ggml.add_sep_token",
        "tokenizer.ggml.normalizer.lowercase",
        "tokenizer.ggml.normalizer.strip_accents",
    ] {
        if flag(m, key, false)? {
            return Err(format!("unsupported GGUF {key}=true for {pre}"));
        }
    }
    let nfc = flag(m, "tokenizer.ggml.normalizer.nfc", false)?;
    let tokens: Vec<&str> = array(m, "tokenizer.ggml.tokens")?
        .iter()
        .map(|v| match v {
            Metadata::String(s) => Ok(s.as_str()),
            _ => Err("GGUF tokens must all be strings".to_owned()),
        })
        .collect::<Result<_, _>>()?;
    if tokens.is_empty() {
        return Err("GGUF vocabulary is empty".into());
    }
    // llama.cpp explicitly restores Gemma4's `</s>` to an ordinary token.
    let eog_spelling = |s: &str| EOG_SPELLINGS.contains(&s) && !(gemma && s == "</s>");
    let types = array(m, "tokenizer.ggml.token_type")?;
    if types.len() != tokens.len() {
        return Err("GGUF token_type and tokens lengths differ".into());
    }
    if let Some(scores) = m.get("tokenizer.ggml.scores") {
        match scores {
            Metadata::Array(a)
                if a.len() == tokens.len()
                    && a.iter().all(|v| match v {
                        Metadata::Float(f) => f.is_finite(),
                        Metadata::Signed(_) => true,
                        _ => false,
                    }) =>
            {
                ()
            }
            _ => return Err("GGUF scores must be finite numbers, one per token".into()),
        }
    }
    let mut vocab = Map::new();
    let mut added = Vec::new();
    for (i, (&text, ty)) in tokens.iter().zip(types).enumerate() {
        if vocab.insert(text.into(), json!(i)).is_some() {
            return Err(format!("duplicate GGUF token {text:?}"));
        }
        let ty = match ty {
            Metadata::Signed(t) if (1..=6).contains(t) => *t,
            Metadata::Unsigned(t) if (1..=6).contains(t) => *t as i64,
            _ => return Err(format!("invalid GGUF token_type at {i}")),
        };
        if matches!(ty, 3 | 4) || eog_spelling(text) {
            added.push(
                json!({"id":i,"content":text,"special":true,"single_word":false,
                              "lstrip":false,"rstrip":false,"normalized":false}),
            );
        }
    }
    let merges: Vec<Value> = array(m, "tokenizer.ggml.merges")?
        .iter()
        .map(|v| {
            let Metadata::String(s) = v else {
                return Err("GGUF merges must be strings".to_owned());
            };
            // Gemma permits a literal leading space in a token; the separator is
            // the first space after the first UTF-8 character, as in llama.cpp.
            let first_end = s.chars().next().ok_or("empty GGUF merge")?.len_utf8();
            let at = s[first_end..]
                .find(' ')
                .map(|i| i + first_end)
                .ok_or("invalid GGUF merge")?;
            Ok(json!([&s[..at], &s[at + 1..]]))
        })
        .collect::<Result<_, String>>()?;
    let bos = id(m, "tokenizer.ggml.bos_token_id", tokens.len())?;
    let eos = id(m, "tokenizer.ggml.eos_token_id", tokens.len())?;
    let add_bos = flag(m, "tokenizer.ggml.add_bos_token", llama || gemma)?;
    let add_eos = flag(m, "tokenizer.ggml.add_eos_token", false)?;
    if add_bos && bos.is_none() || add_eos && eos.is_none() {
        return Err("GGUF automatic BOS/EOS enabled without its token id".into());
    }
    let mut stop_ids = Vec::new();
    let mut config = json!({"_gguf":true,"add_bos_token":add_bos,"add_eos_token":add_eos,
                           "chat_template":string(m, "tokenizer.chat_template")?});
    // EOT/EOM are generation stops too. UNK/PAD/SEP/FIM ids are validated even
    // when this text-only tokenizer has no use for them.
    for key in [
        "bos",
        "eos",
        "eot",
        "eom",
        "unknown",
        "padding",
        "separator",
        "mask",
        "fim_pre",
        "fim_suf",
        "fim_mid",
        "fim_pad",
        "fim_rep",
        "fim_sep",
    ] {
        if let Some(n) = id(m, &format!("tokenizer.ggml.{key}_token_id"), tokens.len())? {
            if matches!(
                key,
                "eos" | "eot" | "eom" | "fim_pad" | "fim_rep" | "fim_sep"
            ) {
                stop_ids.push(n);
            }
            if matches!(key, "bos" | "eos") {
                config[format!("{key}_token")] = json!(tokens[n as usize]);
            }
        }
    }
    // llama.cpp also identifies end-of-generation vocabulary spellings when
    // the converter did not emit separate eot/eom metadata keys.
    for &text in EOG_SPELLINGS {
        if !eog_spelling(text) {
            continue;
        }
        if let Some(n) = tokens.iter().position(|s| *s == text) {
            stop_ids.push(n as u32);
        }
    }
    config["eos_token_id"] = json!(stop_ids);
    let mut single = Vec::new();
    let mut specials = Map::new();
    let mut special = |n: u32| {
        let text = tokens[n as usize];
        specials.insert(text.into(), json!({"id":text,"ids":[n],"tokens":[text]}));
        json!({"SpecialToken":{"id":text,"type_id":0}})
    };
    if add_bos {
        single.push(special(bos.unwrap()));
    }
    single.push(json!({"Sequence":{"id":"A","type_id":0}}));
    if add_eos {
        single.push(special(eos.unwrap()));
    }
    let split =
        json!({"type":"Split","pattern":{"Regex":pattern},"behavior":"Isolated","invert":false});
    let mut normalizers = Vec::new();
    if nfc {
        normalizers.push(json!({"type":"NFC"}));
    }
    if gemma {
        normalizers.push(json!({"type":"Replace","pattern":{"String":" "},"content":"▁"}));
    }
    let pipeline = json!({
        "version":"1.0", "truncation":null, "padding":null, "added_tokens":added,
        "normalizer":if normalizers.is_empty() { Value::Null } else { json!({"type":"Sequence","normalizers":normalizers}) },
        "pre_tokenizer":if gemma { split } else { json!({"type":"Sequence","pretokenizers":[split,
            {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]}) },
        "post_processor":{"type":"TemplateProcessing","single":single,"pair":[],"special_tokens":specials},
        "decoder":if gemma { json!({"type":"Sequence","decoders":[
            {"type":"Replace","pattern":{"String":"▁"},"content":" "},
            {"type":"ByteFallback"},{"type":"Fuse"}]}) }
            else { json!({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}) },
        "model":{"type":"BPE","vocab":vocab,"merges":merges,"dropout":null,
            "unk_token":if gemma { id(m, "tokenizer.ggml.unknown_token_id", tokens.len())?.map(|n| tokens[n as usize]) } else { None },
            "continuing_subword_prefix":null,"end_of_word_suffix":null,
            "fuse_unk":gemma,"byte_fallback":gemma,"ignore_merges":llama}
    });
    let tk = Tokenizer::from_bytes(serde_json::to_vec(&pipeline).map_err(|e| e.to_string())?)
        .map_err(|e| format!("building GGUF tokenizer: {e}"))?;
    Ok((tk, config))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::{ChatMessage, Tokenizer as ChatTokenizer};
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);

    fn fixture() -> Meta {
        let mut m = Meta::new();
        for (key, text) in [("model", "gpt2"), ("pre", "qwen2")] {
            m.insert(
                format!("tokenizer.ggml.{key}"),
                Metadata::String(text.into()),
            );
        }
        let tokens = ["a", "b", "ab", "Ġ", "<bos>", "<eos>", "<pad>"];
        m.insert(
            "tokenizer.ggml.tokens".into(),
            Metadata::Array(
                tokens
                    .iter()
                    .map(|s| Metadata::String((*s).into()))
                    .collect(),
            ),
        );
        m.insert(
            "tokenizer.ggml.token_type".into(),
            Metadata::Array(
                (0..7)
                    .map(|i| Metadata::Signed(if i >= 4 { 3 } else { 1 }))
                    .collect(),
            ),
        );
        m.insert(
            "tokenizer.ggml.merges".into(),
            Metadata::Array(vec![Metadata::String("a b".into())]),
        );
        m.insert("tokenizer.ggml.bos_token_id".into(), Metadata::Unsigned(4));
        m.insert("tokenizer.ggml.eos_token_id".into(), Metadata::Unsigned(5));
        m.insert(
            "tokenizer.chat_template".into(),
            Metadata::String(
                "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}{{ eos_token }}"
                    .into(),
            ),
        );
        m
    }

    fn chat(m: &Meta) -> ChatTokenizer {
        let (tk, config) = from_metadata(m).unwrap();
        ChatTokenizer::from_parts(Path::new("/nonexistent"), tk, config, Value::Null).unwrap()
    }

    #[test]
    fn bpe_specials_bos_eos_and_chat_use_metadata() {
        let mut m = fixture();
        m.insert("tokenizer.ggml.add_bos_token".into(), Metadata::Bool(true));
        let tk = chat(&m);
        assert_eq!(tk.encode_with("ab ab<eos>", false).unwrap(), [2, 3, 2, 5]);
        assert_eq!(tk.encode("ab").unwrap(), [4, 2]);
        assert_eq!(tk.encode("<bos>ab").unwrap(), [4, 2]);
        assert_eq!(tk.decode(&[4, 2, 3, 2, 5]).unwrap(), "<bos>ab ab<eos>");
        let rendered = tk
            .apply_chat_template(
                &[ChatMessage {
                    role: "user".into(),
                    content: "ab".into(),
                    extra: Default::default(),
                }],
                true,
                false,
            )
            .unwrap();
        assert_eq!(rendered, "<bos>ab<eos>");
        assert_eq!(tk.encode(&rendered).unwrap(), [4, 2, 5]);
        assert!(tk.is_eos(5));
        assert!(!tk.is_eos(6));
        m.insert("tokenizer.ggml.add_eos_token".into(), Metadata::Bool(true));
        assert_eq!(chat(&m).encode("<bos>ab").unwrap(), [4, 2, 5]);
        m.insert("tokenizer.ggml.add_bos_token".into(), Metadata::Bool(false));
        assert_eq!(chat(&m).encode("ab").unwrap(), [2, 5]);
    }

    #[test]
    fn invalid_or_unsupported_metadata_fails_closed() {
        for (key, value) in [
            ("tokenizer.ggml.pre", Metadata::String("unknown-pre".into())),
            (
                "tokenizer.ggml.model",
                Metadata::String("unknown-model".into()),
            ),
            ("tokenizer.ggml.eos_token_id", Metadata::Unsigned(99)),
            (
                "tokenizer.ggml.add_bos_token",
                Metadata::String("true".into()),
            ),
            ("tokenizer.ggml.add_space_prefix", Metadata::Bool(true)),
            ("tokenizer.ggml.scores", Metadata::Array(vec![])),
            ("tokenizer.ggml.token_type", Metadata::Array(vec![])),
            (
                "tokenizer.ggml.merges",
                Metadata::Array(vec![Metadata::String("invalid".into())]),
            ),
        ] {
            let mut m = fixture();
            m.insert(key.into(), value);
            assert!(from_metadata(&m).is_err(), "{key}");
        }
        let mut m = fixture();
        m.remove("tokenizer.chat_template");
        assert!(from_metadata(&m).is_err());
    }

    #[test]
    fn eog_spellings_match_llama_cpp_even_when_marked_normal() {
        let mut m = fixture();
        let Metadata::Array(tokens) = m.get_mut("tokenizer.ggml.tokens").unwrap() else {
            unreachable!()
        };
        tokens.push(Metadata::String("</s>".into()));
        let Metadata::Array(types) = m.get_mut("tokenizer.ggml.token_type").unwrap() else {
            unreachable!()
        };
        types.push(Metadata::Signed(1));
        let tk = chat(&m);
        assert_eq!(tk.encode_with("</s>", false).unwrap(), [7]);
        assert!(tk.is_eos(7));
        m.insert(
            "tokenizer.ggml.model".into(),
            Metadata::String("gemma4".into()),
        );
        m.insert(
            "tokenizer.ggml.pre".into(),
            Metadata::String("gemma4".into()),
        );
        let (native, _) = from_metadata(&m).unwrap();
        assert!(!native.get_added_tokens_decoder().contains_key(&7));
        assert!(!chat(&m).is_eos(7));
    }

    #[test]
    fn nfc_is_applied_only_when_metadata_requests_it() {
        let mut m = fixture();
        let Metadata::Array(tokens) = m.get_mut("tokenizer.ggml.tokens").unwrap() else {
            unreachable!()
        };
        tokens.extend(["e", "Ì", "ģ", "Ã", "©", "Ã©"].map(|s| Metadata::String(s.into())));
        let Metadata::Array(types) = m.get_mut("tokenizer.ggml.token_type").unwrap() else {
            unreachable!()
        };
        types.extend((0..6).map(|_| Metadata::Signed(1)));
        let Metadata::Array(merges) = m.get_mut("tokenizer.ggml.merges").unwrap() else {
            unreachable!()
        };
        merges.push(Metadata::String("Ã ©".into()));
        assert_eq!(chat(&m).encode_with("e\u{301}", false).unwrap(), [7, 8, 9]);
        m.insert("tokenizer.ggml.normalizer.nfc".into(), Metadata::Bool(true));
        assert_eq!(chat(&m).encode_with("e\u{301}", false).unwrap(), [12]);
    }

    fn value(out: &mut Vec<u8>, v: &Metadata) {
        match v {
            Metadata::String(s) => {
                out.extend(8u32.to_le_bytes());
                out.extend((s.len() as u64).to_le_bytes());
                out.extend(s.as_bytes());
            }
            Metadata::Unsigned(n) => {
                out.extend(4u32.to_le_bytes());
                out.extend((*n as u32).to_le_bytes());
            }
            Metadata::Signed(n) => {
                out.extend(5u32.to_le_bytes());
                out.extend((*n as i32).to_le_bytes());
            }
            Metadata::Bool(b) => {
                out.extend(7u32.to_le_bytes());
                out.push(u8::from(*b));
            }
            Metadata::Array(a) => {
                out.extend(9u32.to_le_bytes());
                let mut first = Vec::new();
                value(&mut first, &a[0]);
                out.extend(&first[..4]);
                out.extend((a.len() as u64).to_le_bytes());
                for v in a {
                    let mut bytes = Vec::new();
                    value(&mut bytes, v);
                    out.extend(&bytes[4..]);
                }
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn bare_publisher_filename_loads_and_ambiguous_or_truncated_files_fail() {
        let dir = std::env::temp_dir().join(format!(
            "phlegm-s6-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&dir).unwrap();
        let m = fixture();
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3u32.to_le_bytes());
        // Tensor descriptors and payloads are deliberately absent.
        bytes.extend(100u64.to_le_bytes());
        bytes.extend((m.len() as u64).to_le_bytes());
        for (key, v) in &m {
            bytes.extend((key.len() as u64).to_le_bytes());
            bytes.extend(key.as_bytes());
            value(&mut bytes, v);
        }
        std::fs::write(dir.join("publisher-Q4_K_M.gguf"), &bytes).unwrap();
        assert_eq!(
            ChatTokenizer::load(&dir).unwrap().encode("ab").unwrap(),
            [2]
        );
        std::fs::write(dir.join("second.gguf"), &bytes).unwrap();
        assert!(ChatTokenizer::load(&dir)
            .err()
            .unwrap()
            .contains("exactly one GGUF"));
        std::fs::remove_file(dir.join("second.gguf")).unwrap();
        bytes.pop();
        std::fs::write(dir.join("publisher-Q4_K_M.gguf"), &bytes).unwrap();
        assert!(ChatTokenizer::load(&dir).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn existing_tokenizer_json_retains_precedence_over_gguf() {
        let (dir, _) = crate::tokenizer::tests::fixture(false, false);
        std::fs::write(dir.join("published.gguf"), b"invalid GGUF").unwrap();
        assert_eq!(ChatTokenizer::load(&dir).unwrap().encode("A").unwrap(), [2]);
        std::fs::write(dir.join("tokenizer.json"), b"invalid JSON").unwrap();
        assert!(ChatTokenizer::load(&dir)
            .err()
            .unwrap()
            .contains("loading tokenizer.json"));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
