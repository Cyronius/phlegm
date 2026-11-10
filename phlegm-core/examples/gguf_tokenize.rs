//! Offline acceptance probe: model directory + corpus JSON -> rendered text/ids.
use phlegm_core::tokenizer::{ChatMessage, Tokenizer};
use serde_json::{json, Map, Value};
use std::{
    io::{self, Write},
    path::Path,
};

fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: gguf_tokenize MODEL_DIR CORPUS.json".into());
    }
    let tk = Tokenizer::load(Path::new(&args[1]))?;
    let corpus: Value =
        serde_json::from_slice(&std::fs::read(&args[2]).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let mut cases = Vec::new();
    for case in corpus["texts"].as_array().ok_or("texts must be an array")? {
        cases.push((
            case["name"].as_str().ok_or("case needs name")?.to_owned(),
            case["text"].as_str().ok_or("case needs text")?.to_owned(),
        ));
    }
    for case in corpus["conversations"]
        .as_array()
        .ok_or("conversations must be an array")?
    {
        let messages: Vec<_> = case["messages"]
            .as_array()
            .ok_or("messages must be an array")?
            .iter()
            .map(|m| {
                Ok(ChatMessage {
                    role: m["role"].as_str().ok_or("role")?.into(),
                    content: m["content"].as_str().ok_or("content")?.into(),
                    extra: Default::default(),
                })
            })
            .collect::<Result<_, String>>()?;
        let kwargs: Map<String, Value> =
            serde_json::from_value(corpus["template_kwargs"].clone()).map_err(|e| e.to_string())?;
        cases.push((
            case["name"].as_str().ok_or("name")?.into(),
            tk.render_chat(&messages, true, &kwargs)?,
        ));
    }
    let mut result = Vec::new();
    for (name, text) in cases {
        let ids = tk.encode_with(&text, false)?;
        result.push(json!({"name":name,"text":text,"ids":ids,
            "with_special":tk.encode_with(&text, true)?,"auto":tk.encode(&text)?,
            "decoded":tk.decode(&ids)?}));
    }
    let mut out = io::stdout().lock();
    serde_json::to_writer(&mut out, &json!({"cases":result,"stop_ids":tk.stop_ids()}))
        .map_err(|e| e.to_string())?;
    writeln!(&mut out).map_err(|e| e.to_string())
}
