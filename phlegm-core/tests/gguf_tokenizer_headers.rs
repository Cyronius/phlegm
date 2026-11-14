//! Opt-in acceptance using range-fetched, checksum-pinned headers (no weights).
use phlegm_core::tokenizer::{ChatMessage, Tokenizer};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

#[test]
#[ignore = "requires PHLEGM_S6_HEADERS with all nine pinned public GGUF headers"]
fn all_published_headers_match_llama_cpp_ids_and_rendered_conversations() {
    let root = PathBuf::from(std::env::var_os("PHLEGM_S6_HEADERS").expect("set PHLEGM_S6_HEADERS"));
    let expected: Value =
        serde_json::from_str(include_str!("fixtures/gguf-tokenizer/expected.json")).unwrap();
    let corpus: Value =
        serde_json::from_str(include_str!("fixtures/gguf-tokenizer/corpus.json")).unwrap();
    assert_eq!(expected.as_object().unwrap().len(), 9);
    for (name, target) in expected.as_object().unwrap() {
        let header = root.join(name).join("header.gguf");
        let bytes = std::fs::read(&header).unwrap_or_else(|e| panic!("{}: {e}", header.display()));
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            target["source"]["header_sha256"].as_str().unwrap(),
            "{name}: header provenance"
        );
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "phlegm-header-{}-{nonce}-{name}",
            std::process::id()
        ));
        std::fs::create_dir(&dir).unwrap();
        std::fs::write(dir.join("publisher-Q4_K_M.gguf"), bytes).unwrap();
        let tk = Tokenizer::load(&dir).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            target["cases"].as_array().unwrap().len(),
            corpus["texts"].as_array().unwrap().len()
                + corpus["conversations"].as_array().unwrap().len(),
            "{name}: fixture must cover the complete corpus"
        );
        let mut stops: Vec<_> = tk.stop_ids().into_iter().collect();
        stops.sort_unstable();
        let expected_stops: Vec<u32> = serde_json::from_value(target["stop_ids"].clone()).unwrap();
        assert_eq!(stops, expected_stops, "{name}: generation stop IDs");
        for case in target["cases"].as_array().unwrap() {
            let case_name = case["name"].as_str().unwrap();
            let text = case["text"].as_str().unwrap();
            if let Some(input) = corpus["texts"]
                .as_array()
                .unwrap()
                .iter()
                .find(|c| c["name"] == case["name"])
            {
                assert_eq!(
                    input["text"].as_str().unwrap(),
                    text,
                    "{name}/{case_name}: fixed corpus input"
                );
            }
            let ids: Vec<u32> = serde_json::from_value(case["ids"].clone()).unwrap();
            let special: Vec<u32> = serde_json::from_value(case["with_special"].clone()).unwrap();
            assert_eq!(
                tk.encode_with(text, false).unwrap(),
                ids,
                "{name}/{case_name}"
            );
            assert_eq!(
                tk.encode_with(text, true).unwrap(),
                special,
                "{name}/{case_name} +special"
            );
            let mut stream = tk.decode_stream();
            let mut decoded = String::new();
            // The native stream deliberately holds a terminal replacement
            // character (also used for incomplete UTF-8). A following ASCII
            // sentinel flushes it; the server separately flushes on finish.
            let mut stream_ids = ids.clone();
            stream_ids.extend(tk.encode_with("\nS6_STREAM_END", false).unwrap());
            for id in &stream_ids {
                if let Some(piece) = stream(*id).unwrap() {
                    decoded.push_str(&piece);
                }
            }
            assert_eq!(
                decoded,
                tk.decode(&stream_ids).unwrap(),
                "{name}/{case_name}: streaming decoder"
            );
            if let Some(chat) = corpus["conversations"]
                .as_array()
                .unwrap()
                .iter()
                .find(|c| c["name"] == case["name"])
            {
                let messages: Vec<_> = chat["messages"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|m| ChatMessage {
                        role: m["role"].as_str().unwrap().into(),
                        content: m["content"].as_str().unwrap().into(),
                        extra: Default::default(),
                    })
                    .collect();
                let rendered = tk
                    .render_chat(
                        &messages,
                        true,
                        corpus["template_kwargs"].as_object().unwrap(),
                    )
                    .unwrap();
                assert_eq!(rendered, text, "{name}/{case_name}: rendering");
                assert_eq!(
                    tk.encode(&rendered).unwrap(),
                    ids,
                    "{name}/{case_name}: template owns BOS"
                );
            }
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
