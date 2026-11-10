use phlegm_core::{
    tokenizer::{ChatMessage, Tokenizer},
    tool_calls::Format,
};
use serde_json::{json, Map, Value};
#[test]
fn cpu_model_fixtures() {
    for (format, fixture) in [
        (
            Format::Function,
            include_str!("fixtures/tool_calls/KAT-Coder-V2.5-Dev-Q4pure-NPU2.json"),
        ),
        (
            Format::Gemma,
            include_str!("fixtures/tool_calls/gemma-4-26B-A4B-it-NPU2.json"),
        ),
        (
            Format::Harmony,
            include_str!("fixtures/tool_calls/gpt-oss-20b-NPU2.json"),
        ),
        (
            Format::Tagged,
            include_str!("fixtures/tool_calls/flint-NPU2.json"),
        ),
        (
            Format::Hy,
            include_str!("fixtures/tool_calls/Hy-Memory-30B-A3B-NPU2.json"),
        ),
        (
            Format::Lfm,
            include_str!("fixtures/tool_calls/LFM2.5-24B-A2B-NPU2.json"),
        ),
    ] {
        let fixture: Value = serde_json::from_str(fixture).unwrap();
        let p = format.parse(fixture["output"].as_str().unwrap());
        assert_eq!(p.calls.len(), 1, "{}: {p:?}", fixture["model"]);
        assert_eq!(p.calls[0]["function"]["name"], "get_weather");
        assert_eq!(
            serde_json::from_str::<Value>(p.calls[0]["function"]["arguments"].as_str().unwrap())
                .unwrap(),
            json!({"city":"Paris"})
        );
    }
}
#[test]
#[ignore = "requires installed model files under /home/cyrus/models"]
fn installed_templates_round_trip_tools() {
    let mut tested = 0;
    for dir in std::fs::read_dir("/home/cyrus/models").unwrap() {
        let dir = dir.unwrap().path();
        if !dir.join("chat_template.jinja").exists() {
            continue;
        }
        if Format::of(&std::fs::read_to_string(dir.join("chat_template.jinja")).unwrap()).is_none()
        {
            continue;
        }
        println!("rendering {}", dir.display());
        let t = Tokenizer::load(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        let mut kwargs = Map::new();
        kwargs.insert("tools".into(),json!([{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]));
        kwargs.insert("enable_thinking".into(), json!(false));
        let messages=vec![ChatMessage{role:"user".into(),content:"Weather?".into(),..Default::default()},
            ChatMessage{role:"assistant".into(),extra:serde_json::from_value(json!({"tool_calls":[{"id":"call_a","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]})).unwrap(),..Default::default()},
            ChatMessage{role:"tool".into(),content:"sunny".into(),extra:serde_json::from_value(json!({"tool_call_id":"call_a"})).unwrap()}];
        let prompt = t
            .render_chat(&messages, true, &kwargs)
            .unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        assert!(
            prompt.contains("get_weather") && prompt.contains("Paris") && prompt.contains("sunny"),
            "{}",
            dir.display()
        );
        assert!(!prompt.contains("response:unknown"), "{}", dir.display());
        tested += 1;
    }
    assert!(tested >= 18, "{tested}");
    println!("{tested} installed templates round-tripped");
}
