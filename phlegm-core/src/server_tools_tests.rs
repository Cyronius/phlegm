#[test]
fn schema_mask_drives_fake_logits_backend_and_rejects_unsupported_features() {
    struct Logits;
    impl Backend for Logits {
        fn generate(
            &mut self,
            _: &[u32],
            p: &mut GenParams,
            emit: &mut dyn FnMut(crate::backend::TokenEvent<'_>) -> std::ops::ControlFlow<()>,
        ) -> Result<(), String> {
            let target = br#"{"answer":true,"count":2}"#;
            let mut history = vec![];
            for pos in 0..p.max_tokens {
                let mut logits = vec![0.0; 257];
                // Invalid token would win without the grammar, under greedy and sampled paths.
                logits[b'x' as usize] = 100.0;
                logits[if pos < target.len() {
                    target[pos] as usize
                } else {
                    256
                }] = 90.0;
                let id = p.sampler.try_sample(&logits, &history)? as u32;
                if p.stop_ids.contains(&id) {
                    break;
                }
                history.push(i64::from(id));
                if emit(crate::backend::TokenEvent { id, logits: Some(&logits) }).is_break() { break; }
            }
            Ok(())
        }
    }
    for streaming in [false, true] {
        let cfg = ServerConfig {
            port: 0,
            model_id: "fixture".into(),
            ..ServerConfig::from_env()
        };
        let server = bind(&cfg).unwrap();
        let addr = server.server_addr().to_ip().unwrap().to_string();
        let state = Arc::new(ServerState {
            backend: Mutex::new(Box::new(Logits)),
            context_capacity: None,
            tokenizer: ServerTokenizer::Placeholder(PlaceholderTokenizer::new()),
            cfg,
            embedder: None,
            embedding_model: None,
            embedding_only: false,
        });
        let thread = std::thread::spawn(move || {
            for _ in 0..4 {
                handle_request(server.recv().unwrap(), &state);
            }
        });
        let schema = json!({"type":"object","properties":{"answer":{"const":true},"count":{"type":"integer"}},"required":["answer","count"],"additionalProperties":false});
        let mut req = json!({"messages":[{"role":"user","content":"JSON please"}],"stream":streaming,"max_tokens":64,"response_format":{"type":"json_schema","json_schema":{"name":"answer","strict":true,"schema":schema}}});
        // Tools auto can coexist with JSON output; plain JSON is parsed independently of calls.
        req["tools"] = json!([{"type":"function","function":{"name":"get_weather"}}]);
        let (status, _, body) = http_request(
            &addr,
            "POST",
            "/v1/chat/completions",
            Some(&req.to_string()),
        );
        assert_eq!(status, 200);
        let text = if streaming {
            body.split("\n\n")
                .filter_map(|s| s.strip_prefix("data: "))
                .filter_map(|s| serde_json::from_str::<Value>(s).ok())
                .filter_map(|v| {
                    v["choices"][0]["delta"]["content"]
                        .as_str()
                        .map(str::to_owned)
                })
                .collect::<String>()
        } else {
            serde_json::from_str::<Value>(&body).unwrap()["choices"][0]["message"]["content"]
                .as_str()
                .unwrap()
                .to_owned()
        };
        assert_eq!(
            serde_json::from_str::<Value>(&text).unwrap(),
            json!({"answer":true,"count":2})
        );
        for (format, needle) in [
            (
                json!({"type":"json_schema","json_schema":{"name":"x","strict":true,"schema":{"type":"string","pattern":"a"}}}),
                "pattern",
            ),
            (json!({"type":"bad"}), "response_format.type"),
            (
                json!({"type":"json_schema","json_schema":{"name":"x","strict":"yes","schema":{}}}),
                "strict",
            ),
        ] {
            req["response_format"] = format;
            let (status, _, body) = http_request(
                &addr,
                "POST",
                "/v1/chat/completions",
                Some(&req.to_string()),
            );
            assert_eq!(status, 400);
            assert!(body.contains(needle));
        }
        thread.join().unwrap();
    }
}

#[test]
fn tool_loop_streaming_and_full_and_malformed() {
    struct Replies {
        step: usize,
        malformed: bool,
    }
    impl Backend for Replies {
        fn generate(
            &mut self,
            prompt: &[u32],
            _: &mut GenParams,
            emit: &mut dyn FnMut(crate::backend::TokenEvent<'_>) -> std::ops::ControlFlow<()>,
        ) -> Result<(), String> {
            if self.step == 1 {
                let prompt = PlaceholderTokenizer::new().decode(prompt);
                assert!(prompt.contains("sunny"));
            }
            let text = if self.malformed {
                "<tool_call>broken"
            } else if self.step == 0 {
                "Checking.<tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call>"
            } else {
                "It is sunny."
            };
            self.step += 1;
            for b in text.bytes() {
                if emit(crate::backend::TokenEvent { id: b as u32, logits: None }).is_break() { break; }
            }
            Ok(())
        }
    }
    for streaming in [false, true] {
        for choice in [
            json!("auto"),
            json!("none"),
            json!("required"),
            json!({"type":"function","function":{"name":"get_weather"}}),
        ] {
            for malformed in [false, true] {
                let cfg = ServerConfig {
                    port: 0,
                    model_id: "fixture".into(),
                    ..ServerConfig::from_env()
                };
                let server = bind(&cfg).unwrap();
                let addr = server.server_addr().to_ip().unwrap().to_string();
                let state = Arc::new(ServerState {
                    backend: Mutex::new(Box::new(Replies { step: 0, malformed })),
                    context_capacity: None,
                    tokenizer: ServerTokenizer::Placeholder(PlaceholderTokenizer::new()),
                    cfg,
                    embedder: None,
                    embedding_model: None,
                    embedding_only: false,
                });
                let thread = std::thread::spawn(move || {
                    for _ in 0..2 {
                        handle_request(server.recv().unwrap(), &state);
                    }
                });
                let mut req = json!({"messages":[{"role":"user","content":"Weather?"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}],"stream":streaming,"tool_choice":choice});
                let (status, _, body) = http_request(
                    &addr,
                    "POST",
                    "/v1/chat/completions",
                    Some(&req.to_string()),
                );
                assert_eq!(status, 200);
                let message = if streaming {
                    let frames: Vec<Value> = body
                        .split("\n\n")
                        .filter_map(|s| s.strip_prefix("data: "))
                        .filter(|s| *s != "[DONE]")
                        .map(|s| serde_json::from_str(s).unwrap())
                        .collect();
                    let calls: Vec<Value> = frames
                        .iter()
                        .filter_map(|f| f["choices"][0]["delta"]["tool_calls"].as_array())
                        .flatten()
                        .cloned()
                        .collect();
                    let content: String = frames
                        .iter()
                        .filter_map(|f| f["choices"][0]["delta"]["content"].as_str())
                        .collect();
                    assert_eq!(
                        frames.last().unwrap()["choices"][0]["finish_reason"],
                        if choice != "none" && !malformed {
                            "tool_calls"
                        } else {
                            "stop"
                        }
                    );
                    json!({"role":"assistant","content":content,"tool_calls":calls})
                } else {
                    let response: Value = serde_json::from_str(&body).unwrap();
                    assert_eq!(
                        response["choices"][0]["finish_reason"],
                        if choice != "none" && !malformed {
                            "tool_calls"
                        } else {
                            "stop"
                        }
                    );
                    response["choices"][0]["message"].clone()
                };
                if choice != "none" && !malformed {
                    assert_eq!(message["tool_calls"][0]["function"]["name"], "get_weather");
                    assert!(message["tool_calls"][0]["id"]
                        .as_str()
                        .unwrap()
                        .starts_with("call_"));
                    assert_eq!(message["content"], "Checking.");
                } else {
                    assert!(message["content"].as_str().unwrap().contains("<tool_call>"));
                }
                let id = message["tool_calls"][0]["id"]
                    .as_str()
                    .unwrap_or("call_fixture");
                req["messages"]
                    .as_array_mut()
                    .unwrap()
                    .push(message.clone());
                req["messages"]
                    .as_array_mut()
                    .unwrap()
                    .push(json!({"role":"tool","tool_call_id":id,"content":"sunny"}));
                let (status, _, body) = http_request(
                    &addr,
                    "POST",
                    "/v1/chat/completions",
                    Some(&req.to_string()),
                );
                assert_eq!(status, 200);
                let answer = if streaming {
                    body.split("\n\n")
                        .filter_map(|s| s.strip_prefix("data: "))
                        .filter_map(|s| serde_json::from_str::<Value>(s).ok())
                        .filter_map(|v| {
                            v["choices"][0]["delta"]["content"]
                                .as_str()
                                .map(str::to_owned)
                        })
                        .collect::<String>()
                } else {
                    serde_json::from_str::<Value>(&body).unwrap()["choices"][0]["message"]
                        ["content"]
                        .as_str()
                        .unwrap()
                        .to_owned()
                };
                assert!(answer.contains(if malformed { "broken" } else { "sunny" }));
                thread.join().unwrap();
            }
        }
    }
}

#[test]
fn mock_server_returns_constrained_json() {
    let addr = spawn_test_server(|_| {});
    for format in [
        json!({"type":"json_object"}),
        json!({"type":"json_schema","json_schema":{"name":"status","strict":true,"schema":{"type":"object","properties":{"ok":{"const":true}},"required":["ok"],"additionalProperties":false}}}),
    ] {
        let req =
            json!({"messages":[{"role":"user","content":"Return JSON"}],"response_format":format});
        let (status, _, body) = http_request(
            &addr,
            "POST",
            "/v1/chat/completions",
            Some(&req.to_string()),
        );
        assert_eq!(status, 200);
        let response: Value = serde_json::from_str(&body).unwrap();
        let value: Value = serde_json::from_str(
            response["choices"][0]["message"]["content"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
        assert!(value.is_object());
        assert_eq!(response["choices"][0]["finish_reason"], "stop");
        if format["type"] == "json_schema" {
            assert_eq!(value, json!({"ok":true}));
        }
    }
}

#[test]
fn harmony_commentary_and_malformed_calls_remain_visible() {
    use crate::tokenizer::ReasoningFormat;
    use crate::tool_calls::{Format, Tools};
    let cfg = Tools::from_request(
        &json!({"tools":[{"type":"function","function":{"name":"get_weather"}}]}),
    )
    .unwrap();
    let r = Reasoning::for_prompt(ReasoningFormat::Harmony, "");
    let lead="<|channel|>analysis<|message|>Think.<|end|><|start|>assistant<|channel|>commentary<|message|>Checking.<|end|><|start|>assistant to=functions.get_weather<|channel|>commentary json<|message|>";
    let (message, called) = assistant_message(
        &format!("{lead}{{\"city\":\"Paris\"}}"),
        Some(&r),
        Some(&(Format::Harmony, cfg.clone())),
    );
    assert!(called);
    assert_eq!(message["content"], "Checking.");
    assert_eq!(message["reasoning_content"], "Think.");
    let (message, called) = assistant_message(
        &format!("{lead}{{\"city\":"),
        Some(&r),
        Some(&(Format::Harmony, cfg)),
    );
    assert!(!called);
    assert!(message["content"].as_str().unwrap().contains("{\"city\":"));
    assert_eq!(message["reasoning_content"], "Think.");
}
