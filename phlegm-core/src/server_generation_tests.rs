fn frames(body: &str) -> Vec<Value> {
    body.split("\n\n")
        .filter_map(|s| s.strip_prefix("data: "))
        .filter_map(|s| serde_json::from_str(s).ok())
        .collect()
}
fn text_and_finish(body: &str, stream: bool, legacy: bool) -> (String, String) {
    if stream {
        let f = frames(body);
        let text = f
            .iter()
            .filter_map(|v| {
                if legacy {
                    v["choices"][0]["text"].as_str()
                } else {
                    v["choices"][0]["delta"]["content"].as_str()
                }
            })
            .collect();
        let finish = f
            .iter()
            .find_map(|v| v["choices"][0]["finish_reason"].as_str())
            .unwrap()
            .to_owned();
        (text, finish)
    } else {
        let v: Value = serde_json::from_str(body).unwrap();
        (
            if legacy {
                v["choices"][0]["text"].as_str()
            } else {
                v["choices"][0]["message"]["content"].as_str()
            }
            .unwrap()
            .into(),
            v["choices"][0]["finish_reason"].as_str().unwrap().into(),
        )
    }
}

#[test]
fn stop_strings_hold_partial_suffixes_flush_and_validate_on_both_routes() {
    let addr = spawn_test_server(|_| {});
    for legacy in [false, true] {
        for stream in [false, true] {
            let route = if legacy {
                "/v1/completions"
            } else {
                "/v1/chat/completions"
            };
            let mut req = json!({"messages":[{"role":"user","content":"x"}],"prompt":"x","stream":stream,"max_tokens":100});
            for (stop, limit, expected, finish) in [
                (json!(" npu]"), 100, "[mock", "stop"),
                (json!(["never", "mock", "npu"]), 100, "[", "stop"),
                (json!("[mockX"), 5, "[mock", "length"),
                (json!("[mock"), 5, "", "stop"),
                (json!(["[mockZ", "[mock"]), 100, "", "stop"),
            ] {
                req["stop"] = stop;
                req["max_tokens"] = json!(limit);
                let (status, _, body) = http_request(&addr, "POST", route, Some(&req.to_string()));
                assert_eq!(status, 200, "{body}");
                assert_eq!(
                    text_and_finish(&body, stream, legacy),
                    (expected.into(), finish.into())
                );
                if stream {
                    // No unsafe fragment may appear in any intermediate content delta.
                    let mut all = String::new();
                    for f in frames(&body) {
                        if let Some(t) = if legacy {
                            f["choices"][0]["text"].as_str()
                        } else {
                            f["choices"][0]["delta"]["content"].as_str()
                        } {
                            all.push_str(t);
                            assert!(expected.starts_with(&all));
                        }
                    }
                }
            }
            for bad in [
                json!(""),
                json!(["ok", 2]),
                json!(["a", "b", "c", "d", "e"]),
                json!({}),
                json!(false),
            ] {
                req["stop"] = bad;
                assert_eq!(
                    http_request(&addr, "POST", route, Some(&req.to_string())).0,
                    400
                );
            }
        }
    }
}

#[test]
fn legacy_completions_prompt_shapes_sampling_usage_and_errors() {
    let addr = spawn_test_server(|_| {});
    for prompt in [
        json!("abc"),
        json!(["abc"]),
        json!([97, 98, 99]),
        json!([[97, 98, 99]]),
    ] {
        for stream in [false, true] {
            let req = json!({"prompt":prompt,"max_tokens":3,"temperature":0.7,"top_p":0.9,"top_k":3,"presence_penalty":1,"frequency_penalty":0.5,"seed":7,"stream":stream,"stream_options":{"include_usage":true},"echo":false});
            let (status, _, body) =
                http_request(&addr, "POST", "/v1/completions", Some(&req.to_string()));
            assert_eq!(status, 200, "{body}");
            assert_eq!(
                text_and_finish(&body, stream, true),
                ("[mo".into(), "length".into())
            );
            let v = if stream {
                let fs = frames(&body);
                assert!(fs.iter().all(|f| f["object"] == "text_completion"));
                assert!(fs.iter().all(|f| f["choices"][0].get("delta").is_none()));
                fs.last().unwrap().clone()
            } else {
                serde_json::from_str(&body).unwrap()
            };
            assert_eq!(v["usage"]["prompt_tokens"], 3, "no chat template");
            assert_eq!(v["usage"]["completion_tokens"], 3);
        }
    }
    for req in [
        json!({}),
        json!({"prompt":[]}),
        json!({"prompt":""}),
        json!({"prompt":["a","b"]}),
        json!({"prompt":[[1],[2]]}),
        json!({"prompt":[-1]}),
        json!({"prompt":[257]}),
        json!({"prompt":[1.5]}),
        json!({"prompt":[4294967296u64]}),
        json!({"prompt":"a","suffix":"x"}),
        json!({"prompt":"a","echo":true}),
        json!({"prompt":"a","n":2}),
        json!({"prompt":"a","best_of":2}),
        json!({"prompt":"a","max_tokens":0}),
        json!({"prompt":"a","temperature":"bad"}),
    ] {
        let (status, _, body) =
            http_request(&addr, "POST", "/v1/completions", Some(&req.to_string()));
        assert_eq!(status, 400, "{req}: {body}");
    }
}

#[test]
fn logprobs_shapes_limits_raw_distribution_and_stop_exclusion() {
    let addr = spawn_test_server(|_| {});
    for legacy in [false, true] {
        for stream in [false, true] {
            let route = if legacy {
                "/v1/completions"
            } else {
                "/v1/chat/completions"
            };
            let mut req = json!({"prompt":"abc","messages":[{"role":"user","content":"abc"}],"stream":stream,"max_tokens":10,"stop":"mock","logprobs":if legacy {json!(2)}else{json!(true)}});
            if !legacy {
                req["top_logprobs"] = json!(2);
            }
            let (status, _, body) = http_request(&addr, "POST", route, Some(&req.to_string()));
            assert_eq!(status, 200, "{body}");
            assert_eq!(
                text_and_finish(&body, stream, legacy),
                ("[".into(), "stop".into())
            );
            let values: Vec<Value> = if stream {
                frames(&body)
                    .iter()
                    .filter_map(|f| f["choices"][0].get("logprobs"))
                    .filter(|l| !l.is_null())
                    .cloned()
                    .collect()
            } else {
                vec![
                    serde_json::from_str::<Value>(&body).unwrap()["choices"][0]["logprobs"].clone(),
                ]
            };
            let entries: Vec<Value> = if legacy {
                values
                    .iter()
                    .flat_map(|v| v["tokens"].as_array().unwrap().clone())
                    .collect()
            } else {
                values
                    .iter()
                    .flat_map(|v| v["content"].as_array().unwrap().clone())
                    .collect()
            };
            assert_eq!(entries.len(), 1, "stop tokens are excluded: {body}");
            let expected = -(1.0 + 256.0 * (-10.0f64).exp()).ln();
            if legacy {
                assert_eq!(entries[0], "[");
                assert!((values[0]["token_logprobs"][0].as_f64().unwrap() - expected).abs() < 1e-6);
                assert_eq!(values[0]["text_offset"], json!([0]));
            } else {
                assert_eq!(entries[0]["token"], "[");
                assert_eq!(entries[0]["bytes"], json!([91]));
                assert_eq!(entries[0]["top_logprobs"].as_array().unwrap().len(), 2);
                assert!((entries[0]["logprob"].as_f64().unwrap() - expected).abs() < 1e-6);
            }
        }
    }
    for extra in [
        json!({"logprobs":1}),
        json!({"logprobs":true,"top_logprobs":21}),
        json!({"top_logprobs":0}),
        json!({"logprobs":false,"top_logprobs":2}),
        json!({"logprobs":true,"top_logprobs":-1}),
        json!({"logprobs":true,"top_logprobs":1.2}),
    ] {
        let mut req = json!({"messages":[{"role":"user","content":"x"}]});
        req.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        assert_eq!(
            http_request(
                &addr,
                "POST",
                "/v1/chat/completions",
                Some(&req.to_string())
            )
            .0,
            400
        );
    }
    for bad in [json!(true), json!(-1), json!(6), json!(1.2)] {
        assert_eq!(
            http_request(
                &addr,
                "POST",
                "/v1/completions",
                Some(&json!({"prompt":"x","logprobs":bad}).to_string())
            )
            .0,
            400
        );
    }
    for n in [0, 20] {
        let (_,_,body)=http_request(&addr,"POST","/v1/chat/completions",Some(&json!({"messages":[{"role":"user","content":"x"}],"logprobs":true,"top_logprobs":n,"max_tokens":1}).to_string()));
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(
            v["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
                .as_array()
                .unwrap()
                .len(),
            n as usize
        );
    }
}

struct ByteReply {
    text: String,
    seen: Arc<AtomicU64>,
}
impl Backend for ByteReply {
    fn generate(
        &mut self,
        _: &[u32],
        params: &mut GenParams,
        emit: &mut dyn FnMut(crate::backend::TokenEvent<'_>) -> std::ops::ControlFlow<()>,
    ) -> Result<(), String> {
        for b in self.text.bytes().take(params.max_tokens) {
            self.seen.fetch_add(1, Ordering::Relaxed);
            let mut logits = vec![0.; 257];
            logits[b as usize] = 10.;
            if emit(crate::backend::TokenEvent {
                id: b as u32,
                logits: Some(&logits),
            })
            .is_break()
            {
                break;
            }
        }
        Ok(())
    }
}
fn scripted_request(text: &str, req: Value, reasoning: Option<Reasoning>) -> (String, u64) {
    let cfg = ServerConfig {
        port: 0,
        ..ServerConfig::from_env()
    };
    let server = bind(&cfg).unwrap();
    let addr = server.server_addr().to_ip().unwrap().to_string();
    let seen = Arc::new(AtomicU64::new(0));
    let backend = ByteReply {
        text: text.into(),
        seen: seen.clone(),
    };
    let state = Arc::new(ServerState {
        backend: Mutex::new(Box::new(backend)),
        context_capacity: None,
        tokenizer: ServerTokenizer::Placeholder(PlaceholderTokenizer::new()),
        cfg,
        embedder: None,
        embedding_model: None,
        embedding_only: false,
    });
    let client_req = req.clone();
    let worker = std::thread::spawn(move || {
        let request = server.recv().unwrap();
        if let Some(r) = reasoning {
            let options = generation::Options::parse(&req, false).unwrap();
            generation::respond(
                request,
                &state,
                vec![1],
                GenParams {
                    max_tokens: 1000,
                    sampler: Sampler::new(0., 0, 1., 1., None),
                    stop_ids: Default::default(),
                    reuse_mark: None,
                },
                "fixture".into(),
                Some(r),
                None,
                options,
            );
        } else {
            handle_request(request, &state)
        }
    });
    let (_, _, body) = http_request(
        &addr,
        "POST",
        "/v1/chat/completions",
        Some(&client_req.to_string()),
    );
    worker.join().unwrap();
    (body, seen.load(Ordering::Relaxed))
}
#[test]
fn stop_composes_with_reasoning_and_stops_backend_immediately() {
    use crate::tokenizer::ReasoningFormat;
    let text = "<think>why</think>answer END ignored";
    for stream in [false, true] {
        let req = json!({"stream":stream,"stop":" END"});
        let (body, seen) = scripted_request(
            text,
            req,
            Some(Reasoning::for_prompt(
                ReasoningFormat::Tagged {
                    open: "<think>".into(),
                    close: "</think>".into(),
                },
                "",
            )),
        );
        assert_eq!(seen, "<think>why</think>answer END".len() as u64);
        assert_eq!(
            text_and_finish(&body, stream, false),
            ("answer".into(), "stop".into())
        );
        if stream {
            assert_eq!(
                frames(&body)
                    .iter()
                    .filter_map(|f| f["choices"][0]["delta"]["reasoning_content"].as_str())
                    .collect::<String>(),
                "why"
            );
        }
    }
}

#[test]
fn all_tool_markers_hold_partial_prefixes() {
    use crate::tool_calls::Format;
    for (format, marker) in [
        (Format::Function, "<tool_call>"),
        (Format::Tagged, "<tool_call>"),
        (Format::Gemma, "<|tool_call>"),
        (Format::Hy, "<tool_calls>"),
        (Format::Hy, "<｜tool▁calls▁begin｜>"),
        (Format::Lfm, "<|tool_call_start|>"),
        (Format::Lfm, "["),
    ] {
        for (end, _) in marker.char_indices().skip(1) {
            let s = format!("Checking.{}", &marker[..end]);
            assert_eq!(format.stream_prefix(&s), "Checking.", "{format:?}: {s}");
        }
        assert_eq!(
            format.stream_prefix(&format!("Checking.{marker}payload")),
            "Checking."
        );
        assert_eq!(format.stream_prefix("Checking."), "Checking.");
    }
    let prefix = "<|channel|>commentary<|message|>Checking.<|end|>";
    assert_eq!(
        Format::Harmony.stream_prefix(&format!(
            "{prefix}<|start|>assistant to=functions.x<|channel|>commentary<|message|>{{}}"
        )),
        prefix
    );
}

#[test]
fn tool_content_reaches_socket_before_backend_finishes() {
    use std::io::{Read, Write};
    use std::net::TcpStream;
    struct Gated(std::sync::mpsc::Receiver<()>);
    impl Backend for Gated {
        fn generate(
            &mut self,
            _: &[u32],
            _: &mut GenParams,
            emit: &mut dyn FnMut(crate::backend::TokenEvent<'_>) -> std::ops::ControlFlow<()>,
        ) -> Result<(), String> {
            for b in b"Checking." {
                if emit(crate::backend::TokenEvent {
                    id: *b as u32,
                    logits: None,
                })
                .is_break()
                {
                    return Ok(());
                }
            }
            self.0
                .recv_timeout(std::time::Duration::from_secs(3))
                .map_err(|_| "client did not receive content during generation")?;
            for b in b"<tool_call><function=get_status></function></tool_call>" {
                if emit(crate::backend::TokenEvent {
                    id: *b as u32,
                    logits: None,
                })
                .is_break()
                {
                    break;
                }
            }
            Ok(())
        }
    }
    let (tx, rx) = std::sync::mpsc::channel();
    let cfg = ServerConfig {
        port: 0,
        ..ServerConfig::from_env()
    };
    let server = bind(&cfg).unwrap();
    let addr = server.server_addr().to_ip().unwrap().to_string();
    std::thread::spawn(move || run(server, Box::new(Gated(rx)), cfg));
    let body=json!({"messages":[{"role":"user","content":"x"}],"stream":true,"tools":[{"type":"function","function":{"name":"get_status"}}]}).to_string();
    let mut socket = TcpStream::connect(addr).unwrap();
    socket
        .set_read_timeout(Some(std::time::Duration::from_secs(5)))
        .unwrap();
    write!(socket,"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len()).unwrap();
    let mut received = String::new();
    let mut acknowledged = false;
    loop {
        let mut buf = [0; 4096];
        let n = socket.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        received.push_str(std::str::from_utf8(&buf[..n]).unwrap());
        if !acknowledged && received.contains("\"content\":\".\"") {
            tx.send(()).unwrap();
            acknowledged = true;
        }
    }
    assert!(acknowledged);
    assert!(
        received.contains("\"finish_reason\":\"tool_calls\""),
        "{received}"
    );
    assert!(!received.contains("client did not receive"));
}

#[test]
fn unicode_stop_and_incomplete_prefix_flush_are_exact() {
    for (stop, expected, seen) in [("🙂", "A", 5), ("🙂AX", "A🙂A", 6)] {
        for stream in [false, true] {
            let req = json!({"messages":[{"role":"user","content":"x"}],"stop":stop,"stream":stream,"logprobs":true});
            let (body, count) = scripted_request("A🙂A", req, None);
            assert_eq!(count, seen);
            assert_eq!(
                text_and_finish(&body, stream, false),
                (expected.into(), "stop".into())
            );
        }
    }
}

#[test]
fn stop_inside_tool_call_uses_plain_text_fallback_and_stops() {
    let text = "Checking.<tool_call><function=get_status></function></tool_call> ignored";
    for stream in [false, true] {
        let req = json!({"messages":[{"role":"user","content":"x"}],"tools":[{"type":"function","function":{"name":"get_status"}}],"stream":stream,"stop":"get_status"});
        let (body, seen) = scripted_request(text, req, None);
        assert_eq!(
            seen,
            "Checking.<tool_call><function=get_status".len() as u64
        );
        assert_eq!(
            text_and_finish(&body, stream, false),
            ("Checking.<tool_call><function=".into(), "stop".into())
        );
    }
}

#[test]
fn raw_logprobs_ignore_penalties_temperature_filters_and_schema_mask() {
    struct Constrained;
    impl Backend for Constrained {
        fn generate(
            &mut self,
            _: &[u32],
            p: &mut GenParams,
            emit: &mut dyn FnMut(crate::backend::TokenEvent<'_>) -> std::ops::ControlFlow<()>,
        ) -> Result<(), String> {
            for pos in 0..p.max_tokens {
                let mut logits = vec![0.; 257];
                logits[b'x' as usize] = 10.;
                logits[if pos < 3 { b"\"A\""[pos] as usize } else { 256 }] = 1.;
                let id = p.sampler.try_sample(&logits, &[b'x' as i64])? as u32;
                if p.stop_ids.contains(&id) {
                    break;
                }
                if emit(crate::backend::TokenEvent {
                    id,
                    logits: Some(&logits),
                })
                .is_break()
                {
                    break;
                }
            }
            Ok(())
        }
    }
    let cfg = ServerConfig {
        port: 0,
        ..ServerConfig::from_env()
    };
    let server = bind(&cfg).unwrap();
    let addr = server.server_addr().to_ip().unwrap().to_string();
    std::thread::spawn(move || run(server, Box::new(Constrained), cfg));
    let req = json!({"messages":[{"role":"user","content":"x"}],"response_format":{"type":"json_schema","json_schema":{"name":"a","schema":{"const":"A"}}},"temperature":0.7,"top_k":1,"top_p":0.5,"repetition_penalty":2,"presence_penalty":2,"logprobs":true,"top_logprobs":1,"max_tokens":8});
    let (_, _, body) = http_request(
        &addr,
        "POST",
        "/v1/chat/completions",
        Some(&req.to_string()),
    );
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], "\"A\"");
    let logs = v["choices"][0]["logprobs"]["content"].as_array().unwrap();
    assert_eq!(logs.len(), 3);
    for l in logs {
        assert_eq!(l["top_logprobs"][0]["token"], "x");
        assert!(
            (l["logprob"].as_f64().unwrap()
                - (-9. - (1. + (-9.0f64).exp() + 255. * (-10.0f64).exp()).ln()))
            .abs()
                < 1e-6
        );
    }
}

#[test]
fn real_token_logprobs_preserve_partial_utf8_bytes_and_intra_token_stop() {
    use tokenizers::AddedToken;
    let (dir, _) = crate::tokenizer::tests::fixture(false, false);
    let path = dir.join("tokenizer.json");
    let mut t = tokenizers::Tokenizer::from_file(&path).unwrap();
    t.add_tokens([AddedToken::from("beforeSTOPafter", false)])
        .unwrap();
    t.save(&path, false).unwrap();
    let tokenizer = Tokenizer::load(&dir).unwrap();
    let merged = t.token_to_id("beforeSTOPafter").unwrap();
    struct Ids(Vec<u32>);
    impl Backend for Ids {
        fn generate(
            &mut self,
            _: &[u32],
            _: &mut GenParams,
            emit: &mut dyn FnMut(crate::backend::TokenEvent<'_>) -> std::ops::ControlFlow<()>,
        ) -> Result<(), String> {
            for &id in &self.0 {
                let mut logits = vec![0.; 8];
                logits[id as usize] = 2.;
                if emit(crate::backend::TokenEvent {
                    id,
                    logits: Some(&logits),
                })
                .is_break()
                {
                    break;
                }
            }
            Ok(())
        }
    }
    // Tokenize a string spanning a UTF-8 character into its four fallback bytes.
    for stream in [false, true] {
        for (ids, stop, expected) in [
            (vec![3, 4, 5, 6, 2], Value::Null, "🙂A"),
            (vec![merged, 2], json!("STOP"), "before"),
        ] {
            let cfg = ServerConfig {
                port: 0,
                ..ServerConfig::from_env()
            };
            let server = bind(&cfg).unwrap();
            let addr = server.server_addr().to_ip().unwrap().to_string();
            let tokenizer = Tokenizer::load(&dir).unwrap();
            let state = Arc::new(ServerState {
                backend: Mutex::new(Box::new(Ids(ids))),
                context_capacity: None,
                tokenizer: ServerTokenizer::Real(tokenizer),
                cfg,
                embedder: None,
                embedding_model: None,
                embedding_only: false,
            });
            let worker = std::thread::spawn(move || handle_request(server.recv().unwrap(), &state));
            let req = json!({"messages":[{"role":"user","content":"A"}],"logprobs":true,"stream":stream,"stop":stop});
            let (_, _, body) = http_request(
                &addr,
                "POST",
                "/v1/chat/completions",
                Some(&req.to_string()),
            );
            worker.join().unwrap();
            assert_eq!(
                text_and_finish(&body, stream, false),
                (expected.into(), "stop".into())
            );
            if stop.is_null() {
                let logs: Vec<Value> = if stream {
                    frames(&body)
                        .iter()
                        .filter_map(|f| f["choices"][0]["logprobs"]["content"].as_array())
                        .flatten()
                        .cloned()
                        .collect()
                } else {
                    serde_json::from_str::<Value>(&body).unwrap()["choices"][0]["logprobs"]
                        ["content"]
                        .as_array()
                        .unwrap()
                        .clone()
                };
                let bytes: Vec<u8> = logs
                    .iter()
                    .flat_map(|l| l["bytes"].as_array().unwrap())
                    .map(|b| b.as_u64().unwrap() as u8)
                    .collect();
                assert_eq!(bytes, "🙂A".as_bytes());
                assert_eq!(logs.len(), 5);
                assert_eq!(logs[0]["bytes"], json!([240]));
            }
        }
    }
    drop(tokenizer);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn six_cpu_protocol_fixtures_stream_with_reasoning_and_validated_calls() {
    use crate::tokenizer::ReasoningFormat;
    use crate::tool_calls::{Format, Tools};
    for (format, fixture) in [
        (
            Format::Function,
            include_str!("../tests/fixtures/tool_calls/KAT-Coder-V2.5-Dev-Q4pure-NPU2.json"),
        ),
        (
            Format::Tagged,
            include_str!("../tests/fixtures/tool_calls/flint-NPU2.json"),
        ),
        (
            Format::Gemma,
            include_str!("../tests/fixtures/tool_calls/gemma-4-26B-A4B-it-NPU2.json"),
        ),
        (
            Format::Hy,
            include_str!("../tests/fixtures/tool_calls/Hy-Memory-30B-A3B-NPU2.json"),
        ),
        (
            Format::Lfm,
            include_str!("../tests/fixtures/tool_calls/LFM2.5-24B-A2B-NPU2.json"),
        ),
        (
            Format::Harmony,
            include_str!("../tests/fixtures/tool_calls/gpt-oss-20b-NPU2.json"),
        ),
    ] {
        let fixture: Value = serde_json::from_str(fixture).unwrap();
        let output = fixture["output"].as_str().unwrap();
        let reasoning = if format == Format::Harmony {
            Some(Reasoning::for_prompt(ReasoningFormat::Harmony, ""))
        } else if output.contains("</think>") {
            Some(Reasoning::for_prompt(
                ReasoningFormat::Tagged {
                    open: "<think>".into(),
                    close: "</think>".into(),
                },
                if output.starts_with("<think>") {
                    ""
                } else {
                    "<think>"
                },
            ))
        } else {
            None
        };
        let req = json!({"stream":true,"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]});
        let tools = Tools::from_request(&req).unwrap();
        let expected =
            assistant_message(output, reasoning.as_ref(), Some(&(format, tools.clone()))).0;
        let cfg = ServerConfig {
            port: 0,
            ..ServerConfig::from_env()
        };
        let server = bind(&cfg).unwrap();
        let addr = server.server_addr().to_ip().unwrap().to_string();
        let backend = ByteReply {
            text: output.into(),
            seen: Arc::new(AtomicU64::new(0)),
        };
        let state = Arc::new(ServerState {
            backend: Mutex::new(Box::new(backend)),
            context_capacity: None,
            tokenizer: ServerTokenizer::Placeholder(PlaceholderTokenizer::new()),
            cfg,
            embedder: None,
            embedding_model: None,
            embedding_only: false,
        });
        let worker = std::thread::spawn(move || {
            generation::respond(
                server.recv().unwrap(),
                &state,
                vec![1],
                GenParams {
                    max_tokens: 1000,
                    sampler: Sampler::new(0., 0, 1., 1., None),
                    stop_ids: Default::default(),
                    reuse_mark: None,
                },
                "fixture".into(),
                reasoning,
                Some((format, tools)),
                generation::Options::parse(&req, false).unwrap(),
            )
        });
        let (_, _, body) = http_request(&addr, "POST", "/v1/chat/completions", Some("{}"));
        worker.join().unwrap();
        let frames = frames(&body);
        assert!(
            !frames.iter().any(|f| f.get("error").is_some()),
            "{format:?}: {body}"
        );
        for field in ["content", "reasoning_content"] {
            let text = frames
                .iter()
                .filter_map(|f| f["choices"][0]["delta"][field].as_str())
                .collect::<String>();
            assert_eq!(
                text.trim(),
                expected[field].as_str().unwrap_or("").trim(),
                "{format:?}: {body}"
            );
        }
        let calls: Vec<Value> = frames
            .iter()
            .filter_map(|f| f["choices"][0]["delta"]["tool_calls"].as_array())
            .flatten()
            .cloned()
            .collect();
        assert_eq!(calls.len(), 1, "{format:?}: {body}");
        assert_eq!(calls[0]["index"], 0);
        assert_eq!(calls[0]["function"], expected["tool_calls"][0]["function"]);
        assert_eq!(
            frames.last().unwrap()["choices"][0]["finish_reason"],
            "tool_calls"
        );
    }
}

#[test]
fn legacy_unicode_offsets_and_bytes_preserve_split_characters() {
    for stream in [false, true] {
        let cfg = ServerConfig {
            port: 0,
            ..ServerConfig::from_env()
        };
        let server = bind(&cfg).unwrap();
        let addr = server.server_addr().to_ip().unwrap().to_string();
        let seen = Arc::new(AtomicU64::new(0));
        let backend = ByteReply {
            text: "🙂A".into(),
            seen,
        };
        let state = Arc::new(ServerState {
            backend: Mutex::new(Box::new(backend)),
            context_capacity: None,
            tokenizer: ServerTokenizer::Placeholder(PlaceholderTokenizer::new()),
            cfg,
            embedder: None,
            embedding_model: None,
            embedding_only: false,
        });
        let worker = std::thread::spawn(move || handle_request(server.recv().unwrap(), &state));
        let req = json!({"prompt":"x","logprobs":0,"stream":stream});
        let (status, _, body) =
            http_request(&addr, "POST", "/v1/completions", Some(&req.to_string()));
        worker.join().unwrap();
        assert_eq!(status, 200, "{body}");
        assert_eq!(
            text_and_finish(&body, stream, true),
            ("🙂A".into(), "stop".into())
        );
        let logs: Vec<Value> = if stream {
            frames(&body)
                .into_iter()
                .map(|v| v["choices"][0]["logprobs"].clone())
                .filter(|v| !v.is_null())
                .collect()
        } else {
            vec![serde_json::from_str::<Value>(&body).unwrap()["choices"][0]["logprobs"].clone()]
        };
        assert_eq!(
            logs.iter()
                .flat_map(|v| v["text_offset"].as_array().unwrap())
                .cloned()
                .collect::<Vec<_>>(),
            vec![json!(0), json!(0), json!(0), json!(0), json!(1)]
        );
        let bytes: Vec<u8> = logs
            .iter()
            .flat_map(|v| v["bytes"].as_array().unwrap())
            .flat_map(|v| v.as_array().unwrap())
            .map(|v| v.as_u64().unwrap() as u8)
            .collect();
        assert_eq!(bytes, "🙂A".as_bytes());
    }
}
