#[cfg(unix)]
#[test]
fn speculative_bridge_stream_stop_and_reuse_match_plain_http() {
    use crate::bench::reference::speculation::tests::Fixture;
    use crate::speculative::SpeculativeMode;
    use tokenizers::{decoders::fuse::Fuse, models::bpe::BPE};
    for mode in [SpeculativeMode::Off, SpeculativeMode::Lookup] {
        let fixture = Fixture::new(json!({"sequence":[1,2]}));
        let vocab: tokenizers::models::bpe::Vocab = ["\"", "a", "b", "c", " ", "{", "}", "<eos>"]
            .into_iter()
            .enumerate()
            .map(|(i, s)| (s.to_string(), i as u32))
            .collect();
        let mut tk = tokenizers::Tokenizer::new(
            BPE::builder()
                .vocab_and_merges(vocab, vec![])
                .build()
                .unwrap(),
        );
        tk.with_decoder(Some(Fuse::default()));
        tk.save(fixture.dir.join("tokenizer.json"), false).unwrap();
        std::fs::write(
            fixture.dir.join("tokenizer_config.json"),
            json!({
                "eos_token":"<eos>","chat_template":"{{ messages[0].content }}"
            })
            .to_string(),
        )
        .unwrap();
        let tokenizer = Tokenizer::load(&fixture.dir).unwrap();
        let backend = fixture.load(32);
        let cfg = ServerConfig {
            port: 0,
            defaults: GenDefaults {
                speculative: mode,
                ..GenDefaults::default()
            },
            ..ServerConfig::from_env()
        };
        let server = bind(&cfg).unwrap();
        let addr = server.server_addr().to_ip().unwrap().to_string();
        let state = Arc::new(ServerState {
            backend: Mutex::new(Box::new(backend)),
            context_capacity: Some(32),
            tokenizer: ServerTokenizer::Real(tokenizer),
            cfg,
            embedder: None,
            embedding_model: None,
            embedding_only: false,
        });
        let worker = std::thread::spawn(move || {
            for _ in 0..5 {
                handle_request(server.recv().unwrap(), &state);
            }
        });
        for stream in [false, true] {
            let req = json!({"prompt":[1,2,1,2],"max_tokens":12,"stream":stream,"temperature":0});
            let (status, _, body) =
                http_request(&addr, "POST", "/v1/completions", Some(&req.to_string()));
            assert_eq!(status, 200, "{body}");
            assert_eq!(
                text_and_finish(&body, stream, true),
                ("abababababab".into(), "length".into())
            );
        }
        for stream in [false, true] {
            let req = json!({"prompt":[1,2,1,2],"max_tokens":12,"stream":stream,"temperature":0,"stop":"bab"});
            let (status, _, body) =
                http_request(&addr, "POST", "/v1/completions", Some(&req.to_string()));
            assert_eq!(status, 200, "{body}");
            assert_eq!(
                text_and_finish(&body, stream, true),
                ("a".into(), "stop".into())
            );
            let last = fixture.log().last().unwrap().clone();
            assert!(last["pending"].is_null());
            // Four output tokens detect "bab"; the fourth must remain unrun.
            assert_eq!(last["committed"], json!([1, 2, 1, 2, 1, 2, 1]));
        }
        let follow = json!({"prompt":[1,2,1,2,1,2,1,2,1,2],"max_tokens":2});
        let (status, _, body) =
            http_request(&addr, "POST", "/v1/completions", Some(&follow.to_string()));
        assert_eq!(status, 200, "{body}");
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["usage"]["prompt_tokens_details"]["cached_tokens"], 7);
        worker.join().unwrap();
    }
}
