//! Shared chat/legacy generation, stop filtering and opt-in model log probabilities.
use super::*;
use crate::backend::TokenEvent;
use std::ops::ControlFlow;

pub(super) struct Options {
    pub legacy: bool,
    pub stream: bool,
    pub include_usage: bool,
    pub stops: Vec<String>,
    pub logprobs: Option<usize>,
}
impl Options {
    pub fn parse(req: &Value, legacy: bool) -> Result<Self, String> {
        let field = |key: &str| req.get(key).filter(|v| !v.is_null());
        for key in ["max_tokens", "max_completion_tokens"] {
            if field(key)
                .is_some_and(|v| !v.as_u64().is_some_and(|n| n > 0 && n <= usize::MAX as u64))
            {
                return Err(format!("{key} must be a positive integer"));
            }
        }
        for key in ["temperature", "top_p"] {
            if field(key).is_some_and(|v| !v.is_number()) {
                return Err(format!("{key} must be numeric"));
            }
        }
        if field("n").is_some_and(|n| n.as_u64() != Some(1)) {
            return Err("only n=1 is supported".into());
        }
        if field("stream").is_some_and(|v| !v.is_boolean()) {
            return Err("stream must be boolean".into());
        }
        if let Some(v) = field("stream_options") {
            if !v.is_object() || v.get("include_usage").is_some_and(|b| !b.is_boolean()) {
                return Err("stream_options must be an object with boolean include_usage".into());
            }
        }
        let stops = match field("stop") {
            None => vec![],
            Some(Value::String(s)) => vec![s.clone()],
            Some(Value::Array(a)) if a.len() <= 4 => a
                .iter()
                .map(|s| {
                    s.as_str()
                        .map(str::to_owned)
                        .ok_or("stop entries must be strings".to_owned())
                })
                .collect::<Result<Vec<_>, _>>()?,
            _ => return Err("stop must be a string or an array of at most four strings".into()),
        };
        if stops.iter().any(String::is_empty) {
            return Err("stop strings must be nonempty".into());
        }
        let logprobs = if legacy {
            if field("suffix").is_some() {
                return Err("suffix (insertion) is unsupported".into());
            }
            if field("echo").is_some_and(|v| v != false) {
                return Err("echo is unsupported; use echo=false".into());
            }
            if field("best_of").is_some_and(|v| v.as_u64() != Some(1)) {
                return Err("only best_of=1 is supported".into());
            }
            if field("top_logprobs").is_some() {
                return Err("use integer logprobs on completions".into());
            }
            field("logprobs")
                .map(|v| count(v, 5, "logprobs"))
                .transpose()?
        } else {
            if field("logprobs").is_some_and(|v| !v.is_boolean()) {
                return Err("logprobs must be boolean on chat completions".into());
            }
            let top = field("top_logprobs")
                .map(|v| count(v, 20, "top_logprobs"))
                .transpose()?;
            if top.is_some() && field("logprobs") != Some(&Value::Bool(true)) {
                return Err("top_logprobs requires logprobs=true".into());
            }
            (field("logprobs") == Some(&Value::Bool(true))).then_some(top.unwrap_or(0))
        };
        Ok(Self {
            legacy,
            stream: field("stream").and_then(Value::as_bool).unwrap_or(false),
            include_usage: field("stream_options")
                .and_then(|v| v.get("include_usage"))
                .and_then(Value::as_bool)
                .unwrap_or(false),
            stops,
            logprobs,
        })
    }
}
fn count(v: &Value, max: u64, name: &str) -> Result<usize, String> {
    v.as_u64()
        .filter(|n| *n <= max)
        .map(|n| n as usize)
        .ok_or_else(|| format!("{name} must be an integer from 0 to {max}"))
}

pub(super) fn prompt_ids(req: &Value, tokenizer: &ServerTokenizer) -> Result<Vec<u32>, String> {
    fn ids(v: &[Value]) -> Result<Vec<u32>, String> {
        v.iter()
            .map(|v| {
                v.as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .ok_or("prompt token IDs must be unsigned 32-bit integers".into())
            })
            .collect()
    }
    let value = req.get("prompt").ok_or("prompt is required")?;
    let result = match value {
        Value::String(s) => match tokenizer {
            ServerTokenizer::Real(t) => t.encode(s)?,
            ServerTokenizer::Placeholder(t) => t.encode(s),
        },
        Value::Array(a) if a.iter().all(Value::is_number) => ids(a)?,
        Value::Array(a) if a.len() == 1 => match &a[0] {
            Value::String(s) => return prompt_ids(&json!({"prompt":s}), tokenizer),
            Value::Array(a) => ids(a)?,
            _ => return Err("prompt must contain strings or token-ID arrays".into()),
        },
        Value::Array(_) => return Err("prompt batching is unsupported; provide one prompt".into()),
        _ => {
            return Err(
                "prompt must be a string, token IDs, or a single-element prompt array".into(),
            )
        }
    };
    if result.is_empty() {
        return Err("prompt must encode at least one token".into());
    }
    if result.iter().any(|&id| match tokenizer {
        ServerTokenizer::Real(t) => t.id_to_token(id).is_none(),
        ServerTokenizer::Placeholder(_) => id > 256,
    }) {
        return Err("prompt contains an out-of-vocabulary token ID".into());
    }
    Ok(result)
}

/// Length of the longest suffix that could still become a marker. UTF-8 boundaries only.
pub(super) fn safe_prefix_len(text: &str, markers: &[&str]) -> usize {
    let held = markers
        .iter()
        .flat_map(|m| {
            (1..m.len().min(text.len() + 1)).rev().find(|&n| {
                text.is_char_boundary(text.len() - n) && m.starts_with(&text[text.len() - n..])
            })
        })
        .max()
        .unwrap_or(0);
    text.len() - held
}
fn stopped_prefix<'a>(text: &'a str, stops: &[String], done: bool) -> (&'a str, bool) {
    if let Some(at) = stops.iter().filter_map(|s| text.find(s)).min() {
        return (&text[..at], true);
    }
    let end = if done {
        text.len()
    } else {
        safe_prefix_len(text, &stops.iter().map(String::as_str).collect::<Vec<_>>())
    };
    (&text[..end], false)
}

struct LogRecord {
    start: usize,
    end: usize,
    value: Value,
}
struct Logprobs {
    top: usize,
    vocab: Option<Arc<crate::structured_output::Vocabulary>>,
    records: Vec<LogRecord>,
    cursor: usize,
}
impl Logprobs {
    fn new(top: usize, tokenizer: &ServerTokenizer) -> Result<Self, String> {
        let vocab = match tokenizer {
            ServerTokenizer::Real(t) => Some(t.structured_vocabulary()?),
            _ => None,
        };
        Ok(Self {
            top,
            vocab,
            records: vec![],
            cursor: 0,
        })
    }
    fn token(&self, id: u32, tokenizer: &ServerTokenizer) -> (String, Option<Vec<u8>>) {
        let bytes = match &self.vocab {
            Some(v) => v.token_bytes(id).map(<[u8]>::to_vec),
            None => u8::try_from(id).ok().map(|b| vec![b]),
        };
        let token = bytes
            .as_ref()
            .map(|b| String::from_utf8_lossy(b).into_owned())
            .unwrap_or_else(|| match tokenizer {
                ServerTokenizer::Real(t) => t.id_to_token(id).unwrap_or_default(),
                _ => "<eos>".into(),
            });
        (token, bytes)
    }
    fn push(&mut self, event: &TokenEvent<'_>, tokenizer: &ServerTokenizer) -> Result<(), String> {
        let logits = event
            .logits
            .ok_or("backend does not provide logits for logprobs")?;
        if event.id as usize >= logits.len()
            || logits.is_empty()
            || logits.iter().any(|v| !v.is_finite())
        {
            return Err("invalid backend logits for logprobs".into());
        }
        // Stable f64 normalization. Do not apply temperature, filters, penalties, or the U4 mask.
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let norm = logits
            .iter()
            .map(|&v| (v as f64 - max).exp())
            .sum::<f64>()
            .ln();
        let entry = |id: u32| {
            let (token, bytes) = self.token(id, tokenizer);
            json!({"token":token,"bytes":bytes,"logprob": logits[id as usize] as f64-max-norm})
        };
        let mut order: Vec<usize> = if self.top == 0 {
            vec![]
        } else {
            (0..logits.len()).collect()
        };
        let compare =
            |a: &usize, b: &usize| logits[*b].total_cmp(&logits[*a]).then_with(|| a.cmp(b));
        let n = self.top.min(order.len());
        if n > 0 && n < order.len() {
            order.select_nth_unstable_by(n, compare);
        }
        order.truncate(n);
        order.sort_unstable_by(compare);
        let mut value = entry(event.id);
        value["top_logprobs"] = json!(order
            .into_iter()
            .map(|id| entry(id as u32))
            .collect::<Vec<_>>());
        let start = self.cursor;
        self.cursor += value["bytes"].as_array().map_or(0, Vec::len);
        self.records.push(LogRecord {
            start,
            end: self.cursor,
            value,
        });
        Ok(())
    }
    // Align context-sensitive decoders (e.g. SentencePiece's initial-space rule) when a
    // complete piece arrives. Pending UTF-8 byte tokens keep their own byte boundaries.
    fn align(&mut self, decoded: usize) {
        if decoded != self.cursor {
            let old = self.cursor;
            for r in self.records.iter_mut().rev() {
                if r.end != old {
                    break;
                }
                r.end = decoded;
                r.start = r.start.min(decoded);
            }
            self.cursor = decoded;
        }
    }
    fn values(&self, end: usize, text: &str) -> Vec<Value> {
        self.records
            .iter()
            .filter(|r| r.end <= end && r.start < end)
            .map(|r| {
                let mut value = r.value.clone();
                let mut boundary = r.start.min(text.len());
                while !text.is_char_boundary(boundary) {
                    boundary -= 1;
                }
                value["_text_offset"] = json!(text[..boundary].chars().count());
                value
            })
            .collect()
    }
}
fn log_payload(values: &[Value], legacy: bool) -> Value {
    if !legacy {
        let values: Vec<Value> = values
            .iter()
            .map(|v| {
                let mut v = v.clone();
                v.as_object_mut().unwrap().remove("_text_offset");
                v
            })
            .collect();
        return json!({"content":values,"refusal":null});
    }
    let offsets: Vec<Value> = values.iter().map(|v| v["_text_offset"].clone()).collect();
    let tops: Vec<Value> = values
        .iter()
        .map(|v| {
            let mut map = serde_json::Map::new();
            for e in v["top_logprobs"].as_array().unwrap() {
                map.insert(e["token"].as_str().unwrap().into(), e["logprob"].clone());
            }
            map.insert(v["token"].as_str().unwrap().into(), v["logprob"].clone());
            Value::Object(map)
        })
        .collect();
    json!({"tokens":values.iter().map(|v| &v["token"]).collect::<Vec<_>>(),"token_logprobs":values.iter().map(|v| &v["logprob"]).collect::<Vec<_>>(),"top_logprobs":tops,"text_offset":offsets,"bytes":values.iter().map(|v| &v["bytes"]).collect::<Vec<_>>()})
}

fn parts(
    text: &str,
    reasoning: Option<&Reasoning>,
    harmony_tools: bool,
    done: bool,
) -> (String, String) {
    if harmony_tools {
        return crate::tool_calls::harmony_stream_parts(text, done);
    }
    reasoning.map_or_else(
        || (String::new(), text.to_owned()),
        |r| {
            if done {
                r.progress_done(text)
            } else {
                r.progress(text)
            }
        },
    )
}
fn stream_text(
    text: &str,
    reasoning: Option<&Reasoning>,
    tools: Option<&(crate::tool_calls::Format, crate::tool_calls::Tools)>,
    done: bool,
) -> (String, String) {
    let prefix = if done {
        text
    } else {
        tools.map_or(text, |(f, _)| f.stream_prefix(text))
    };
    parts(
        prefix,
        reasoning,
        tools.is_some_and(|(f, _)| *f == crate::tool_calls::Format::Harmony),
        done,
    )
}

pub(super) fn respond(
    request: Request,
    state: &Arc<ServerState>,
    prompt: Vec<u32>,
    mut params: GenParams,
    model: String,
    reasoning: Option<Reasoning>,
    tools: Option<(crate::tool_calls::Format, crate::tool_calls::Tools)>,
    options: Options,
) {
    let mut logs = match options
        .logprobs
        .map(|top| Logprobs::new(top, &state.tokenizer))
        .transpose()
    {
        Ok(v) => v,
        Err(e) => {
            respond_json(
                request,
                400,
                &json!({"error":{"message":e,"type":"invalid_request_error"}}),
            );
            return;
        }
    };
    let cid = cmpl_id();
    let created = now_unix();
    let object = if options.legacy {
        "text_completion"
    } else {
        "chat.completion.chunk"
    };
    let chunk = |delta: Value, finish: Option<&str>, lp: Value| {
        let choice = if options.legacy {
            json!({"index":0,"text":delta["content"].as_str().unwrap_or(""),"finish_reason":finish,"logprobs":lp})
        } else {
            json!({"index":0,"delta":delta,"finish_reason":finish,"logprobs":lp})
        };
        json!({"id":cid,"object":object,"created":created,"model":model,"choices":[choice]})
    };
    let mut request = Some(request);
    let mut writer = if options.stream {
        Some(request.take().unwrap().into_writer())
    } else {
        None
    };
    if let Some(w) = writer.as_mut() {
        if write_sse_headers(&mut **w).is_err() {
            return;
        }
        if !options.legacy {
            let _ = sse_write(
                &mut **w,
                &sse_frame(&chunk(json!({"role":"assistant"}), None, Value::Null)),
            );
        }
    }
    let mut ids = Vec::new();
    let mut decoded = String::new();
    let mut error = None;
    let mut stopped = false;
    let mut sent = (String::new(), String::new());
    let mut log_sent = 0;
    let mut decoder = state.tokenizer.decode_stream();
    let mut send = |now: (String, String), log_values: Vec<Value>| -> Result<(), String> {
        if let Some(w) = writer.as_mut() {
            for (field, now, previous) in [
                ("reasoning_content", now.0, &mut sent.0),
                ("content", now.1, &mut sent.1),
            ] {
                let delta = now
                    .strip_prefix(previous.as_str())
                    .ok_or("stream rewrote emitted text")?;
                if !delta.is_empty() {
                    sse_write(
                        &mut **w,
                        &sse_frame(&chunk(json!({field:delta}), None, Value::Null)),
                    )
                    .map_err(|e| e.to_string())?;
                }
                *previous = now;
            }
            if options.logprobs.is_some() && log_values.len() > log_sent {
                let fresh = &log_values[log_sent..];
                sse_write(
                    &mut **w,
                    &sse_frame(&chunk(json!({}), None, log_payload(fresh, options.legacy))),
                )
                .map_err(|e| e.to_string())?;
                log_sent = log_values.len();
            }
        }
        Ok(())
    };
    let (result, cached) = {
        let mut backend = state.backend.lock().unwrap();
        let result = backend.generate(&prompt, &mut params, &mut |event| {
            ids.push(event.id);
            let result = (|| -> Result<(), String> {
                if let Some(l) = logs.as_mut() {
                    l.push(&event, &state.tokenizer)?;
                }
                if let Some(piece) = decoder(event.id)? {
                    decoded.push_str(&piece);
                    if let Some(l) = logs.as_mut() {
                        l.align(decoded.len());
                    }
                }
                let (safe, matched) = stopped_prefix(&decoded, &options.stops, false);
                stopped = matched;
                if options.stream {
                    let visible = tools.as_ref().map_or(safe, |(f, _)| f.stream_prefix(safe));
                    send(
                        stream_text(safe, reasoning.as_ref(), tools.as_ref(), false),
                        logs.as_ref()
                            .map_or_else(Vec::new, |l| l.values(visible.len(), &decoded)),
                    )?;
                }
                Ok(())
            })();
            if let Err(e) = result {
                error = Some(e);
            }
            if stopped || error.is_some() {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        });
        (result, backend.cached_tokens())
    };
    let mut called = false;
    let mut message = Value::Null;
    let mut full_logs = vec![];
    let result = result.and_then(|_| {
        if let Some(e) = error.take() {
            return Err(e);
        }
        let full = state.tokenizer.final_decode(&ids)?;
        final_stream_tail(&full, &decoded)?;
        let (text, matched) = stopped_prefix(&full, &options.stops, true);
        stopped |= matched;
        let (m, c) = assistant_message(text, reasoning.as_ref(), tools.as_ref());
        message = m;
        called = c;
        full_logs = logs
            .as_ref()
            .map_or_else(Vec::new, |l| l.values(text.len(), &full));
        if options.stream {
            // Reparse once at the end for validation/fallback. Only suffixes of already-sent
            // content are written; partial markers were held back throughout generation.
            let raw = tools.as_ref().map(|(f, cfg)| {
                let mut p = f.parse(text);
                cfg.restore_argument_types(&mut p.calls);
                if cfg.accepts(&p.calls) {
                    p.content
                } else {
                    text.to_owned()
                }
            });
            let raw = raw.as_deref().unwrap_or(text);
            send(
                parts(
                    raw,
                    reasoning.as_ref(),
                    tools
                        .as_ref()
                        .is_some_and(|(f, _)| *f == crate::tool_calls::Format::Harmony),
                    true,
                ),
                full_logs.clone(),
            )?;
        }
        Ok(())
    });
    drop(send);
    if let Err(e) = result {
        if let Some(w) = writer.as_mut() {
            let _ = sse_write(
                &mut **w,
                &sse_frame(&json!({"error":{"message":e,"type":"internal_error"}})),
            );
            let _ = sse_write(&mut **w, "data: [DONE]\n\n");
            let _ = sse_end(&mut **w);
            let _ = w.flush();
        } else {
            respond_json(
                request.take().unwrap(),
                500,
                &json!({"error":{"message":format!("generation error: {e}"),"type":"internal_error"}}),
            );
        }
        return;
    }
    let finish = if stopped {
        "stop"
    } else if called {
        "tool_calls"
    } else if ids.len() >= params.max_tokens {
        "length"
    } else {
        "stop"
    };
    let usage = json!({"prompt_tokens":prompt.len(),"completion_tokens":ids.len(),"total_tokens":prompt.len()+ids.len(),"prompt_tokens_details":{"cached_tokens":cached}});
    if let Some(w) = writer.as_mut() {
        if let Some(calls) = message["tool_calls"].as_array() {
            for (index, call) in calls.iter().enumerate() {
                let mut call = call.clone();
                call["index"] = json!(index);
                let _ = sse_write(
                    &mut **w,
                    &sse_frame(&chunk(json!({"tool_calls":[call]}), None, Value::Null)),
                );
            }
        }
        let _ = sse_write(
            &mut **w,
            &sse_frame(&chunk(json!({}), Some(finish), Value::Null)),
        );
        if options.include_usage {
            let _ = sse_write(
                &mut **w,
                &sse_frame(
                    &json!({"id":cid,"object":object,"created":created,"model":model,"choices":[],"usage":usage}),
                ),
            );
        }
        let _ = sse_write(&mut **w, "data: [DONE]\n\n");
        let _ = sse_end(&mut **w);
        let _ = w.flush();
    } else {
        let lp = if logs.is_some() {
            log_payload(&full_logs, options.legacy)
        } else {
            Value::Null
        };
        let choice = if options.legacy {
            json!({"index":0,"text":message["content"],"finish_reason":finish,"logprobs":lp})
        } else {
            json!({"index":0,"message":message,"finish_reason":finish,"logprobs":lp})
        };
        respond_json(
            request.take().unwrap(),
            200,
            &json!({"id":cid,"object":if options.legacy {"text_completion"}else{"chat.completion"},"created":created,"model":model,"choices":[choice],"usage":usage}),
        );
    }
}
