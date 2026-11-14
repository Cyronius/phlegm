use super::*;
use crate::sampler::{Penalties, Ramp, Sampler};
use crate::speculative::{NoneDrafter, SpeculativeMode};
use std::os::unix::fs::PermissionsExt;
use std::sync::atomic::{AtomicUsize, Ordering};

const FAKE: &str = r#"#!/usr/bin/env python3
import json, struct, sys
from pathlib import Path
root = Path(__file__).parent
config = json.loads((root/'config.json').read_text())
protocol = config.get('protocol', 2)
print(json.dumps({'ok':True,'protocol':protocol,'engine':'openflowlm-cpp-reference'}), flush=True)
committed, pending, mark = [], None, None
def logits(ctx):
    if 'sequence' in config:
        seq = config['sequence']; token = seq[(len(ctx)-4)%len(seq)]
        return [10. if i == token else -10. for i in range(8)]
    h = 1
    for t in ctx: h = (h*17+t+1)%97
    return [((h+i*13)%23-11)/3 for i in range(8)]
for line in sys.stdin:
    r = json.loads(line); cmd = r['command']; payload = b''; out = {}
    if cmd == 'init':
        cap = r['max_ctx']
        out = {'context_capacity':cap, 'real_vocab':8, 'recurrent_layers':1}
        if protocol == 2:
            out.update(verify_width=4, verify_ref=True, draft={'method':'mtp','max':3})
    elif cmd == 'reset':
        assert pending is None
        committed, mark = [], None
    elif cmd == 'rewind':
        assert pending is None and r['pos'] == mark
        committed = committed[:mark]
    elif cmd == 'prefill_logits':
        assert pending is None
        committed.extend(r['prompt']); mark = r['mark']
        out['mark'] = mark
        payload = struct.pack('<8f', *logits(committed))
    elif cmd == 'decode_logits':
        assert pending is None
        committed.append(r['token'])
        payload = struct.pack('<8f', *logits(committed))
    elif cmd == 'verify_logits':
        assert protocol == 2 and pending is None and r['accept'] == 0
        pending = list(r['tokens'])
        # Deterministic bridge-owned drafts; their quality cannot affect output.
        pending.extend([1]*r['draft'])
        assert 1 <= len(pending) <= 4 and len(committed)+len(pending) <= cap
        temp = list(committed); rows = []
        for t in pending:
            temp.append(t); rows.extend(logits(temp))
        out.update(rows=len(pending), tokens=pending, position=len(committed))
        payload = struct.pack('<%df'%len(rows), *rows)
        fault = config.get('fault')
        if fault == 'position': out['position'] += 1
        if fault == 'tokens': out['tokens'] = [7]*len(pending)
        if fault == 'rows': out['rows'] += 1
        if fault == 'payload': payload = payload[:-4]
        if fault == 'nonfinite': payload = struct.pack('<f', float('nan')) + payload[4:]
    elif cmd == 'accept':
        assert pending is not None and 0 <= r['n'] <= len(pending)
        if config.get('fault') == 'accept':
            print(json.dumps({'ok':False,'error':'injected accept failure'}), flush=True)
            continue
        committed.extend(pending[:r['n']]); pending = None
        out['position'] = len(committed)
    else:
        raise RuntimeError(cmd)
    assert len(committed) <= cap
    with (root/'log.jsonl').open('a') as log:
        log.write(json.dumps({'request':r, 'committed':committed, 'mark':mark, 'pending':pending})+'\n')
    out['ok'] = True
    if payload: out['logits_f32_le_bytes'] = len(payload)
    sys.stdout.write(json.dumps(out)+'\n'); sys.stdout.flush()
    sys.stdout.buffer.write(payload); sys.stdout.buffer.flush()
"#;

pub(crate) struct Fixture {
    pub(crate) dir: PathBuf,
}
impl Fixture {
    pub(crate) fn new(config: Value) -> Self {
        static ID: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "phlegm-spec-{}-{}",
            std::process::id(),
            ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&dir).unwrap();
        let bridge = dir.join("bridge");
        fs::write(&bridge, FAKE).unwrap();
        fs::set_permissions(&bridge, fs::Permissions::from_mode(0o700)).unwrap();
        fs::write(dir.join("config.json"), config.to_string()).unwrap();
        let adapter = format!("{:x}", Sha256::digest(BRIDGE_SOURCE));
        let files = BTreeMap::from([("bridge/bridge.cpp".into(), adapter.clone())]);
        fs::write(bridge.with_extension("build.json"), json!({
            "protocol":config["protocol"].as_u64().unwrap_or(2), "engine":BACKEND_NAME,
            "source_commit":"0".repeat(40), "source_dirty":false,
            "source_tree_sha256":manifest_hash(&files), "engine_tree_sha256":ENGINE_TREE_SHA256, "adapter_sha256":adapter,
            "binary_sha256":hash_file(&bridge).unwrap(), "compiler_flags":"fake protocol 2 test", "source_files":files
        }).to_string()).unwrap();
        Self { dir }
    }
    pub(crate) fn load(&self, capacity: usize) -> NpuBackend {
        NpuBackend::load_with_lock(
            NpuOptions {
                bridge: self.dir.join("bridge"),
                model_dir: self.dir.clone(),
                kernel_dir: self.dir.clone(),
                turbo: false,
                context_capacity: capacity,
                prefill_route: "auto".into(),
                prefill_mode: "fast".into(),
                threads: 1,
                request_timeout: Duration::from_secs(5),
                dispatch_timeout_ms: 1000,
            },
            || Ok(NpuLock::for_test(self.dir.join("fake.lock"))),
        )
        .unwrap()
    }
    pub(crate) fn log(&self) -> Vec<Value> {
        fs::read_to_string(self.dir.join("log.jsonl"))
            .unwrap()
            .lines()
            .map(|s| serde_json::from_str(s).unwrap())
            .collect()
    }
    fn assert_committed(&self, backend: &NpuBackend) {
        let log = self.log();
        let last = log.last().unwrap();
        assert!(last["pending"].is_null());
        assert_eq!(last["committed"], json!(backend.context));
        assert_eq!(last["mark"], json!(backend.mark));
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}

fn params(setting: usize, max: usize, constrained: bool) -> GenParams {
    let mut sampler = match setting {
        0 => Sampler::new(0., 0, 1., 1., Some(9)),
        1 => Sampler::new(1., 0, 1., 1., Some(1)),
        2 => Sampler::new(0.7, 3, 0.95, 1.1, Some(2)),
        3 => Sampler::new(1.3, 5, 0.8, 1.2, Some(3)),
        4 => Sampler::new(1., 4, 0.9, 1.05, Some(41)).with_penalties(Penalties {
            presence: 0.8,
            frequency: 0.3,
            repetition_reply_only: true,
            ramps: vec![],
        }),
        _ => Sampler::new(0.9, 6, 0.95, 1.1, Some(99)).with_penalties(Penalties {
            ramps: vec![Ramp {
                ids: vec![1, 2],
                strength: 3.,
                start: 2,
                window: 5,
                until_seen: false,
            }],
            ..Default::default()
        }),
    };
    if constrained {
        use crate::structured_output::{Constraint, Vocabulary};
        let bytes = ["\"", "a", "b", "c", " ", "{", "}", ""]
            .iter()
            .map(|s| Some(s.as_bytes().to_vec()))
            .collect();
        sampler = sampler.with_constraint(
            Constraint::new(
                &json!({"type":"string", "enum":["ababababab"]}),
                std::sync::Arc::new(Vocabulary::new(bytes, [7].into_iter().collect())),
            )
            .unwrap(),
        );
    }
    GenParams {
        speculative: SpeculativeMode::Off,
        max_tokens: max,
        sampler,
        stop_ids: [7].into_iter().collect(),
        reuse_mark: Some(2),
    }
}

struct Adversary {
    kind: usize,
    oracle: Vec<u32>,
    prompt_len: usize,
}
impl Drafter for Adversary {
    fn draft(&mut self, history: &[u32], max: usize) -> Draft {
        let at = history.len() - self.prompt_len;
        let mut expected = self.oracle.get(at..).unwrap_or(&[]).to_vec();
        expected.resize(64, 1);
        let out = match self.kind {
            0 => expected[..max].to_vec(), // always correct
            1 => expected[..max].iter().map(|t| (t + 1) % 8).collect(),
            2 => expected[..max]
                .iter()
                .enumerate()
                .map(|(i, t)| if i == 0 { *t } else { (t + 1) % 8 })
                .collect(),
            3 => vec![7; max], // draft a stop ID; it is not a stop until sampled
            4 => expected,     // ignore max, capacity and the policy's cooldown
            5 => vec![5; max], // '{' is forbidden inside the constrained string
            _ => vec![],
        };
        Draft::Tokens(out)
    }
}

fn generate(
    backend: &mut NpuBackend,
    prompt: &[u32],
    params: &mut GenParams,
    drafter: Option<Box<dyn Drafter>>,
    break_at: usize,
) -> Vec<u32> {
    let mut out = Vec::new();
    backend
        .generate_with_drafter(
            prompt,
            params,
            &mut |event| {
                assert_eq!(event.logits.unwrap().len(), 8);
                out.push(event.id);
                if out.len() == break_at {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            },
            drafter,
        )
        .unwrap();
    out
}

#[test]
fn adversarial_drafts_preserve_ids_rng_masks_commits_and_reuse() {
    let fixture = Fixture::new(json!({}));
    let mut backend = fixture.load(24);
    let prompt = [1, 2, 1, 2];
    for constrained in [false, true] {
        for setting in 0..6 {
            // No stop IDs in the unmasked runs: exercise the full output/capacity boundary.
            let mut plain = params(setting, 21, constrained);
            if !constrained {
                plain.stop_ids.clear();
            }
            let expected = generate(&mut backend, &prompt, &mut plain, None, usize::MAX);
            fixture.assert_committed(&backend);
            for kind in 0..7 {
                for break_at in [usize::MAX, 1, 3, 4] {
                    let mut p = params(setting, 21, constrained);
                    if !constrained {
                        p.stop_ids.clear();
                    }
                    let drafter = Box::new(Adversary {
                        kind,
                        oracle: expected.clone(),
                        prompt_len: prompt.len(),
                    });
                    let got = generate(&mut backend, &prompt, &mut p, Some(drafter), break_at);
                    assert_eq!(
                        got,
                        expected[..expected.len().min(break_at)],
                        "setting={setting} kind={kind} constrained={constrained} break={break_at}"
                    );
                    fixture.assert_committed(&backend);
                    let stop_id = expected.len() < 21 && break_at > got.len();
                    let committed = if stop_id {
                        got.len()
                    } else {
                        got.len().saturating_sub(1)
                    };
                    assert_eq!(
                        backend.context,
                        [prompt.to_vec(), got[..committed].to_vec()].concat()
                    );
                    // Second request appends to exactly the committed prefix.
                    let mut follow = backend.context.clone();
                    let reused = if follow.len() < backend.context_capacity {
                        follow.push(3);
                        backend.context.len()
                    } else {
                        *follow.last_mut().unwrap() = 3;
                        backend.mark.unwrap()
                    };
                    let mut p = params(setting, 1, false);
                    p.reuse_mark = Some(follow.len());
                    generate(&mut backend, &follow, &mut p, None, usize::MAX);
                    assert_eq!(backend.cached_tokens(), reused);
                    fixture.assert_committed(&backend);
                }
            }
        }
    }
    assert!(fixture
        .log()
        .iter()
        .any(|v| v["request"]["command"] == "verify_logits"
            && v["request"]["tokens"].as_array().unwrap().len() == 4));
}

#[test]
fn negotiated_modes_and_protocol_one_fallback() {
    for protocol in [1, 2] {
        let f = Fixture::new(json!({"protocol":protocol}));
        let mut backend = f.load(32);
        let mut expected = Vec::new();
        for mode in [
            SpeculativeMode::Off,
            SpeculativeMode::Auto,
            SpeculativeMode::Lookup,
            SpeculativeMode::Mtp,
        ] {
            let mut p = params(2, 20, false);
            p.stop_ids.clear();
            p.speculative = mode;
            let mut out = Vec::new();
            backend
                .generate(&[1, 2, 1, 2], &mut p, &mut |e| {
                    out.push(e.id);
                    ControlFlow::Continue(())
                })
                .unwrap();
            if mode == SpeculativeMode::Off {
                expected = out.clone();
            }
            assert_eq!(out, expected);
            f.assert_committed(&backend);
        }
        let verifies = f
            .log()
            .iter()
            .filter(|v| v["request"]["command"] == "verify_logits")
            .count();
        assert_eq!(verifies > 0, protocol == 2);
        if protocol == 2 {
            backend.draft_max = 0;
            assert!(backend
                .drafter(SpeculativeMode::Mtp)
                .err()
                .unwrap()
                .contains("stage 4"));
        }
    }
}

#[test]
fn invalid_protocol_two_replies_invalidate_all_reuse() {
    for fault in [
        "position",
        "tokens",
        "rows",
        "payload",
        "nonfinite",
        "accept",
    ] {
        let f = Fixture::new(json!({"fault":fault}));
        let mut backend = f.load(32);
        let mut p = params(0, 10, false);
        p.stop_ids.clear();
        p.speculative = SpeculativeMode::Lookup;
        let err = backend
            .generate(&[1, 2, 1, 2], &mut p, &mut |_| ControlFlow::Continue(()))
            .unwrap_err();
        assert!(!err.is_empty(), "{fault}");
        assert!(backend.context.is_empty(), "{fault}");
        assert!(backend.mark.is_none(), "{fault}");
    }
}

#[test]
fn no_drafter_uses_width_one_and_never_samples_tentative_rows() {
    let f = Fixture::new(json!({}));
    let mut backend = f.load(12);
    let mut p = params(0, 9, false);
    p.stop_ids.clear();
    generate(
        &mut backend,
        &[1, 2, 1, 2],
        &mut p,
        Some(Box::new(NoneDrafter)),
        usize::MAX,
    );
    for v in f
        .log()
        .iter()
        .filter(|v| v["request"]["command"] == "verify_logits")
    {
        assert_eq!(v["request"]["tokens"].as_array().unwrap().len(), 1);
    }
    f.assert_committed(&backend);
}
