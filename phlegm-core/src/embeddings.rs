//! Model-independent embeddings contract and resident worker adapters.
use crate::npu_lock::NpuLock;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

#[derive(Debug, Clone, PartialEq)]
pub enum EmbeddingInput {
    Texts(Vec<String>),
    TokenIds(Vec<Vec<u32>>),
}

impl EmbeddingInput {
    pub fn len(&self) -> usize {
        match self {
            Self::Texts(v) => v.len(),
            Self::TokenIds(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub struct EmbeddingOutput {
    pub vectors: Vec<Vec<f32>>,
    pub tokens: usize,
    pub sparse_vectors: Option<Vec<SparseEmbedding>>,
}

#[derive(serde::Deserialize, serde::Serialize)]
pub struct SparseEmbedding {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

#[derive(Debug)]
pub enum EmbeddingError {
    Invalid(String),
    Backend(String),
}

pub trait EmbeddingBackend: Send {
    fn model_id(&self) -> &str;
    fn embed(&mut self, input: EmbeddingInput) -> Result<EmbeddingOutput, EmbeddingError>;
    fn backend_name(&self) -> &str {
        "trit-embed"
    }
    fn embed_with_options(
        &mut self,
        input: EmbeddingInput,
        _prompt_name: Option<&str>,
        dimensions: Option<u32>,
        return_sparse: bool,
    ) -> Result<EmbeddingOutput, EmbeddingError> {
        if return_sparse {
            return Err(EmbeddingError::Invalid(
                "this model has no sparse output".into(),
            ));
        }
        if dimensions.is_some() {
            return Err(EmbeddingError::Invalid(
                "this embedding model does not support dimensions".into(),
            ));
        }
        self.embed(input)
    }
}

struct Worker {
    child: Child,
    input: ChildStdin,
    replies: Receiver<Result<Value, String>>,
}

impl Worker {
    fn start_npue(
        executable: &Path,
        pack: &Path,
        design: &Path,
        attention: Option<&Path>,
    ) -> Result<(Self, Value), String> {
        let mut command = Command::new(executable);
        command.arg(pack).arg(design).args(attention);
        let (worker, hello) = Self::start(command)?;
        if hello.get("npu_attention").and_then(Value::as_bool) != Some(attention.is_some()) {
            return Err("embedding worker did not confirm the requested attention route".into());
        }
        Ok((worker, hello))
    }

    fn spawn(python: &Path, source_dir: &Path, model_dir: &Path) -> Result<Self, String> {
        let mut command = Command::new(python);
        command
            .args(["-I", "-u", "-c", include_str!("embeddings_worker.py")])
            .arg(source_dir)
            .arg(model_dir);
        Self::start(command).map(|(worker, _)| worker)
    }

    fn start(mut command: Command) -> Result<(Self, Value), String> {
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        crate::bench::reference::die_with_parent(&mut command);
        let mut child = command
            .spawn()
            .map_err(|e| format!("starting embedding worker: {e}"))?;
        let input = child.stdin.take().ok_or("missing embedding worker stdin")?;
        let output = child
            .stdout
            .take()
            .ok_or("missing embedding worker stdout")?;
        let (tx, replies) = mpsc::channel();
        std::thread::spawn(move || {
            let mut reader = BufReader::new(output);
            loop {
                let mut line = Vec::new();
                let value = match (&mut reader).take(32_000_001).read_until(b'\n', &mut line) {
                    Ok(0) => Err("embedding worker exited before replying".to_string()),
                    Ok(_) if line.len() > 32_000_000 => {
                        Err("embedding worker reply exceeds 32 MB".to_string())
                    }
                    Ok(_) => serde_json::from_slice(&line)
                        .map_err(|e| format!("invalid embedding worker reply: {e}")),
                    Err(e) => Err(format!("reading embedding worker: {e}")),
                };
                let failed = value.is_err();
                if tx.send(value).is_err() || failed {
                    break;
                }
            }
        });
        let worker = Self {
            child,
            input,
            replies,
        };
        let hello = worker
            .receive()
            .map_err(|e| format!("embedding worker startup: {e}"))?;
        if hello.get("ok").and_then(Value::as_bool) != Some(true)
            || hello.get("protocol").and_then(Value::as_u64) != Some(1)
        {
            return Err(format!(
                "embedding worker startup: {}",
                hello
                    .get("error")
                    .and_then(Value::as_str)
                    .unwrap_or("unsupported protocol")
            ));
        }
        Ok((worker, hello))
    }
    fn receive(&self) -> Result<Value, String> {
        self.replies
            .recv_timeout(Duration::from_secs(600))
            .map_err(|e| format!("embedding worker reply failed or timed out: {e}"))?
    }
    fn request(&mut self, value: Value) -> Result<Value, String> {
        serde_json::to_writer(&mut self.input, &value).map_err(|e| e.to_string())?;
        self.input
            .write_all(b"\n")
            .and_then(|_| self.input.flush())
            .map_err(|e| format!("writing embedding request: {e}"))?;
        self.receive()
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub struct ResidentEmbeddingBackend {
    worker: Option<Worker>,
    short: Option<ShortRoute>,
    id: String,
    lock: Option<NpuLock>,
    width: usize,
    native: bool,
    sparse: bool,
}

struct ShortRoute {
    worker: Worker,
    context: usize,
}
pub type TritEmbeddingBackend = ResidentEmbeddingBackend;

impl ResidentEmbeddingBackend {
    pub fn load(
        id: String,
        python: &Path,
        source_dir: &Path,
        model_dir: &Path,
    ) -> Result<Self, String> {
        let lock = NpuLock::acquire("phlegm embeddings")?;
        let worker = Worker::spawn(python, source_dir, model_dir)?;
        Ok(Self {
            worker: Some(worker),
            short: None,
            id,
            lock: Some(lock),
            width: 384,
            native: false,
            sparse: false,
        })
    }

    pub fn load_npue(
        id: String,
        executable: &Path,
        pack: &Path,
        design: &Path,
        attention: Option<&Path>,
    ) -> Result<Self, String> {
        Self::load_npue_routed(id, executable, pack, design, attention, None)
    }

    pub fn load_npue_routed(
        id: String,
        executable: &Path,
        pack: &Path,
        design: &Path,
        attention: Option<&Path>,
        short: Option<(&Path, Option<&Path>)>,
    ) -> Result<Self, String> {
        let lock = NpuLock::acquire("phlegm NPUE embeddings")?;
        let (worker, hello) = Worker::start_npue(executable, pack, design, attention)?;
        let width = hello
            .get("dimensions")
            .and_then(Value::as_u64)
            .and_then(|n| usize::try_from(n).ok())
            .filter(|&n| n > 0)
            .ok_or("embedding worker omitted its vector width")?;
        let short =
            short
                .map(|(design, attention)| {
                    let (worker, probe) = Worker::start_npue(executable, pack, design, attention)?;
                    let full_context = hello
                        .get("context")
                        .and_then(Value::as_u64)
                        .ok_or("full embedding worker omitted its context capacity")?;
                    let context = probe.get("context").and_then(Value::as_u64)
                .filter(|&n| n > 0 && n < full_context)
                .and_then(|n| usize::try_from(n).ok())
                .ok_or("short embedding design must have a smaller context than the full design")?;
                    if probe.get("length_probe").and_then(Value::as_bool) != Some(true)
                        || ["dimensions", "prompt_names", "sparse"]
                            .iter()
                            .any(|key| probe.get(key) != hello.get(key))
                    {
                        return Err(
                            "short embedding worker is incompatible with the full worker"
                                .to_string(),
                        );
                    }
                    Ok(ShortRoute { worker, context })
                })
                .transpose()?;
        lock.ensure_held()?;
        Ok(Self {
            worker: Some(worker),
            short,
            id,
            lock: Some(lock),
            width,
            native: true,
            sparse: hello.get("sparse").and_then(Value::as_bool) == Some(true),
        })
    }

    fn stop(&mut self) {
        self.short.take();
        self.worker.take();
        self.lock.take();
    }

    fn request(&mut self, request: Value) -> Result<Value, String> {
        if let Some(route) = &mut self.short {
            let fits = if let Some(rows) = request.get("token_ids").and_then(Value::as_array) {
                rows.iter()
                    .all(|row| row.as_array().is_some_and(|ids| ids.len() <= route.context))
            } else {
                let mut probe = request.clone();
                probe["length_only"] = json!(true);
                let reply = route.worker.request(probe)?;
                if reply.get("ok").and_then(Value::as_bool) != Some(true) {
                    return Ok(reply);
                }
                reply
                    .get("fits")
                    .and_then(Value::as_bool)
                    .ok_or("short embedding worker omitted the input length decision")?
            };
            if fits {
                return route.worker.request(request);
            }
        }
        self.worker
            .as_mut()
            .ok_or("embedding worker is no longer available")?
            .request(request)
    }
}
impl EmbeddingBackend for ResidentEmbeddingBackend {
    fn model_id(&self) -> &str {
        &self.id
    }
    fn backend_name(&self) -> &str {
        if self.native {
            "npue"
        } else {
            "trit-embed"
        }
    }
    fn embed(&mut self, input: EmbeddingInput) -> Result<EmbeddingOutput, EmbeddingError> {
        self.embed_with_options(input, None, None, false)
    }
    fn embed_with_options(
        &mut self,
        input: EmbeddingInput,
        prompt_name: Option<&str>,
        dimensions: Option<u32>,
        return_sparse: bool,
    ) -> Result<EmbeddingOutput, EmbeddingError> {
        if return_sparse && !self.sparse {
            return Err(EmbeddingError::Invalid(
                "this model has no sparse output".into(),
            ));
        }
        if !self.native && dimensions.is_some() {
            return Err(EmbeddingError::Invalid(
                "this embedding model does not support dimensions".into(),
            ));
        }
        let held = self
            .lock
            .as_ref()
            .ok_or_else(|| {
                EmbeddingError::Backend("embedding worker is no longer available".into())
            })?
            .ensure_held();
        if let Err(e) = held {
            self.stop();
            return Err(EmbeddingError::Backend(e));
        }
        let expected_count = input.len();
        let expected_width = dimensions.map_or(self.width, |n| n as usize);
        let mut request = match input {
            EmbeddingInput::Texts(texts) => json!({"texts": texts}),
            EmbeddingInput::TokenIds(ids) => json!({"token_ids": ids}),
        };
        if self.native {
            if let Some(prompt) = prompt_name {
                request["prompt_name"] = json!(prompt);
            }
            if let Some(dimensions) = dimensions {
                request["dimensions"] = json!(dimensions);
            }
            if return_sparse {
                request["return_sparse"] = json!(true);
            }
        }
        let reply = match self.request(request) {
            Ok(reply) => reply,
            Err(e) => {
                self.stop();
                return Err(EmbeddingError::Backend(e));
            }
        };
        if let Err(e) = self.lock.as_ref().unwrap().ensure_held() {
            self.stop();
            return Err(EmbeddingError::Backend(e));
        }
        if reply.get("ok").and_then(Value::as_bool) != Some(true) {
            let error = reply
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("embedding worker failed")
                .to_string();
            return Err(
                if reply.get("invalid").and_then(Value::as_bool) == Some(true) {
                    EmbeddingError::Invalid(error)
                } else {
                    self.stop();
                    EmbeddingError::Backend(error)
                },
            );
        }
        let parsed = (|| -> Result<EmbeddingOutput, String> {
            let vectors: Vec<Vec<f32>> =
                serde_json::from_value(reply.get("vectors").cloned().unwrap_or(Value::Null))
                    .map_err(|e| format!("invalid embedding vectors: {e}"))?;
            let tokens = reply
                .get("tokens")
                .and_then(Value::as_u64)
                .and_then(|n| usize::try_from(n).ok())
                .ok_or("invalid embedding token count")?;
            if vectors.len() != expected_count
                || vectors
                    .iter()
                    .any(|v| v.len() != expected_width || v.iter().any(|n| !n.is_finite()))
            {
                return Err("embedding worker returned invalid vectors".into());
            }
            let sparse_vectors = if return_sparse {
                let sparse: Vec<SparseEmbedding> = serde_json::from_value(
                    reply.get("sparse_vectors").cloned().unwrap_or(Value::Null),
                )
                .map_err(|e| format!("invalid sparse embedding vectors: {e}"))?;
                if sparse.len() != expected_count
                    || sparse.iter().any(|row| {
                        row.indices.len() != row.values.len()
                            || row.indices.windows(2).any(|pair| pair[0] >= pair[1])
                            || row.values.iter().any(|v| !v.is_finite() || *v <= 0.0)
                    })
                {
                    return Err("embedding worker returned invalid sparse vectors".into());
                }
                Some(sparse)
            } else {
                None
            };
            Ok(EmbeddingOutput {
                vectors,
                tokens,
                sparse_vectors,
            })
        })();
        parsed.map_err(|e| {
            self.stop();
            EmbeddingError::Backend(e)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn python() -> &'static Path {
        Path::new(if cfg!(windows) { "python" } else { "python3" })
    }

    #[test]
    fn npue_protocol_passes_request_options_and_recovers_from_invalid_input() {
        let mut command = Command::new(python());
        command.args([
            "-I",
            "-u",
            "-c",
            r#"
import json, sys
print(json.dumps(dict(ok=True, protocol=1, dimensions=2, npu_attention=True)))
for line in sys.stdin:
    request = json.loads(line)
    if request.get('prompt_name') != 'search_document':
        print(json.dumps(dict(ok=False, invalid=True, error='prompt_name is required')))
    else:
        assert request['dimensions'] == 1 and request['texts'] == ['one', 'two']
        result = dict(ok=True, tokens=4, vectors=[[1], [-1]])
        if request.get('return_sparse'):
            result['sparse_vectors'] = [
                dict(indices=[1, 5], values=[0.25, 0.5]),
                dict(indices=[], values=[]),
            ]
        print(json.dumps(result))
"#,
        ]);
        let (worker, _) = Worker::start(command).unwrap();
        let lock_path =
            std::env::temp_dir().join(format!("phlegm-npue-test-{}.lock", std::process::id()));
        let mut backend = ResidentEmbeddingBackend {
            worker: Some(worker),
            short: None,
            id: "nomic-test".into(),
            lock: Some(NpuLock::for_test(lock_path)),
            width: 2,
            native: true,
            sparse: true,
        };
        let input = EmbeddingInput::Texts(vec!["one".into(), "two".into()]);
        assert!(matches!(
            backend.embed(input.clone()),
            Err(EmbeddingError::Invalid(_))
        ));
        let result = backend
            .embed_with_options(input, Some("search_document"), Some(1), true)
            .unwrap();
        assert_eq!(result.tokens, 4);
        assert_eq!(result.vectors, [vec![1.], vec![-1.]]);
        let sparse = result.sparse_vectors.unwrap();
        assert_eq!(sparse[0].indices, [1, 5]);
        assert_eq!(sparse[0].values, [0.25, 0.5]);
        assert!(sparse[1].values.is_empty());
    }

    #[test]
    fn resident_worker_protocol_handles_batches_and_invalid_ids() {
        let dir = std::env::temp_dir().join(format!("phlegm-embed-worker-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("trit_npu.py"),
            r#"
class Table: shape = (100, 384)
class Word: qweight = Table()
class Model: word = Word()
class Vectors:
    def __init__(self, ids): self.ids = ids
    def tolist(self): return [[float(len(row))] * 384 for row in self.ids]
class TritNpu:
    def __init__(self, path): self.model = Model()
    def tokenize(self, texts): return [[1] * len(text) for text in texts]
    def encode_ids(self, ids): return Vectors(ids), 1
    def close(self): pass
"#,
        )
        .unwrap();
        let mut worker = Worker::spawn(python(), &dir, &dir).unwrap();
        let response = worker.request(json!({"texts": ["ab", "c"]})).unwrap();
        assert_eq!(response["tokens"], 3);
        assert_eq!(response["vectors"][0].as_array().unwrap().len(), 384);
        let response = worker.request(json!({"token_ids": [[100]]})).unwrap();
        assert_eq!(response["invalid"], true);
        let response = worker.request(json!({"token_ids": [[5, 6]]})).unwrap();
        assert_eq!(response["tokens"], 2);
        drop(worker);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
