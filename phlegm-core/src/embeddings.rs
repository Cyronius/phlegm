//! Model-independent embeddings contract and the resident TritNpu adapter.
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
}

#[derive(Debug)]
pub enum EmbeddingError {
    Invalid(String),
    Backend(String),
}

pub trait EmbeddingBackend: Send {
    fn model_id(&self) -> &str;
    fn embed(&mut self, input: EmbeddingInput) -> Result<EmbeddingOutput, EmbeddingError>;
}

struct Worker {
    child: Child,
    input: ChildStdin,
    replies: Receiver<Result<Value, String>>,
}

impl Worker {
    fn spawn(python: &Path, source_dir: &Path, model_dir: &Path) -> Result<Self, String> {
        let mut command = Command::new(python);
        command
            .args(["-I", "-u", "-c", include_str!("embeddings_worker.py")])
            .arg(source_dir)
            .arg(model_dir)
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
        Ok(worker)
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

pub struct TritEmbeddingBackend {
    worker: Option<Worker>,
    id: String,
    lock: Option<NpuLock>,
}
impl TritEmbeddingBackend {
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
            id,
            lock: Some(lock),
        })
    }
}
impl EmbeddingBackend for TritEmbeddingBackend {
    fn model_id(&self) -> &str {
        &self.id
    }
    fn embed(&mut self, input: EmbeddingInput) -> Result<EmbeddingOutput, EmbeddingError> {
        let held = self
            .lock
            .as_ref()
            .ok_or_else(|| {
                EmbeddingError::Backend("embedding worker is no longer available".into())
            })?
            .ensure_held();
        if let Err(e) = held {
            self.worker.take();
            self.lock.take();
            return Err(EmbeddingError::Backend(e));
        }
        let expected_count = input.len();
        let request = match input {
            EmbeddingInput::Texts(texts) => json!({"texts": texts}),
            EmbeddingInput::TokenIds(ids) => json!({"token_ids": ids}),
        };
        let reply = match self
            .worker
            .as_mut()
            .ok_or_else(|| {
                EmbeddingError::Backend("embedding worker is no longer available".into())
            })?
            .request(request)
        {
            Ok(reply) => reply,
            Err(e) => {
                self.worker.take();
                self.lock.take();
                return Err(EmbeddingError::Backend(e));
            }
        };
        if let Err(e) = self.lock.as_ref().unwrap().ensure_held() {
            self.worker.take();
            self.lock.take();
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
                    self.worker.take();
                    self.lock.take();
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
                    .any(|v| v.len() != 384 || v.iter().any(|n| !n.is_finite()))
            {
                return Err("embedding worker returned invalid vectors".into());
            }
            Ok(EmbeddingOutput { vectors, tokens })
        })();
        parsed.map_err(|e| {
            self.worker.take();
            self.lock.take();
            EmbeddingError::Backend(e)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let mut worker = Worker::spawn(Path::new("python3"), &dir, &dir).unwrap();
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
