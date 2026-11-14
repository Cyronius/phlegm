//! HTTP contract for any configured embedding model.
use super::*;
use crate::embeddings::{EmbeddingError, EmbeddingInput};
use std::io::Read;

fn invalid(request: Request, status: u16, msg: &str) {
    respond_json(
        request,
        status,
        &json!({"error": {"message": msg, "type": "invalid_request_error"}}),
    );
}

fn parse_input(value: Option<&Value>) -> Result<EmbeddingInput, &'static str> {
    let input = value.ok_or("input is required")?;
    let ids = |a: &[Value]| -> Option<Vec<u32>> {
        if a.is_empty() || a.len() > 128 {
            return None;
        }
        a.iter()
            .map(|v| v.as_u64().and_then(|n| u32::try_from(n).ok()))
            .collect()
    };
    match input {
        Value::String(s) => Ok(EmbeddingInput::Texts(vec![s.clone()])),
        Value::Array(a) if !a.is_empty() && a.len() <= 2048 => {
            if a.iter().all(Value::is_string) {
                Ok(EmbeddingInput::Texts(
                    a.iter().map(|v| v.as_str().unwrap().to_owned()).collect(),
                ))
            } else if a.iter().all(Value::is_number) {
                Ok(EmbeddingInput::TokenIds(vec![
                    ids(a).ok_or("invalid token sequence")?
                ]))
            } else if a.iter().all(Value::is_array) {
                let rows = a
                    .iter()
                    .map(|v| ids(v.as_array().unwrap()).ok_or("invalid token sequence"))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(EmbeddingInput::TokenIds(rows))
            } else {
                Err("input must be strings or token ID sequences")
            }
        }
        _ => Err("input must be a string or a non-empty array of at most 2048 inputs"),
    }
}

pub(super) fn handle(mut request: Request, state: &ServerState) {
    let mut body = Vec::new();
    if request
        .as_reader()
        .take(16_000_001)
        .read_to_end(&mut body)
        .is_err()
        || body.len() > 16_000_000
    {
        return invalid(request, 400, "invalid or oversized request body");
    }
    handle_body(request, state, &body);
}

pub(super) fn handle_body(request: Request, state: &ServerState, body: &[u8]) {
    let Some(embedder) = &state.embedder else {
        return invalid(request, 404, "no embedding model is loaded");
    };
    let req: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(_) => return invalid(request, 400, "invalid JSON body"),
    };
    let Some(model) = req.get("model").and_then(Value::as_str) else {
        return invalid(request, 400, "model is required");
    };
    if Some(model) != state.embedding_model.as_deref() {
        return invalid(request, 404, "embedding model not found");
    }
    if req.get("dimensions").is_some_and(|v| !v.is_null()) {
        return invalid(
            request,
            400,
            "this embedding model does not support dimensions",
        );
    }
    let format = match req.get("encoding_format") {
        None | Some(Value::Null) => "float",
        Some(Value::String(s)) if s == "float" || s == "base64" => s.as_str(),
        _ => return invalid(request, 400, "encoding_format must be float or base64"),
    };
    let input = match parse_input(req.get("input")) {
        Ok(v) => v,
        Err(e) => return invalid(request, 400, e),
    };
    let count = input.len();
    let output = match embedder.lock().unwrap().embed(input) {
        Ok(v) => v,
        Err(EmbeddingError::Invalid(e)) => return invalid(request, 400, &e),
        Err(EmbeddingError::Backend(e)) => {
            return respond_json(
                request,
                500,
                &json!({"error": {"message": e, "type": "internal_error"}}),
            )
        }
    };
    if output.vectors.len() != count
        || output
            .vectors
            .iter()
            .any(|v| v.is_empty() || v.iter().any(|f| !f.is_finite()))
    {
        return respond_json(
            request,
            500,
            &json!({"error": {"message": "embedding backend returned invalid vectors", "type": "internal_error"}}),
        );
    }
    let data: Vec<Value> = output
        .vectors
        .iter()
        .enumerate()
        .map(|(index, vector)| {
            let embedding = if format == "base64" {
                let bytes: Vec<u8> = vector.iter().flat_map(|f| f.to_le_bytes()).collect();
                json!(base64::encode(bytes))
            } else {
                json!(vector)
            };
            json!({"object": "embedding", "index": index, "embedding": embedding})
        })
        .collect();
    respond_json(
        request,
        200,
        &json!({"object": "list", "data": data, "model": model,
        "usage": {"prompt_tokens": output.tokens, "total_tokens": output.tokens}}),
    );
}
