//! Protocol 2 client and verified row walk. The bridge owns all tentative state.
use super::*;
use crate::backend::TokenEvent;
use crate::speculative::{Bridge, Draft, Drafter, Lookup, SpecPolicy, SpeculativeMode};
use std::ops::ControlFlow;

impl NpuBackend {
    pub(super) fn drafter(
        &self,
        mode: SpeculativeMode,
    ) -> Result<Option<Box<dyn Drafter>>, String> {
        // Old bridges always use the plain loop, even if speculation was requested.
        if mode == SpeculativeMode::Off || self.verify_width == 0 {
            return Ok(None);
        }
        match mode {
            SpeculativeMode::Mtp if self.draft_max == 0 => {
                Err("mtp speculative decoding requires a bridge MTP drafter (P4 stage 4)".into())
            }
            SpeculativeMode::Mtp => Ok(Some(Box::new(Bridge {
                max: self.draft_max,
            }))),
            SpeculativeMode::Auto if self.draft_max > 0 => Ok(Some(Box::new(Bridge {
                max: self.draft_max,
            }))),
            _ => Ok(Some(Box::new(Lookup::default()))),
        }
    }

    /// The request has already prefilled. Between cycles `history.last()` is the
    /// one emitted token outside the committed context, as in ordinary decode.
    pub(super) fn generate_speculative(
        &mut self,
        prompt: &[u32],
        params: &mut GenParams,
        on_token: &mut dyn FnMut(TokenEvent<'_>) -> ControlFlow<()>,
        logits: Vec<f32>,
        drafter: &mut dyn Drafter,
    ) -> Result<(), String> {
        let mut history = prompt.to_vec();
        let mut sample_history: Vec<i64> = prompt.iter().map(|&t| i64::from(t)).collect();
        let first = params.sampler.try_sample(&logits, &sample_history)? as u32;
        if params.stop_ids.contains(&first) {
            return Ok(());
        }
        if on_token(TokenEvent {
            id: first,
            logits: Some(&logits),
        })
        .is_break()
        {
            return Ok(());
        }
        history.push(first);
        sample_history.push(i64::from(first));
        let mut emitted = 1;
        let mut policy = SpecPolicy::new(self.verify_width);
        while emitted < params.max_tokens {
            let remaining = params.max_tokens - emitted;
            let room = self.context_capacity - self.context.len();
            let k = policy.max_draft(remaining, room);
            let mut tokens = vec![*history.last().unwrap()];
            let mut bridge_drafts = 0;
            match drafter.draft(&history, k) {
                Draft::Tokens(drafts) => {
                    tokens.extend(
                        drafts
                            .into_iter()
                            .take(k)
                            .take_while(|&t| (t as usize) < self.real_vocab),
                    );
                }
                Draft::Bridge(n) => {
                    bridge_drafts = n.min(k);
                }
            }
            // Take before *every* mutating command. A failed/invalid reply must
            // never leave a reusable prefix or mark pointing at uncertain state.
            let context = std::mem::take(&mut self.context);
            let mark = self.mark.take();
            let (reply, bytes) = self.process.request_with_payload(json!({
                "command":"verify_logits", "accept":0, "tokens":tokens, "draft":bridge_drafts
            }))?;
            let rows = reply["rows"]
                .as_u64()
                .and_then(|v| usize::try_from(v).ok())
                .ok_or("missing verify rows")?;
            let inputs: Vec<u32> = serde_json::from_value(reply["tokens"].clone())
                .map_err(|_| "invalid verify tokens")?;
            let max_rows = tokens.len() + bridge_drafts;
            if rows == 0
                || rows > max_rows
                || inputs.len() != rows
                || reply["position"].as_u64() != Some(context.len() as u64)
                || inputs.iter().any(|&t| t as usize >= self.real_vocab)
                || inputs.get(..tokens.len()) != Some(tokens.as_slice())
                || (bridge_drafts == 0 && inputs != tokens)
                || bytes.len() != rows * self.real_vocab * 4
            {
                return Err("invalid protocol 2 verify reply".into());
            }
            let row_logits = bytes
                .chunks_exact(self.real_vocab * 4)
                .map(|b| self.logits_of(b))
                .collect::<Result<Vec<_>, _>>()?;
            let drafted = rows - 1;
            let mut walked = 0;
            let mut accepted = 0;
            let mut done = false;
            let mut failure = None;
            for (i, logits) in row_logits.iter().enumerate() {
                // This row's input has already been emitted, so it is safe to
                // commit even when its prediction is a stop or sampling fails.
                walked += 1;
                let next = match params.sampler.try_sample(logits, &sample_history) {
                    Ok(t) => t as u32,
                    Err(e) => {
                        failure = Some(e);
                        done = true;
                        break;
                    }
                };
                if params.stop_ids.contains(&next) {
                    done = true;
                    break;
                }
                emitted += 1;
                let stop = on_token(TokenEvent {
                    id: next,
                    logits: Some(logits),
                })
                .is_break();
                if stop || emitted == params.max_tokens {
                    done = true;
                    break;
                }
                history.push(next);
                sample_history.push(i64::from(next));
                if i + 1 < rows && next == inputs[i + 1] {
                    accepted += 1;
                } else {
                    break;
                }
            }
            // Explicit commits also drain the pass on Break/stop. No tentative
            // row after the last verified input can survive into prefix reuse.
            self.process
                .request(json!({"command":"accept", "n":walked}))?;
            self.context = context;
            self.context.extend_from_slice(&inputs[..walked]);
            self.mark = mark;
            if let Some(e) = failure {
                return Err(e);
            }
            policy.observe(drafted, accepted);
            if done {
                break;
            }
        }
        Ok(())
    }
}

#[cfg(all(test, unix))]
pub(crate) mod tests;
