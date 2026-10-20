# Porting the tokenizer + sampler to the Rust core

The Python `tokenizer.py` / `sampler.py` here are the reference. Both port to
`npu-engine/src/` with no functional changes — the formats and algorithms are
identical, only the language differs.

## Tokenizer -> `tokenizers` crate

Qwen3.6 uses a byte-level BPE (`Qwen2Tokenizer`) whose full state lives in the
model's `tokenizer.json`. The Python side loads it with the HF `tokenizers`
Python binding; Rust uses the **same library, native crate** — so encode/decode
are byte-for-byte identical to Python and to HF.

```toml
# npu-engine/Cargo.toml
[dependencies]
tokenizers = { version = "0.20", default-features = false, features = ["onig"] }
```

```rust
use tokenizers::Tokenizer;

let tk = Tokenizer::from_file(model_dir.join("tokenizer.json"))?;   // Box<dyn Error>

// encode: Qwen3.6 adds no BOS (tokenizer_config add_bos_token = false)
let enc = tk.encode(text, /*add_special_tokens=*/ false)?;
let ids: &[u32] = enc.get_ids();

// decode: byte-level BPE — decode the whole id slice, not per token
let text = tk.decode(&ids, /*skip_special_tokens=*/ false)?;
```

Constants to lift verbatim from `tokenizer.py` (source: `tokenizer_config.json`):

| name        | id       | role                                   |
|-------------|----------|----------------------------------------|
| `<|endoftext|>` | 248044 | pad / eos                            |
| `<|im_start|>`  | 248045 | chat turn open                       |
| `<|im_end|>`    | 248046 | assistant-turn stop (primary eos)    |
| `<|object_ref_end|>` | 248048 | eos                             |
| `<think>` / `</think>` | 248068 / 248069 | reasoning block          |

`EOS_TOKEN_IDS = {248044, 248046, 248048}` — stop when the sampler emits any.
Note tokenizer.json holds 248070 real tokens but the lm_head logit width is
`MODEL_VOCAB_SIZE = 248320` (padded); the sampler operates on the full 248320.

### Chat template

`chat_template.jinja` is a Jinja2 template. Options for Rust, cheapest first:

1. **Precompute the ids in Python, feed the Rust engine token ids.** The server
   layer already does chat orchestration; the Rust core can stay text/id-only.
2. **`minijinja` crate** (Jinja2-compatible) renders the packaged template. You
   must register a `raise_exception` global (the template calls it) and the
   `tojson` filter (built in). This matches how HF renders it and how
   `tokenizer.py` does it. Recommended if the Rust core must own chat formatting.
3. **Hand-write the ChatML format** for the common (system?, user, assistant)
   case: `<|im_start|>{role}\n{content}<|im_end|>\n` per message, then
   `<|im_start|>assistant\n<think>\n` for the generation prompt (or
   `<think>\n\n</think>\n\n` when thinking is disabled). Fine for a fixed server
   contract; skip it if you need tool-calls / vision, which the full template
   encodes.

Prefer (1) or (2). Only hand-roll (3) if you want zero template deps.

## Sampler -> pure Rust

`sampler.py` is ~60 lines of array math with no dependencies beyond a RNG. The
transform order is the contract — keep it: **repetition penalty -> temperature
-> top-k -> top-p -> softmax -> multinomial** (temperature 0 short-circuits to
argmax).

```rust
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

pub struct Sampler {
    pub temperature: f32,
    pub top_k: usize,          // 0 = disabled
    pub top_p: f32,            // 1.0 = disabled
    pub repetition_penalty: f32,
    rng: StdRng,
}

impl Sampler {
    pub fn new(seed: u64, temperature: f32, top_k: usize, top_p: f32, rep: f32) -> Self {
        Self { temperature, top_k, top_p, repetition_penalty: rep, rng: StdRng::seed_from_u64(seed) }
    }

    /// logits: full [248320] f32 from the CPU lm_head projection. history: ids
    /// seen so far (for repetition penalty). Returns the next token id.
    pub fn sample(&mut self, logits: &mut [f32], history: &[u32]) -> u32 {
        // 1. repetition penalty (HF CTRL convention: +logit /=pen, -logit *=pen)
        if self.repetition_penalty != 1.0 {
            for &t in history {
                let v = &mut logits[t as usize];
                *v = if *v > 0.0 { *v / self.repetition_penalty } else { *v * self.repetition_penalty };
            }
        }
        // greedy
        if self.temperature == 0.0 {
            return argmax(logits);
        }
        // 2. temperature
        for v in logits.iter_mut() { *v /= self.temperature; }
        // 3. top-k: keep k largest (partial sort / select_nth), mask rest to -inf
        // 4. top-p: sort desc, cumulate softmax, mask the tail past top_p
        // 5. softmax + multinomial via self.rng.gen::<f32>() over the CDF
        //    (see sampler.py for the exact masking; always keep >=1 token)
        // ... straightforward translation of the numpy code ...
        todo!("mechanical port of _apply_top_k / _apply_top_p / _softmax")
    }
}
```

Use `rand` + `rand::rngs::StdRng` (or `SmallRng`) for the seedable draw; there is
no numpy `default_rng` equivalence requirement — reproducibility is *within* the
Rust engine, not cross-language. The masking uses a sentinel `-1e30` (as in
Python) so masked tokens vanish under softmax.

## Where it plugs into the engine

The decode loop already computes the full [248320] f32 logits on CPU from the
NPU's final hidden (the lm_head kernel emits only the odd vocab half; the CPU
projection is a plain matmul with the cached dequantized lm_head matrix — see
`generate_npu.py::full_logits`). That vector is exactly the sampler's input, so
wiring is: `let next = sampler.sample(&mut logits, &history); history.push(next);
if EOS.contains(&next) { break; }` then embed `next` for the following step. The
tokenizer bookends it: `encode(chat_template(prompt))` in, incremental
`decode(&generated)` out (decode the whole running slice and emit the suffix —
byte-level BPE tokens can be partial UTF-8).

## Not yet portable: arbitrary prompt length on the NPU

The captured decode ELFs (`elf_000005/6/3`) bake the sequence position for the
m0c 11-token prompt. A prompt of a different length needs the per-token seqlen
patched into the 480B decode ELF (u32 pokes at byte offsets 160/184/208/232 —
documented in the plan). That patching lives in the **driver** (`decode_driver`),
not in the tokenizer/sampler, and is an existing M4 item; the CPU prefill here is
already length-general.
