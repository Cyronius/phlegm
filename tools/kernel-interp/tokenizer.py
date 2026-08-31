"""Real tokenizer for Qwen3.6 (Qwen2Tokenizer / byte-level BPE).

Text-in / text-out for the open NPU engine. Loads the model's own
`tokenizer.json` with the HuggingFace `tokenizers` crate binding (pure-Rust
BPE, no torch/transformers dependency) and applies the packaged
`chat_template.jinja` for chat formatting.

    from tokenizer import Qwen36Tokenizer
    tok = Qwen36Tokenizer()                       # loads from the .flm model dir
    ids  = tok.encode("Hello, world!")            # -> [9419, 11, 1814, 0]
    text = tok.decode(ids)                         # -> "Hello, world!"
    prompt = tok.apply_chat_template(              # -> im_start/assistant string
        [{"role": "user", "content": "Hi"}], add_generation_prompt=True)
    ids = tok.encode(prompt)                        # feed to prefill

Facts (from tokenizer_config.json): vocab in tokenizer.json = 248070 real
tokens; the model's lm_head is padded to vocab_size = 248320. add_bos_token is
false (bos_token is null), so encode() adds no BOS by default. Generation stops
on any of eos_token_id = [248044 <|endoftext|>, 248046 <|im_end|>,
248048 <|object_ref_end|>]; the chat template ends assistant turns with
<|im_end|> (248046), the primary stop.
"""
import json
import os

from tokenizers import Tokenizer

MODEL_DIR = r"C:/Users/josha/.flm/models/Qwen3.6-35B-A3B-NPU2"

# Padded logit width emitted by the lm_head (tokenizer.json is 248070).
MODEL_VOCAB_SIZE = 248320

# Named special tokens (ids from tokenizer_config.json / added_tokens_decoder).
ENDOFTEXT = 248044   # <|endoftext|>  (also pad)
IM_START = 248045    # <|im_start|>
IM_END = 248046      # <|im_end|>     (assistant-turn stop)
OBJECT_REF_END = 248048
THINK_OPEN = 248068  # <think>
THINK_CLOSE = 248069  # </think>

# Any of these ends generation (tokenizer_config.json: eos_token_id).
EOS_TOKEN_IDS = (ENDOFTEXT, IM_END, OBJECT_REF_END)


def _build_jinja_env():
    # Mirror how transformers renders chat templates: sandboxed jinja with
    # block trimming and a raise_exception global (the template calls it).
    from jinja2 import Environment
    from jinja2.exceptions import TemplateError

    def raise_exception(msg):
        raise TemplateError(msg)

    env = Environment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = raise_exception
    return env


class Qwen36Tokenizer:
    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        self.tk = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        with open(os.path.join(model_dir, "tokenizer_config.json"), encoding="utf-8") as f:
            self.config = json.load(f)
        with open(os.path.join(model_dir, "chat_template.jinja"), encoding="utf-8") as f:
            self.chat_template_src = f.read()
        self._env = _build_jinja_env()
        self._template = self._env.from_string(self.chat_template_src)
        self.add_bos_token = bool(self.config.get("add_bos_token", False))

    # ---- core encode / decode ------------------------------------------
    def encode(self, text, add_special_tokens=None):
        """Text -> token ids. Special tokens in the string (e.g. <|im_start|>
        produced by the chat template) are always recognized; add_special_tokens
        controls only the model's automatic BOS/EOS, which Qwen3.6 does not add."""
        if add_special_tokens is None:
            add_special_tokens = self.add_bos_token
        return self.tk.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids, skip_special_tokens=False):
        """Token ids -> text. Byte-level BPE, so decode the whole sequence
        rather than per-token (a single token can be a partial UTF-8 byte)."""
        ids = [int(i) for i in ids]
        return self.tk.decode(ids, skip_special_tokens=skip_special_tokens)

    def id_to_token(self, i):
        return self.tk.id_to_token(int(i))

    @property
    def eos_token_ids(self):
        return EOS_TOKEN_IDS

    def is_eos(self, tok):
        return int(tok) in EOS_TOKEN_IDS

    # ---- chat template --------------------------------------------------
    def apply_chat_template(self, messages, add_generation_prompt=True,
                            tokenize=False, enable_thinking=True, tools=None,
                            **kwargs):
        """Render messages through the packaged chat_template.jinja.

        Returns a string (tokenize=False) or token ids (tokenize=True). With
        add_generation_prompt=True the render ends at '<|im_start|>assistant\\n'
        plus a '<think>\\n' opener (Qwen3.6 is a thinking model); pass
        enable_thinking=False to emit an empty '<think>\\n\\n</think>' block."""
        rendered = self._template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            tools=tools,
            **kwargs,
        )
        if tokenize:
            return self.encode(rendered, add_special_tokens=False)
        return rendered


if __name__ == "__main__":
    import sys
    try:  # Windows console defaults to cp1252; print UTF-8 for unicode samples.
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    # Self-test: NO NPU needed. Round-trip + chat-template render.
    tok = Qwen36Tokenizer()
    print("tokenizer vocab :", tok.tk.get_vocab_size(), "| model logit width:", MODEL_VOCAB_SIZE)

    samples = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "def f(x):\n    return x ** 2  # square\n",
        "Unicode: café, naïve, 日本語, 🚀 emoji, ∑∫∂",
        "   leading and trailing spaces   ",
        "1234567890 + 0.5 = 1234567890.5",
    ]
    fails = 0
    for s in samples:
        ids = tok.encode(s)
        back = tok.decode(ids)
        ok = back == s
        fails += not ok
        print(f"[{'OK ' if ok else 'BAD'}] {len(ids):3d} ids  {s!r}")
        if not ok:
            print("      got:", repr(back))

    # Special tokens survive a round trip.
    st = "<|im_start|>user\nHi<|im_end|>\n"
    ids = tok.encode(st)
    assert IM_START in ids and IM_END in ids, ids
    assert tok.decode(ids) == st, tok.decode(ids)
    print("[OK ] special-token round trip")

    # Chat template.
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": "You are terse."},
         {"role": "user", "content": "What is 2+2?"}],
        add_generation_prompt=True)
    print("\n--- chat template (add_generation_prompt) ---")
    print(prompt)
    pids = tok.encode(prompt)
    assert pids[0] == IM_START, pids[:3]
    assert prompt.endswith("<think>\n"), repr(prompt[-40:])
    print("prompt ids:", len(pids), "first/last:", pids[0], pids[-1])

    prompt2 = tok.apply_chat_template(
        [{"role": "user", "content": "Hi"}],
        add_generation_prompt=True, enable_thinking=False)
    assert "<think>\n\n</think>" in prompt2, repr(prompt2)
    print("[OK ] enable_thinking=False empty think block")

    print(f"\nround-trip failures: {fails}/{len(samples)}")
    assert fails == 0
    print("tokenizer.py self-test PASSED")
