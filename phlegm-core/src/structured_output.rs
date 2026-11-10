//! Byte-level JSON grammar constraints. Grammar stacks advance incrementally, including inside
//! UTF-8 scalars and tokens that span several JSON terminals. No regex or model execution.
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Sym {
    Byte(u8),
    Range(u8, u8),
    Rule(usize),
}
use Sym::{Byte as B, Range as R, Rule as N};
#[derive(Default, Debug)]
struct Grammar {
    rules: Vec<Vec<Vec<Sym>>>,
}
impl Grammar {
    fn reserve(&mut self) -> usize {
        let id = self.rules.len();
        self.rules.push(vec![]);
        id
    }
    fn rule(&mut self, alternatives: Vec<Vec<Sym>>) -> usize {
        let id = self.reserve();
        self.rules[id] = alternatives;
        id
    }
    fn literal(&mut self, s: &str) -> usize {
        self.rule(vec![s.bytes().map(B).collect()])
    }
    fn lengths(&self) -> Vec<usize> {
        let mut lens = vec![usize::MAX; self.rules.len()];
        loop {
            let mut changed = false;
            for (id, alts) in self.rules.iter().enumerate() {
                let best = alts
                    .iter()
                    .map(|a| {
                        a.iter().fold(0usize, |sum, s| {
                            sum.saturating_add(match s {
                                N(r) => lens[*r],
                                _ => 1,
                            })
                        })
                    })
                    .min()
                    .unwrap_or(usize::MAX);
                if best < lens[id] {
                    lens[id] = best;
                    changed = true;
                }
            }
            if !changed {
                return lens;
            }
        }
    }
    fn prune(&mut self) {
        let lens = self.lengths();
        for alts in &mut self.rules {
            alts.retain(|a| a.iter().all(|s| !matches!(s,N(r) if lens[*r]==usize::MAX)));
        }
    }
}
#[derive(Clone, Debug)]
struct Matcher {
    grammar: Arc<Grammar>,
    stacks: Vec<Vec<Sym>>,
}
impl Matcher {
    fn new(grammar: Arc<Grammar>, root: usize) -> Self {
        let mut m = Self {
            grammar,
            stacks: vec![vec![N(root)]],
        };
        m.normalize();
        m
    }
    fn normalize(&mut self) {
        let mut pending = std::mem::take(&mut self.stacks);
        let mut seen = HashSet::new();
        while let Some(mut stack) = pending.pop() {
            if !seen.insert(stack.clone()) {
                continue;
            }
            if let Some(N(id)) = stack.last().cloned() {
                stack.pop();
                for alt in &self.grammar.rules[id] {
                    let mut s = stack.clone();
                    s.extend(alt.iter().rev().cloned());
                    pending.push(s);
                }
            } else {
                self.stacks.push(stack);
            }
        }
    }
    fn step(&mut self, byte: u8) -> bool {
        self.stacks.retain_mut(|s| {
            let yes = match s.last() {
                Some(B(b)) => *b == byte,
                Some(R(lo, hi)) => (*lo..=*hi).contains(&byte),
                _ => false,
            };
            if yes {
                s.pop();
            }
            yes
        });
        self.normalize();
        !self.stacks.is_empty()
    }
    fn bytes(&mut self, bytes: &[u8]) -> bool {
        bytes.iter().all(|&b| self.step(b))
    }
    fn complete(&self) -> bool {
        self.stacks.iter().any(Vec::is_empty)
    }
}

struct Compiler {
    g: Grammar,
    ws: usize,
    string: usize,
    number: usize,
    integer: usize,
    any: usize,
}
impl Compiler {
    fn new() -> Self {
        let mut g = Grammar::default();
        let ws = g.reserve();
        g.rules[ws] = vec![
            vec![],
            vec![B(b' '), N(ws)],
            vec![B(b'\n'), N(ws)],
            vec![B(b'\r'), N(ws)],
            vec![B(b'\t'), N(ws)],
        ];
        let hex = g.rule(vec![
            vec![R(b'0', b'9')],
            vec![R(b'a', b'f')],
            vec![R(b'A', b'F')],
        ]);
        let low = g.rule(vec![
            vec![R(b'0', b'9')],
            vec![R(b'a', b'c')],
            vec![R(b'A', b'C')],
            vec![R(b'e', b'f')],
            vec![R(b'E', b'F')],
        ]);
        let d = g.rule(vec![vec![B(b'd')], vec![B(b'D')]]);
        let non_surrogate = g.rule(vec![
            vec![N(low), N(hex), N(hex), N(hex)],
            vec![N(d), R(b'0', b'7'), N(hex), N(hex)],
        ]);
        let hi = g.rule(vec![
            vec![R(b'8', b'9')],
            vec![R(b'a', b'b')],
            vec![R(b'A', b'B')],
        ]);
        let lo = g.rule(vec![vec![R(b'c', b'f')], vec![R(b'C', b'F')]]);
        let escaped = g.rule(vec![
            vec![B(b'"')],
            vec![B(b'\\')],
            vec![B(b'/')],
            vec![B(b'b')],
            vec![B(b'f')],
            vec![B(b'n')],
            vec![B(b'r')],
            vec![B(b't')],
            vec![B(b'u'), N(non_surrogate)],
            vec![
                B(b'u'),
                N(d),
                N(hi),
                N(hex),
                N(hex),
                B(b'\\'),
                B(b'u'),
                N(d),
                N(lo),
                N(hex),
                N(hex),
            ],
        ]);
        let char_rule = g.rule(vec![
            vec![R(0x20, 0x21)],
            vec![R(0x23, 0x5b)],
            vec![R(0x5d, 0x7f)],
            vec![B(b'\\'), N(escaped)],
            vec![R(0xc2, 0xdf), R(0x80, 0xbf)],
            vec![B(0xe0), R(0xa0, 0xbf), R(0x80, 0xbf)],
            vec![R(0xe1, 0xec), R(0x80, 0xbf), R(0x80, 0xbf)],
            vec![B(0xed), R(0x80, 0x9f), R(0x80, 0xbf)],
            vec![R(0xee, 0xef), R(0x80, 0xbf), R(0x80, 0xbf)],
            vec![B(0xf0), R(0x90, 0xbf), R(0x80, 0xbf), R(0x80, 0xbf)],
            vec![R(0xf1, 0xf3), R(0x80, 0xbf), R(0x80, 0xbf), R(0x80, 0xbf)],
            vec![B(0xf4), R(0x80, 0x8f), R(0x80, 0xbf), R(0x80, 0xbf)],
        ]);
        let chars = g.reserve();
        g.rules[chars] = vec![vec![], vec![N(char_rule), N(chars)]];
        let string = g.rule(vec![vec![B(b'"'), N(chars), B(b'"')]]);
        // A finite numeric sublanguage avoids overflow even for arbitrarily long token walks.
        // It still covers signed integers and fractional numbers (enum/const use exact literals).
        let mut digits = g.rule(vec![vec![]]);
        for _ in 0..14 {
            digits = g.rule(vec![vec![], vec![R(b'0', b'9'), N(digits)]]);
        }
        let unsigned = g.rule(vec![vec![B(b'0')], vec![R(b'1', b'9'), N(digits)]]);
        let integer = g.rule(vec![vec![N(unsigned)], vec![B(b'-'), N(unsigned)]]);
        let mut fraction = g.rule(vec![vec![]]);
        for _ in 0..8 {
            fraction = g.rule(vec![vec![], vec![R(b'0', b'9'), N(fraction)]]);
        }
        let number = g.rule(vec![
            vec![N(integer)],
            vec![N(integer), B(b'.'), R(b'0', b'9'), N(fraction)],
        ]);
        let any = g.reserve();
        let mut c = Self {
            g,
            ws,
            string,
            number,
            integer,
            any,
        };
        let object = c.generic_object(any);
        let array = c.array(any);
        let t = c.g.literal("true");
        let f = c.g.literal("false");
        let null = c.g.literal("null");
        c.g.rules[any] = vec![
            vec![N(string)],
            vec![N(number)],
            vec![N(object)],
            vec![N(array)],
            vec![N(t)],
            vec![N(f)],
            vec![N(null)],
        ];
        c
    }
    fn array(&mut self, item: usize) -> usize {
        let tail = self.g.reserve();
        self.g.rules[tail] = vec![
            vec![],
            vec![B(b','), N(self.ws), N(item), N(self.ws), N(tail)],
        ];
        self.g.rule(vec![
            vec![B(b'['), N(self.ws), B(b']')],
            vec![B(b'['), N(self.ws), N(item), N(self.ws), N(tail), B(b']')],
        ])
    }
    fn generic_object(&mut self, item: usize) -> usize {
        let pair = vec![
            N(self.string),
            N(self.ws),
            B(b':'),
            N(self.ws),
            N(item),
            N(self.ws),
        ];
        let tail = self.g.reserve();
        let mut more = vec![B(b','), N(self.ws)];
        more.extend(pair.clone());
        more.push(N(tail));
        self.g.rules[tail] = vec![vec![], more];
        let mut filled = vec![B(b'{'), N(self.ws)];
        filled.extend(pair);
        filled.extend([N(tail), B(b'}')]);
        self.g
            .rule(vec![vec![B(b'{'), N(self.ws), B(b'}')], filled])
    }
    fn compile(&mut self, s: &Value) -> Result<usize, String> {
        if s == &Value::Bool(true) {
            return Ok(self.any);
        }
        if s == &Value::Bool(false) {
            return Ok(self.g.rule(vec![]));
        }
        if let Some(v) = s.get("const") {
            return Ok(if validates(s, v) {
                self.g.literal(&v.to_string())
            } else {
                self.g.rule(vec![])
            });
        }
        if let Some(values) = s.get("enum").and_then(Value::as_array) {
            let alternatives = values
                .iter()
                .filter(|v| validates(s, v))
                .map(|v| v.to_string().bytes().map(B).collect())
                .collect();
            return Ok(self.g.rule(alternatives));
        }
        if let Some(variants) = s.get("anyOf").and_then(Value::as_array) {
            let mut base = s.clone();
            base.as_object_mut().unwrap().remove("anyOf");
            let mut alternatives = vec![];
            for variant in variants {
                alternatives.push(vec![N(self.compile(&intersect(&base, variant))?)]);
            }
            return Ok(self.g.rule(alternatives));
        }
        let types = types(s);
        let mut alternatives = vec![];
        for ty in types {
            let id = match ty.as_str() {
                "string" => self.string,
                "number" => self.number,
                "integer" => self.integer,
                "boolean" => self.g.rule(vec![
                    "true".bytes().map(B).collect(),
                    "false".bytes().map(B).collect(),
                ]),
                "null" => self.g.literal("null"),
                "array" => {
                    let item = self.compile(s.get("items").unwrap_or(&Value::Bool(true)))?;
                    self.array(item)
                }
                "object" => self.object(s)?,
                _ => unreachable!(),
            };
            alternatives.push(vec![N(id)]);
        }
        Ok(self.g.rule(alternatives))
    }
    fn object(&mut self, s: &Value) -> Result<usize, String> {
        let mut props: BTreeMap<String, Value> = s
            .get("properties")
            .and_then(Value::as_object)
            .map(|p| p.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();
        let required: HashSet<&str> = s
            .get("required")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        let additional = s.get("additionalProperties").unwrap_or(&Value::Bool(true));
        for key in &required {
            props
                .entry((*key).into())
                .or_insert_with(|| additional.clone());
        }
        if props.is_empty() && additional != &Value::Bool(false) {
            let item = self.compile(additional)?;
            return Ok(self.generic_object(item));
        }
        // Canonical sorted property order. Optional properties can be omitted. Extra properties
        // permitted by a schema need not be generated; closing the object is always valid.
        let mut suffix = [self.g.rule(vec![vec![B(b'}')]]); 2];
        for (key, schema) in props.into_iter().rev() {
            let value = self.compile(&schema)?;
            let literal = self.g.literal(&json!(key).to_string());
            let mut next = [0; 2];
            for first in [false, true] {
                let mut emit = vec![];
                if !first {
                    emit.extend([B(b','), N(self.ws)]);
                }
                emit.extend([
                    N(literal),
                    N(self.ws),
                    B(b':'),
                    N(self.ws),
                    N(value),
                    N(self.ws),
                    N(suffix[0]),
                ]);
                let mut alts = vec![emit];
                if !required.contains(key.as_str()) {
                    alts.push(vec![N(suffix[first as usize])]);
                }
                next[first as usize] = self.g.rule(alts);
            }
            suffix = next;
        }
        Ok(self.g.rule(vec![vec![B(b'{'), N(self.ws), N(suffix[1])]]))
    }
}
fn types(s: &Value) -> Vec<String> {
    match s.get("type") {
        Some(Value::String(t)) => vec![t.clone()],
        Some(Value::Array(ts)) => ts
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect(),
        _ => ["object", "array", "string", "number", "boolean", "null"]
            .map(str::to_owned)
            .to_vec(),
    }
}
/// Independent semantic validator, also used to filter enum/const and schema intersections.
pub fn validates(s: &Value, v: &Value) -> bool {
    if s == &Value::Bool(true) {
        return true;
    }
    if s == &Value::Bool(false) {
        return false;
    }
    if !types(s).iter().any(|t| match t.as_str() {
        "object" => v.is_object(),
        "array" => v.is_array(),
        "string" => v.is_string(),
        "number" => v.is_number(),
        "integer" => {
            v.as_i64().is_some()
                || v.as_u64().is_some()
                || v.as_f64().is_some_and(|n| n.fract() == 0.0)
        }
        "boolean" => v.is_boolean(),
        "null" => v.is_null(),
        _ => false,
    }) {
        return false;
    }
    if s.get("const").is_some_and(|c| c != v)
        || s.get("enum")
            .and_then(Value::as_array)
            .is_some_and(|a| !a.contains(v))
    {
        return false;
    }
    if s.get("anyOf")
        .and_then(Value::as_array)
        .is_some_and(|a| !a.iter().any(|c| validates(c, v)))
    {
        return false;
    }
    if let Some(obj) = v.as_object() {
        if s.get("required")
            .and_then(Value::as_array)
            .is_some_and(|a| {
                a.iter()
                    .any(|k| !obj.contains_key(k.as_str().unwrap_or("")))
            })
        {
            return false;
        }
        for (k, v) in obj {
            let child = s
                .get("properties")
                .and_then(|p| p.get(k))
                .or_else(|| s.get("additionalProperties"))
                .unwrap_or(&Value::Bool(true));
            if !validates(child, v) {
                return false;
            }
        }
    }
    if let Some(a) = v.as_array() {
        if let Some(items) = s.get("items") {
            if !a.iter().all(|v| validates(items, v)) {
                return false;
            }
        }
    }
    true
}
fn intersect(a: &Value, b: &Value) -> Value {
    if a == &Value::Bool(false) || b == &Value::Bool(false) {
        return json!(false);
    }
    if a == &Value::Bool(true) {
        return b.clone();
    }
    if b == &Value::Bool(true) {
        return a.clone();
    }
    // Distribute nested anyOf before merging the remaining constraints.
    for (left, right) in [(a, b), (b, a)] {
        if let Some(variants) = left.get("anyOf").and_then(Value::as_array) {
            let mut base = left.clone();
            base.as_object_mut().unwrap().remove("anyOf");
            return json!({"anyOf":variants.iter().map(|v|intersect(&intersect(&base,v),right)).collect::<Vec<_>>()});
        }
    }
    for (left, right) in [(a, b), (b, a)] {
        let finite = left
            .get("enum")
            .and_then(Value::as_array)
            .cloned()
            .or_else(|| left.get("const").map(|v| vec![v.clone()]));
        if let Some(values) = finite {
            return json!({"enum":values.into_iter().filter(|v|validates(left,v)&&validates(right,v)).collect::<Vec<_>>()});
        }
    }
    let at = types(a);
    let bt = types(b);
    let mut ts = vec![];
    for t in at {
        if bt.contains(&t) {
            ts.push(t);
        } else if t == "integer" && bt.contains(&"number".into()) {
            ts.push(t);
        } else if t == "number" && bt.contains(&"integer".into()) {
            ts.push("integer".into());
        }
    }
    if ts.is_empty() {
        return json!(false);
    }
    let mut out = json!({"type":ts});
    let mut required = vec![];
    for s in [a, b] {
        if let Some(r) = s.get("required").and_then(Value::as_array) {
            for k in r {
                if !required.contains(k) {
                    required.push(k.clone());
                }
            }
        }
    }
    out["required"] = json!(required);
    let mut props = serde_json::Map::new();
    let empty = json!({});
    let ap = a.get("properties").unwrap_or(&empty);
    let bp = b.get("properties").unwrap_or(&empty);
    let keys: HashSet<&String> = ap
        .as_object()
        .into_iter()
        .flat_map(|o| o.keys())
        .chain(bp.as_object().into_iter().flat_map(|o| o.keys()))
        .collect();
    let aa = a.get("additionalProperties").unwrap_or(&Value::Bool(true));
    let ba = b.get("additionalProperties").unwrap_or(&Value::Bool(true));
    for k in keys {
        props.insert(
            k.clone(),
            intersect(ap.get(k).unwrap_or(aa), bp.get(k).unwrap_or(ba)),
        );
    }
    out["properties"] = json!(props);
    out["additionalProperties"] = intersect(aa, ba);
    out["items"] = intersect(
        a.get("items").unwrap_or(&Value::Bool(true)),
        b.get("items").unwrap_or(&Value::Bool(true)),
    );
    out
}

fn check_schema(s: &Value, path: &str, depth: usize) -> Result<(), String> {
    if depth > 48 {
        return Err(format!("schema nesting exceeds 48 at {path}"));
    }
    if s.is_boolean() {
        return Ok(());
    }
    let o = s
        .as_object()
        .ok_or_else(|| format!("schema at {path} must be an object or boolean"))?;
    for (k, v) in o {
        match k.as_str() {
            "title" | "description" | "default" | "examples" | "$schema" | "$id" => {}
            "type" => {
                let ts = match v {
                    Value::String(t) => vec![t.as_str()],
                    Value::Array(a) if !a.is_empty() => {
                        a.iter().map(|t| t.as_str().unwrap_or("")).collect()
                    }
                    _ => return Err(format!("invalid type at {path}")),
                };
                if ts.iter().any(|t| {
                    !matches!(
                        *t,
                        "object" | "array" | "string" | "number" | "integer" | "boolean" | "null"
                    )
                }) {
                    return Err(format!("unsupported type at {path}"));
                }
            }
            "properties" => {
                let p = v
                    .as_object()
                    .ok_or_else(|| format!("properties at {path} must be an object"))?;
                if p.len() > 256 {
                    return Err(format!("properties exceeds 256 at {path}"));
                }
                for (k, c) in p {
                    check_schema(c, &format!("{path}.properties.{k}"), depth + 1)?;
                }
            }
            "required" => {
                if !v.as_array().is_some_and(|a| a.iter().all(Value::is_string)) {
                    return Err(format!("required at {path} must be an array of strings"));
                }
            }
            "enum" => {
                if !v.as_array().is_some_and(|a| !a.is_empty()) {
                    return Err(format!("enum at {path} must be nonempty"));
                }
            }
            "const" => {}
            "items" | "additionalProperties" => check_schema(v, &format!("{path}.{k}"), depth + 1)?,
            "anyOf" => {
                let a = v
                    .as_array()
                    .filter(|a| !a.is_empty())
                    .ok_or_else(|| format!("anyOf at {path} must be nonempty"))?;
                if a.len() > 64 {
                    return Err(format!("anyOf exceeds 64 at {path}"));
                }
                for (i, c) in a.iter().enumerate() {
                    check_schema(c, &format!("{path}.anyOf[{i}]"), depth + 1)?;
                }
            }
            _ => return Err(format!("unsupported schema feature {path}.{k}")),
        }
    }
    Ok(())
}

#[derive(Debug)]
struct Node {
    children: BTreeMap<u8, usize>,
    ids: Vec<u32>,
}
/// Shared across requests; each token stores exact decoded bytes (not lossy Unicode text).
#[derive(Debug)]
pub struct Vocabulary {
    bytes: Vec<Option<Vec<u8>>>,
    trie: Vec<Node>,
    stops: HashSet<u32>,
}
impl Vocabulary {
    pub fn new(bytes: Vec<Option<Vec<u8>>>, stops: HashSet<u32>) -> Self {
        let mut trie = vec![Node {
            children: BTreeMap::new(),
            ids: vec![],
        }];
        for (id, bytes) in bytes.iter().enumerate() {
            if stops.contains(&(id as u32)) {
                continue;
            }
            if let Some(bs) = bytes {
                if bs.is_empty() {
                    continue;
                }
                let mut at = 0;
                for &b in bs {
                    at = if let Some(&i) = trie[at].children.get(&b) {
                        i
                    } else {
                        let i = trie.len();
                        trie.push(Node {
                            children: BTreeMap::new(),
                            ids: vec![],
                        });
                        trie[at].children.insert(b, i);
                        i
                    };
                }
                trie[at].ids.push(id as u32);
            }
        }
        Self { bytes, trie, stops }
    }
    pub fn from_tokenizer(t: &tokenizers::Tokenizer, stops: HashSet<u32>) -> Result<Self, String> {
        let decoder = serde_json::to_value(t.get_decoder()).map_err(|e| e.to_string())?;
        let byte_level = decoder["type"] == "ByteLevel";
        let plain_fallback = decoder["type"] == "ByteFallback";
        let fallback = decoder["type"] == "Sequence"
            && decoder["decoders"]
                .as_array()
                .is_some_and(|a| a.iter().any(|d| d["type"] == "ByteFallback"));
        if !byte_level && !fallback && !plain_fallback {
            return Err(format!(
                "unsupported structured-output tokenizer decoder: {decoder}"
            ));
        }
        let mut inverse = std::collections::HashMap::new();
        let mut next = 256;
        for byte in 0u16..256 {
            let c = if (33..=126).contains(&byte)
                || (161..=172).contains(&byte)
                || (174..=255).contains(&byte)
            {
                byte as u32
            } else {
                let c = next;
                next += 1;
                c
            };
            inverse.insert(char::from_u32(c).unwrap(), byte as u8);
        }
        let specials = t.get_added_tokens_decoder();
        let vocab = t.get_vocab(true);
        let len = vocab.values().max().map_or(0, |v| *v as usize + 1);
        let mut bytes = vec![None; len];
        for (token, id) in vocab {
            if specials.get(&id).is_some_and(|t| t.special) {
                continue;
            }
            bytes[id as usize] = if byte_level {
                token.chars().map(|c| inverse.get(&c).copied()).collect()
            } else if token.starts_with("<0x") && token.ends_with('>') && token.len() == 6 {
                u8::from_str_radix(&token[3..5], 16).ok().map(|b| vec![b])
            } else {
                Some(token.replace('▁', " ").into_bytes())
            };
        }
        Ok(Self::new(bytes, stops))
    }
    pub fn token_bytes(&self, id: u32) -> Option<&[u8]> {
        self.bytes.get(id as usize).and_then(|b| b.as_deref())
    }
    pub fn len(&self) -> usize {
        self.bytes.len()
    }
}
/// Determinize parser-stack sets lazily. Most bytes inside a string return to the same
/// state; vocabulary traversal must not expand the same grammar rules for every token.
#[derive(Debug, Default)]
struct Transitions {
    states: Vec<Matcher>,
    ids: HashMap<Vec<Vec<Sym>>, usize>,
    edges: HashMap<(usize, u8), Option<usize>>,
}
impl Transitions {
    fn intern(&mut self, state: Matcher) -> usize {
        if let Some(&id) = self.ids.get(&state.stacks) {
            return id;
        }
        let id = self.states.len();
        self.ids.insert(state.stacks.clone(), id);
        self.states.push(state);
        id
    }
    fn step(&mut self, at: usize, byte: u8) -> Option<usize> {
        if let Some(&next) = self.edges.get(&(at, byte)) {
            return next;
        }
        let mut next = self.states[at].clone();
        let next = if next.step(byte) {
            Some(self.intern(next))
        } else {
            None
        };
        self.edges.insert((at, byte), next);
        next
    }
}

#[derive(Debug)]
pub struct Constraint {
    state: Matcher,
    vocab: Arc<Vocabulary>,
    transitions: Transitions,
    masks: HashMap<(usize, usize), Vec<bool>>,
}
impl Constraint {
    pub fn from_response_format(format: &Value, vocab: Arc<Vocabulary>) -> Result<Self, String> {
        let schema = match format.get("type").and_then(Value::as_str) {
            Some("json_object") => json!({"type":"object"}),
            Some("json_schema") => {
                let spec = format
                    .get("json_schema")
                    .filter(|v| v.is_object())
                    .ok_or("response_format.json_schema must be an object")?;
                if !spec.get("name").is_some_and(Value::is_string) {
                    return Err("response_format.json_schema.name is required".into());
                }
                if spec.get("strict").is_some_and(|v| !v.is_boolean()) {
                    return Err("response_format.json_schema.strict must be boolean".into());
                }
                spec.get("schema")
                    .cloned()
                    .ok_or("response_format.json_schema.schema is required")?
            }
            _ => {
                return Err("response_format.type must be json_object, json_schema, or text".into())
            }
        };
        Self::new(&schema, vocab)
    }
    pub fn new(schema: &Value, vocab: Arc<Vocabulary>) -> Result<Self, String> {
        check_schema(schema, "schema", 0)?;
        let mut c = Compiler::new();
        let root = c.compile(schema)?;
        let root = c.g.rule(vec![vec![N(c.ws), N(root), N(c.ws)]]);
        c.g.prune();
        let state = Matcher::new(Arc::new(c.g), root);
        if state.stacks.is_empty() {
            return Err("schema has no satisfiable output".into());
        }
        Ok(Self {
            state,
            vocab,
            transitions: Transitions::default(),
            masks: HashMap::new(),
        })
    }
    pub fn complete(&self) -> bool {
        self.state.complete()
    }
    pub fn allowed(&mut self, width: usize) -> Vec<bool> {
        let root = self.transitions.intern(self.state.clone());
        if let Some(mask) = self.masks.get(&(root, width)) {
            return mask.clone();
        }
        let mut allowed = vec![false; width];
        if self.complete() {
            for &id in &self.vocab.stops {
                if let Some(a) = allowed.get_mut(id as usize) {
                    *a = true;
                }
            }
        }
        let mut pending = vec![(0, root)];
        while let Some((at, state)) = pending.pop() {
            for &id in &self.vocab.trie[at].ids {
                if let Some(a) = allowed.get_mut(id as usize) {
                    *a = true;
                }
            }
            for (&b, &next) in &self.vocab.trie[at].children {
                if let Some(state) = self.transitions.step(state, b) {
                    pending.push((next, state));
                }
            }
        }
        if self.masks.len() >= 32 {
            self.masks.clear();
        }
        self.masks.insert((root, width), allowed.clone());
        allowed
    }
    pub fn advance(&mut self, id: usize) -> bool {
        if self.vocab.stops.contains(&(id as u32)) {
            return self.complete();
        }
        self.vocab
            .bytes
            .get(id)
            .and_then(Option::as_ref)
            .is_some_and(|b| !b.is_empty() && self.state.bytes(b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn vocab() -> Arc<Vocabulary> {
        Arc::new(Vocabulary::new(
            (0..256)
                .map(|b| Some(vec![b as u8]))
                .chain([None])
                .collect(),
            [256].into_iter().collect(),
        ))
    }
    fn accepts(s: &Value, text: &str) -> bool {
        let mut c = Constraint::new(s, vocab()).unwrap();
        text.bytes().all(|b| c.advance(b as usize)) && c.complete()
    }
    fn completion(m: &Matcher) -> Vec<u8> {
        let lens = m.grammar.lengths();
        let weight = |s: &Sym| match s {
            N(r) => lens[*r],
            _ => 1,
        };
        let mut stack = m
            .stacks
            .iter()
            .min_by_key(|s| s.iter().map(&weight).sum::<usize>())
            .unwrap()
            .clone();
        let mut out = vec![];
        while let Some(s) = stack.pop() {
            match s {
                B(b) | R(b, _) => out.push(b),
                N(r) => {
                    let a = m.grammar.rules[r]
                        .iter()
                        .min_by_key(|a| a.iter().map(&weight).sum::<usize>())
                        .unwrap();
                    stack.extend(a.iter().rev().cloned());
                }
            }
        }
        out
    }
    fn walk(
        schema: &Value,
        vocab: Arc<Vocabulary>,
        seed: u64,
        tk: Option<&tokenizers::Tokenizer>,
    ) -> String {
        let mut c = Constraint::new(schema, vocab.clone()).unwrap();
        let mut text = vec![];
        let mut ids = vec![];
        let mut rng = seed;
        for _ in 0..12 {
            let allowed = c.allowed(
                vocab
                    .len()
                    .max(vocab.stops.iter().max().map_or(0, |i| *i as usize + 1)),
            );
            for &stop in &vocab.stops {
                assert_eq!(allowed[stop as usize], c.complete());
            }
            let choices: Vec<usize> = allowed
                .iter()
                .enumerate()
                .filter(|(id, a)| {
                    **a && vocab
                        .bytes
                        .get(*id)
                        .and_then(Option::as_ref)
                        .is_some_and(|b| !b.is_empty())
                })
                .map(|(i, _)| i)
                .collect();
            if choices.is_empty() || c.complete() {
                break;
            }
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let id = choices[(rng >> 32) as usize % choices.len()];
            text.extend(vocab.bytes[id].as_ref().unwrap());
            assert!(c.advance(id));
            ids.push(id as u32);
        }
        // Complete the random valid prefix using the grammar's shortest accepting suffix.
        for b in completion(&c.state) {
            let id = vocab
                .bytes
                .iter()
                .position(|bs| bs.as_deref() == Some(&[b]))
                .expect("tokenizer lacks a byte fallback");
            assert!(c.allowed(vocab.len())[id]);
            assert!(c.advance(id));
            ids.push(id as u32);
            text.push(b);
        }
        assert!(c.complete());
        let text = String::from_utf8(text).unwrap();
        let value: Value = serde_json::from_str(&text).unwrap();
        assert!(validates(schema, &value), "{text}");
        if let Some(tk) = tk {
            let decoded = tk.decode(&ids, false).unwrap();
            let value: Value = serde_json::from_str(&decoded).unwrap();
            assert!(validates(schema, &value), "{decoded}");
        }
        text
    }
    fn schemas() -> Vec<Value> {
        vec![
            json!({"type":"object"}),
            json!({"type":"object","properties":{"a":{"type":"array","items":{"anyOf":[{"type":"boolean"},{"type":"null"},{"type":"number"}]}},"b":{"type":"object","properties":{"s":{"enum":["hi","é"]}},"required":["s"],"additionalProperties":false}},"required":["a","b"],"additionalProperties":false}),
            json!({"anyOf":[{"const":{"x":1}},{"type":"array","items":{"type":"string"}}]}),
            json!({"type":["integer","null"]}),
            json!({"type":"object","required":["x"],"additionalProperties":{"type":"string"}}),
        ]
    }
    #[test]
    fn grammar_types_nested_objects_unions_and_eos() {
        for schema in schemas() {
            for seed in 1..5 {
                walk(&schema, vocab(), seed, None);
            }
        }
        let s = json!({"type":"object","properties":{"x":{"type":"integer"}},"required":["x"],"additionalProperties":false});
        assert!(accepts(&s, " {\"x\":-123} \n"));
        for bad in ["{}", "{\"x\":true}", "{\"x\":1,\"extra\":0}", "{\"x\":1.5}"] {
            assert!(!accepts(&s, bad), "{bad}");
        }
        for bad in [
            br#""\uD800""#.as_slice(),
            b"\"\xc0\x80\"",
            b"\"\xed\xa0\x80\"",
        ] {
            let mut c = Constraint::new(&json!({"type":"string"}), vocab()).unwrap();
            assert!(!c.state.bytes(bad));
        }
        assert!(accepts(&json!({"type":"string"}), r#""\uD83D\uDE42""#));
    }
    #[test]
    fn unsupported_features_and_unsatisfiable_schemas_are_errors() {
        for key in [
            "$ref",
            "oneOf",
            "allOf",
            "pattern",
            "minLength",
            "maximum",
            "prefixItems",
            "not",
        ] {
            let s = json!({key:1});
            assert!(Constraint::new(&s, vocab()).unwrap_err().contains(key));
        }
        for s in [
            json!(false),
            json!({"type":"string","const":1}),
            json!({"type":"object","required":["x"],"additionalProperties":false}),
        ] {
            assert!(Constraint::new(&s, vocab()).is_err(), "{s}");
        }
        assert!(Constraint::from_response_format(
            &json!({"type":"json_schema","json_schema":{"name":"x","strict":"yes","schema":{}}}),
            vocab()
        )
        .is_err());
        assert!(accepts(
            &json!({"type":"object","anyOf":[{"properties":{"x":{"const":1}},"required":["x"]},{"const":null}]}),
            "{\"x\":1}"
        ));
    }
    #[test]
    fn cached_masks_equal_independent_token_walks() {
        let vocab = vocab();
        for schema in schemas() {
            let mut c = Constraint::new(&schema, vocab.clone()).unwrap();
            let mut seed = 19u64;
            for _ in 0..32 {
                let mask = c.allowed(257);
                assert_eq!(mask, c.allowed(257));
                assert_eq!(&mask[..128], c.allowed(128).as_slice());
                for id in 0..257 {
                    let valid = if id == 256 {
                        c.complete()
                    } else {
                        let mut state = c.state.clone();
                        state.bytes(vocab.bytes[id].as_ref().unwrap())
                    };
                    assert_eq!(mask[id], valid, "token {id}");
                }
                let choices: Vec<_> = mask
                    .iter()
                    .enumerate()
                    .filter(|(i, a)| **a && *i != 256)
                    .map(|(i, _)| i)
                    .collect();
                if choices.is_empty() || c.complete() {
                    break;
                }
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                assert!(c.advance(choices[(seed >> 32) as usize % choices.len()]));
            }
        }
    }

    #[test]
    fn tokens_spanning_boundaries_and_utf8_are_masked_as_bytes() {
        let bytes = [
            Some(b"{\"x\":\"".to_vec()),
            Some(vec![0xf0, 0x9f]),
            Some(vec![0x99, 0x82]),
            Some(b"\"}".to_vec()),
            None,
            Some(b"oops".to_vec()),
        ]
        .to_vec();
        let mut c=Constraint::new(&json!({"type":"object","properties":{"x":{"type":"string"}},"required":["x"],"additionalProperties":false}),Arc::new(Vocabulary::new(bytes,[4].into_iter().collect()))).unwrap();
        for id in [0, 1, 2, 3] {
            assert!(c.allowed(6)[id]);
            assert!(!c.allowed(6)[4]);
            assert!(c.advance(id));
        }
        assert!(c.allowed(6)[4]);
        assert!(!c.allowed(6)[5]);
    }
    #[test]
    #[ignore = "requires real registry tokenizers under /home/cyrus/models"]
    fn every_registry_tokenizer_random_walks() {
        let registry: Value =
            serde_json::from_str(include_str!("../config/model_list.json")).unwrap();
        let mut count = 0;
        for family in registry["models"].as_object().unwrap().values() {
            for entry in family.as_object().unwrap().values() {
                let name = entry["name"].as_str().unwrap();
                let path = std::path::Path::new("/home/cyrus/models")
                    .join(name)
                    .join("tokenizer.json");
                let tk = tokenizers::Tokenizer::from_file(&path).unwrap();
                let stop = tk.get_vocab_size(true) as u32;
                let vocabulary = Arc::new(
                    Vocabulary::from_tokenizer(&tk, [stop].into_iter().collect()).unwrap(),
                );
                for s in schemas() {
                    for seed in 1..3 {
                        walk(&s, vocabulary.clone(), seed, Some(&tk));
                    }
                }
                count += 1;
                println!("{name}: {} vocabulary entries passed", vocabulary.len());
            }
        }
        assert_eq!(count, 18);
    }
    #[test]
    #[ignore = "offline release microbenchmark; run under dibs --measure"]
    fn largest_vocabulary_mask_cost() {
        let path = "/home/cyrus/models/gemma-4-26B-A4B-it-NPU2/tokenizer.json";
        let tk = tokenizers::Tokenizer::from_file(path).unwrap();
        let stop = tk.get_vocab_size(true) as u32;
        let vocab =
            Arc::new(Vocabulary::from_tokenizer(&tk, [stop].into_iter().collect()).unwrap());
        let schema = json!({"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false});
        fn mask_logits(c: &mut Constraint, logits: &mut [f32]) {
            let mask = c.allowed(logits.len());
            for (v, keep) in logits.iter_mut().zip(mask) {
                if !keep {
                    *v = f32::NEG_INFINITY;
                }
            }
            std::hint::black_box(logits);
        }
        let mut logits = vec![0.0; vocab.len()];
        let mut cold = vec![];
        for _ in 0..3 {
            let mut c = Constraint::new(&schema, vocab.clone()).unwrap();
            assert!(c.state.bytes(b"{\"city\":\""));
            let start = std::time::Instant::now();
            mask_logits(&mut c, &mut logits);
            cold.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        println!("first mask at string state ms={cold:?}");
        let mut c = Constraint::new(&schema, vocab.clone()).unwrap();
        assert!(c.state.bytes(b"{\"city\":\""));
        for _ in 0..3 {
            mask_logits(&mut c, &mut logits);
        }
        let mut samples = vec![];
        for _ in 0..9 {
            let start = std::time::Instant::now();
            mask_logits(&mut c, &mut logits);
            samples.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        println!(
            "vocab={} mask inside unconstrained JSON string ms={samples:?}",
            vocab.len()
        );
    }
}
