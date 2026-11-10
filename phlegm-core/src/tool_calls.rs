//! Model-template tool protocols. Never execute model text: parsers produce JSON data only.
use serde_json::{json, Map, Value};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Format {
    Function,
    Tagged,
    Gemma,
    Hy,
    Lfm,
    Harmony,
}
impl Format {
    pub fn of(template: &str) -> Option<Self> {
        if template.contains("<|channel|>") {
            Some(Self::Harmony)
        } else if template.contains("<|tool_call_start|>") {
            Some(Self::Lfm)
        } else if template.contains("<|tool_call>") {
            Some(Self::Gemma)
        } else if template.contains("<tool_calls>") {
            Some(Self::Hy)
        } else if template.contains("<function=") {
            Some(Self::Function)
        } else if template.contains("<tool_call>") {
            Some(Self::Tagged)
        } else {
            None
        }
    }
    /// Text safe to stream before the first possible call. After a complete
    /// marker, buffer through final validation so malformed calls remain text.
    pub fn stream_prefix(self, text: &str) -> &str {
        if self == Self::Harmony {
            // Headers can put the recipient before or after the channel. Hold
            // an unfinished header until its message delimiter resolves it.
            let mut offset = 0;
            loop {
                let rest = &text[offset..];
                let Some(end) = rest.find("<|message|>") else {
                    return &text[..offset];
                };
                if rest[..end].contains("to=functions.") {
                    return &text[..offset];
                }
                let body = offset + end + "<|message|>".len();
                let Some(next) = text[body..].find("<|start|>") else {
                    return text;
                };
                offset = body + next;
            }
        }
        let markers: &[&str] = match self {
            Self::Function | Self::Tagged => &["<tool_call>"],
            Self::Gemma => &["<|tool_call>"],
            Self::Hy => &["<tool_calls>", "<｜tool▁calls▁begin｜>"],
            // The installed LFM model also emits bare function lists.
            Self::Lfm => &["<|tool_call_start|>", "["],
            Self::Harmony => unreachable!(),
        };
        if let Some(at) = markers.iter().filter_map(|m| text.find(m)).min() {
            return &text[..at];
        }
        let end = markers
            .iter()
            .flat_map(|m| {
                (1..m.len().min(text.len() + 1)).rev().find(|&n| {
                    text.is_char_boundary(text.len() - n) && m.starts_with(&text[text.len() - n..])
                })
            })
            .max()
            .map_or(text.len(), |n| text.len() - n);
        &text[..end]
    }
    pub fn parse(self, text: &str) -> Parsed {
        let parsed = match self {
            Self::Harmony => harmony(text),
            Self::Lfm if !text.contains("<|tool_call_start|>") => text.rfind('[').and_then(|at| {
                lfm(text[at..].trim()).map(|calls| Parsed {
                    content: text[..at].into(),
                    calls,
                })
            }),
            Self::Hy if text.contains("<｜tool▁calls▁begin｜>") => deepseek(text),
            _ => {
                let (open, close) = match self {
                    Self::Function | Self::Tagged => ("<tool_call>", "</tool_call>"),
                    Self::Gemma => ("<|tool_call>", "<tool_call|>"),
                    Self::Hy => ("<tool_calls>", "</tool_calls>"),
                    Self::Lfm => ("<|tool_call_start|>", "<|tool_call_end|>"),
                    Self::Harmony => unreachable!(),
                };
                blocks(text, open, close, |body| match self {
                    Self::Function => function(body).map(|c| vec![c]),
                    Self::Tagged => tagged(body).map(|c| vec![c]),
                    Self::Gemma => gemma(body).map(|c| vec![c]),
                    Self::Hy => hy(body),
                    Self::Lfm => lfm(body),
                    _ => None,
                })
            }
        };
        parsed.unwrap_or_else(|| Parsed {
            content: text.into(),
            calls: vec![],
        })
    }
}
#[derive(Debug)]
pub struct Parsed {
    pub content: String,
    pub calls: Vec<Value>,
}
fn call(name: &str, args: Value) -> Option<Value> {
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_alphanumeric() || matches!(c, '_' | '-' | '.'))
        || !args.is_object()
    {
        return None;
    }
    Some(json!({"type":"function", "function":{"name":name,"arguments":args.to_string()}}))
}
fn blocks(
    text: &str,
    open: &str,
    close: &str,
    mut parse: impl FnMut(&str) -> Option<Vec<Value>>,
) -> Option<Parsed> {
    let mut rest = text;
    let mut content = String::new();
    let mut calls = vec![];
    while let Some((before, after)) = rest.split_once(open) {
        let (body, tail) = after.split_once(close)?;
        content.push_str(before);
        calls.extend(parse(body.trim())?);
        rest = tail;
    }
    content.push_str(rest);
    Some(Parsed { content, calls })
}
fn scalar(s: &str) -> Value {
    serde_json::from_str(s.trim()).unwrap_or_else(|_| Value::String(s.trim_matches('\n').into()))
}
fn pairs(mut s: &str, ko: &str, kc: &str, vo: &str, vc: &str) -> Option<Map<String, Value>> {
    let mut args = Map::new();
    while !s.trim().is_empty() {
        s = s.trim_start().strip_prefix(ko)?;
        let (key, rest) = s.split_once(kc)?;
        let (value, rest) = rest.trim_start().strip_prefix(vo)?.split_once(vc)?;
        if args.insert(key.trim().into(), scalar(value)).is_some() {
            return None;
        }
        s = rest;
    }
    Some(args)
}
fn function(body: &str) -> Option<Value> {
    let (name, mut rest) = body.strip_prefix("<function=")?.split_once('>')?;
    rest = rest.trim_end().strip_suffix("</function>")?;
    let mut args = Map::new();
    while !rest.trim().is_empty() {
        let (key, value) = rest
            .trim_start()
            .strip_prefix("<parameter=")?
            .split_once('>')?;
        let (value, tail) = value.split_once("</parameter>")?;
        if args.insert(key.into(), scalar(value)).is_some() {
            return None;
        }
        rest = tail;
    }
    call(name, Value::Object(args))
}
fn tagged(body: &str) -> Option<Value> {
    if body.starts_with('{') {
        let v: Value = serde_json::from_str(body).ok()?;
        let v = v.get("function").unwrap_or(&v);
        let args = v.get("arguments")?;
        let args = if let Some(s) = args.as_str() {
            serde_json::from_str(s).ok()?
        } else {
            args.clone()
        };
        return call(v.get("name")?.as_str()?, args);
    }
    let at = body.find('<').unwrap_or(body.len());
    let rest = &body[at..];
    let args = if rest.contains("<param_key>") {
        pairs(
            rest,
            "<param_key>",
            "</param_key>",
            "<param_value>",
            "</param_value>",
        )?
    } else {
        pairs(
            rest,
            "<arg_key>",
            "</arg_key>",
            "<arg_value>",
            "</arg_value>",
        )?
    };
    call(body[..at].trim(), Value::Object(args))
}
fn hy(body: &str) -> Option<Vec<Value>> {
    let p = blocks(body, "<tool_call>", "</tool_call>", |b| {
        let (name, args) = b.split_once("<tool_sep>")?;
        Some(vec![call(
            name.trim(),
            Value::Object(pairs(
                args,
                "<arg_key>",
                "</arg_key>",
                "<arg_value>",
                "</arg_value>",
            )?),
        )?])
    })?;
    if p.content.trim().is_empty() && !p.calls.is_empty() {
        Some(p.calls)
    } else {
        None
    }
}
fn deepseek(text: &str) -> Option<Parsed> {
    blocks(
        text,
        "<｜tool▁calls▁begin｜>",
        "<｜tool▁calls▁end｜>",
        |body| {
            let p = blocks(
                body,
                "<｜tool▁call▁begin｜>",
                "<｜tool▁call▁end｜>",
                |b| {
                    let (header, args) = b.split_once("```json")?;
                    let (_, name) = header.split_once("<｜tool▁sep｜>")?;
                    Some(vec![call(
                        name.trim(),
                        serde_json::from_str(args.trim().strip_suffix("```")?.trim()).ok()?,
                    )?])
                },
            )?;
            if p.content.trim().is_empty() {
                Some(p.calls)
            } else {
                None
            }
        },
    )
}
fn gemma(body: &str) -> Option<Value> {
    let (name, args) = body.strip_prefix("call:")?.split_once('{')?;
    let mut p = Literal::new(&format!("{{{args}"));
    let value = p.value()?;
    p.end()?;
    call(name, value)
}
fn lfm(body: &str) -> Option<Vec<Value>> {
    let mut p = Literal::new(body);
    p.take('[')?;
    let mut calls = vec![];
    if p.peek() == Some(']') {
        return None;
    }
    loop {
        let name = p.ident()?;
        p.take('(')?;
        let mut args = Map::new();
        if p.peek() != Some(')') {
            loop {
                let key = p.ident()?;
                p.take('=')?;
                let value = p.value()?;
                if args.insert(key, value).is_some() {
                    return None;
                }
                if p.peek() != Some(',') {
                    break;
                }
                p.take(',')?;
            }
        }
        p.take(')')?;
        calls.push(call(&name, Value::Object(args))?);
        if p.peek() != Some(',') {
            break;
        }
        p.take(',')?;
    }
    p.take(']')?;
    p.end()?;
    Some(calls)
}
fn harmony(text: &str) -> Option<Parsed> {
    // The prompt already supplied `<|start|>assistant`; the first header may start at ` to=`.
    let mut rest = text;
    let mut content = String::new();
    let mut calls = vec![];
    while let Some(at) = rest.find("to=functions.") {
        let prefix = &rest[..at];
        let header_start = prefix.rfind("<|start|>assistant").unwrap_or_else(|| {
            prefix
                .rfind("<|channel|>")
                .unwrap_or(prefix.trim_end().len())
        });
        content.push_str(&prefix[..header_start]);
        let (header, body) = rest[at + "to=functions.".len()..].split_once("<|message|>")?;
        let name = header
            .split(|c: char| c == '<' || c.is_whitespace())
            .next()?;
        let end = body.find("<|call|>").unwrap_or(body.len());
        calls.push(call(name, serde_json::from_str(body[..end].trim()).ok()?)?);
        rest = if end == body.len() {
            ""
        } else {
            &body[end + "<|call|>".len()..]
        };
    }
    content.push_str(rest);
    Some(Parsed { content, calls })
}

/// Separate harmony message bodies after extracting tool calls. Commentary before a call is
/// user-facing content too. Keep malformed recipient frames visible instead of dropping them.
pub fn harmony_parts(mut text: &str) -> (String, String) {
    let (mut reasoning, mut content) = (String::new(), String::new());
    while !text.is_empty() {
        let frame = text.strip_prefix("<|start|>assistant").unwrap_or(text);
        let Some((header, tail)) = frame.split_once("<|message|>") else {
            content.push_str(frame);
            break;
        };
        let end = ["<|end|>", "<|return|>", "<|call|>"]
            .into_iter()
            .filter_map(|m| tail.find(m).map(|at| (at, m.len())))
            .min_by_key(|(at, _)| *at);
        let body = &tail[..end.map_or(tail.len(), |(at, _)| at)];
        if header.contains("to=") {
            content.push_str(header);
            content.push_str("<|message|>");
            content.push_str(body);
        } else if header.contains("<|channel|>analysis") {
            reasoning.push_str(body);
        } else {
            content.push_str(body);
        }
        text = end.map_or("", |(at, len)| &tail[at + len..]);
    }
    (reasoning.trim().into(), content.trim().into())
}

/// Incremental harmony bodies, including commentary. Hold incomplete headers
/// and end markers; preserve trailing whitespace so already-sent text never shrinks.
pub fn harmony_stream_parts(mut text: &str, done: bool) -> (String, String) {
    let (mut reasoning, mut content) = (String::new(), String::new());
    while !text.is_empty() {
        let frame = text.strip_prefix("<|start|>assistant").unwrap_or(text);
        let Some((header, tail)) = frame.split_once("<|message|>") else {
            if done {
                content.push_str(frame);
            }
            break;
        };
        let markers = [
            "<|end|>",
            "<|return|>",
            "<|call|>",
            "<|start|>",
            "<|endoftext|>",
        ];
        let end = markers
            .iter()
            .filter_map(|m| tail.find(m).map(|at| (at, m.len())))
            .min_by_key(|(at, _)| *at);
        let len = end.map_or_else(
            || {
                if done {
                    return tail.len();
                }
                let held = markers
                    .iter()
                    .flat_map(|m| {
                        (1..m.len().min(tail.len() + 1)).rev().find(|&n| {
                            tail.is_char_boundary(tail.len() - n)
                                && m.starts_with(&tail[tail.len() - n..])
                        })
                    })
                    .max()
                    .unwrap_or(0);
                tail.len() - held
            },
            |(at, _)| at,
        );
        if header.contains("to=") {
            content.push_str(header);
            content.push_str("<|message|>");
            content.push_str(&tail[..len]);
        } else if header.contains("<|channel|>analysis") {
            reasoning.push_str(&tail[..len]);
        } else {
            content.push_str(&tail[..len]);
        }
        text = end.map_or("", |(at, len)| &tail[at + len..]);
    }
    (reasoning.trim_start().into(), content.trim_start().into())
}

/// Restricted literal reader for Gemma's delimiters and LFM's Python-like call arguments.
/// Supports nested JSON data and quoted strings; no evaluation, identifiers as values, or code.
struct Literal {
    s: Vec<char>,
    at: usize,
}
impl Literal {
    fn new(s: &str) -> Self {
        Self {
            s: s.chars().collect(),
            at: 0,
        }
    }
    fn peek(&mut self) -> Option<char> {
        while self.s.get(self.at).is_some_and(|c| c.is_whitespace()) {
            self.at += 1;
        }
        self.s.get(self.at).copied()
    }
    fn take(&mut self, c: char) -> Option<()> {
        if self.peek()? != c {
            return None;
        }
        self.at += 1;
        Some(())
    }
    fn end(&mut self) -> Option<()> {
        if self.peek().is_none() {
            Some(())
        } else {
            None
        }
    }
    fn starts(&self, s: &str) -> bool {
        self.s[self.at..]
            .iter()
            .copied()
            .take(s.chars().count())
            .eq(s.chars())
    }
    fn ident(&mut self) -> Option<String> {
        self.peek()?;
        let start = self.at;
        while self
            .s
            .get(self.at)
            .is_some_and(|c| c.is_alphanumeric() || matches!(c, '_' | '-' | '.'))
        {
            self.at += 1;
        }
        (self.at > start).then(|| self.s[start..self.at].iter().collect())
    }
    fn string(&mut self) -> Option<String> {
        let c = self.peek()?;
        if self.starts("<|\"|>") {
            self.at += 5;
            let start = self.at;
            while self.at < self.s.len() && !self.starts("<|\"|>") {
                self.at += 1;
            }
            let value = self.s[start..self.at].iter().collect();
            if self.at == self.s.len() {
                return None;
            }
            self.at += 5;
            return Some(value);
        }
        if !matches!(c, '\'' | '"') {
            return None;
        }
        self.at += 1;
        let mut out = String::new();
        loop {
            let ch = *self.s.get(self.at)?;
            self.at += 1;
            if ch == c {
                return Some(out);
            }
            if ch == '\\' {
                let e = *self.s.get(self.at)?;
                self.at += 1;
                out.push(match e {
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    '\\' => '\\',
                    '\'' => '\'',
                    '"' => '"',
                    _ => return None,
                });
            } else {
                out.push(ch);
            }
        }
    }
    fn value(&mut self) -> Option<Value> {
        self.nested(0)
    }
    fn nested(&mut self, depth: usize) -> Option<Value> {
        if depth > 64 {
            return None;
        }
        match self.peek()? {
            '{' => {
                self.take('{')?;
                let mut obj = Map::new();
                if self.peek() != Some('}') {
                    loop {
                        let key = if matches!(self.peek()?, '\'' | '"' | '<') {
                            self.string()?
                        } else {
                            self.ident()?
                        };
                        self.take(':')?;
                        if obj.insert(key, self.nested(depth + 1)?).is_some() {
                            return None;
                        }
                        if self.peek() != Some(',') {
                            break;
                        }
                        self.take(',')?;
                    }
                }
                self.take('}')?;
                Some(Value::Object(obj))
            }
            '[' => {
                self.take('[')?;
                let mut a = vec![];
                if self.peek() != Some(']') {
                    loop {
                        a.push(self.nested(depth + 1)?);
                        if self.peek() != Some(',') {
                            break;
                        }
                        self.take(',')?;
                    }
                }
                self.take(']')?;
                Some(Value::Array(a))
            }
            '\'' | '"' | '<' => Some(Value::String(self.string()?)),
            _ => {
                self.peek()?;
                let start = self.at;
                while self
                    .s
                    .get(self.at)
                    .is_some_and(|c| !c.is_whitespace() && !matches!(c, ',' | '}' | ']' | ')'))
                {
                    self.at += 1;
                }
                let s: String = self.s[start..self.at].iter().collect();
                match s.as_str() {
                    "True" => Some(json!(true)),
                    "False" => Some(json!(false)),
                    "None" => Some(Value::Null),
                    _ => serde_json::from_str(&s).ok(),
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct Tools {
    pub tools: Value,
    pub choice: Value,
    pub enabled: bool,
    pub named: Option<String>,
}
impl Tools {
    pub fn from_request(req: &Value) -> Result<Self, String> {
        let tools = req.get("tools").cloned().unwrap_or(json!([]));
        let list = tools.as_array().ok_or("tools must be an array")?;
        let mut names = std::collections::HashSet::new();
        for t in list {
            let name = t
                .pointer("/function/name")
                .and_then(Value::as_str)
                .ok_or("tools require function.name")?;
            if t["type"] != "function" || name.is_empty() || !names.insert(name) {
                return Err("tools must have unique named functions of type function".into());
            }
        }
        let choice = req
            .get("tool_choice")
            .cloned()
            .unwrap_or(json!(if list.is_empty() { "none" } else { "auto" }));
        let named = if choice.is_object() && choice["type"] == "function" {
            Some(
                choice
                    .pointer("/function/name")
                    .and_then(Value::as_str)
                    .ok_or("tool_choice.function.name is required")?
                    .to_owned(),
            )
        } else {
            None
        };
        if !matches!(choice.as_str(), Some("none" | "auto" | "required")) && named.is_none() {
            return Err("tool_choice must be auto, none, required, or a named function".into());
        }
        if let Some(n) = &named {
            if !names.contains(n.as_str()) {
                return Err(format!("tool_choice names unknown function {n}"));
            }
        }
        if choice == "required" && list.is_empty() {
            return Err("tool_choice required needs tools".into());
        }
        let enabled = choice != "none" && !list.is_empty();
        let tools = if let Some(n) = &named {
            json!(list
                .iter()
                .filter(|t| t["function"]["name"] == *n)
                .collect::<Vec<_>>())
        } else if enabled {
            tools
        } else {
            Value::Null
        };
        Ok(Self {
            tools,
            choice,
            enabled,
            named,
        })
    }
    /// XML-like protocols render string parameters without quotes. Recover their type from
    /// the advertised schema (e.g. the string "123" must not become the number 123).
    pub fn restore_argument_types(&self, calls: &mut [Value]) {
        fn restore(value: &mut Value, schema: &Value) {
            if schema.get("type") == Some(&json!("string")) && !value.is_string() {
                *value = Value::String(value.to_string());
            } else if let Some(object) = value.as_object_mut() {
                for (key, value) in object {
                    if let Some(child) = schema.get("properties").and_then(|p| p.get(key)) {
                        restore(value, child);
                    }
                }
            } else if let Some(array) = value.as_array_mut() {
                if let Some(items) = schema.get("items") {
                    for value in array {
                        restore(value, items);
                    }
                }
            }
        }
        for call in calls {
            if let Some(tool) = self.tools.as_array().and_then(|ts| {
                ts.iter()
                    .find(|t| t["function"]["name"] == call["function"]["name"])
            }) {
                if let Some(text) = call["function"]["arguments"].as_str() {
                    if let Ok(mut args) = serde_json::from_str::<Value>(text) {
                        restore(&mut args, &tool["function"]["parameters"]);
                        call["function"]["arguments"] = json!(args.to_string());
                    }
                }
            }
        }
    }

    pub fn accepts(&self, calls: &[Value]) -> bool {
        calls.iter().all(|c| {
            self.tools.as_array().is_some_and(|ts| {
                ts.iter()
                    .any(|t| t["function"]["name"] == c["function"]["name"])
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn six_protocols_and_malformed_fallback() {
        let cases=[
            (Format::Function,"<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"),
            (Format::Tagged,"<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}</tool_call>"),
            (Format::Tagged,"<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>"),
            (Format::Tagged,"<tool_call>get_weather<param_key>city</param_key><param_value>Paris</param_value></tool_call>"),
            (Format::Gemma,"<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>"),
            (Format::Hy,"<tool_calls><tool_call>get_weather<tool_sep><arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call></tool_calls>"),
            (Format::Hy,"<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{\"city\":\"Paris\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"),
            (Format::Lfm,"<|tool_call_start|>[get_weather(city='Paris')]<|tool_call_end|>"),
            (Format::Harmony," to=functions.get_weather<|channel|>commentary json<|message|>{\"city\":\"Paris\"}<|call|>"),
        ];
        for (f, t) in cases {
            let p = f.parse(t);
            assert_eq!(p.calls.len(), 1, "{f:?}: {p:?}");
            assert_eq!(p.calls[0]["function"]["name"], "get_weather");
            assert_eq!(
                serde_json::from_str::<Value>(
                    p.calls[0]["function"]["arguments"].as_str().unwrap()
                )
                .unwrap(),
                json!({"city":"Paris"})
            );
            for end in t.char_indices().map(|(i, _)| i) {
                let _ = f.parse(&t[..end]);
            }
        }
        for f in [
            Format::Function,
            Format::Tagged,
            Format::Gemma,
            Format::Hy,
            Format::Lfm,
            Format::Harmony,
        ] {
            let t = "hello <tool_call>broken";
            let p = f.parse(t);
            assert_eq!(p.content, t);
            assert!(p.calls.is_empty());
        }
    }
    #[test]
    fn unquoted_xml_arguments_keep_declared_string_types() {
        let tools=Tools::from_request(&json!({"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{"s":{"type":"string"},"n":{"type":"integer"}}}}}]})).unwrap();
        let mut parsed=Format::Function.parse("<tool_call><function=f><parameter=s>123</parameter><parameter=n>4</parameter></function></tool_call>");
        tools.restore_argument_types(&mut parsed.calls);
        assert_eq!(
            serde_json::from_str::<Value>(
                parsed.calls[0]["function"]["arguments"].as_str().unwrap()
            )
            .unwrap(),
            json!({"s":"123","n":4})
        );
    }

    #[test]
    fn choices_validate_and_filter() {
        let t = json!([{"type":"function","function":{"name":"weather"}}]);
        for c in [
            json!("auto"),
            json!("required"),
            json!({"type":"function","function":{"name":"weather"}}),
        ] {
            assert!(
                Tools::from_request(&json!({"tools":t,"tool_choice":c}))
                    .unwrap()
                    .enabled
            );
        }
        assert!(
            !Tools::from_request(&json!({"tools":t,"tool_choice":"none"}))
                .unwrap()
                .enabled
        );
        for r in [
            json!({"tools":{}}),
            json!({"tool_choice":"required"}),
            json!({"tool_choice":"bad"}),
            json!({"tools":t,"tool_choice":{"type":"function","function":{"name":"missing"}}}),
        ] {
            assert!(Tools::from_request(&r).is_err());
        }
    }
}
