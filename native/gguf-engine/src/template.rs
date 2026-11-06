//! Chat prompt rendering from the GGUF `tokenizer.chat_template` Jinja source.

use anyhow::{Context, Result};
use minijinja::{Environment, Error, ErrorKind, Value};
use serde_json::Value as Json;

/// Compiled chat template plus the special-token texts it references.
pub struct ChatTemplate {
    env: Environment<'static>,
    has_template: bool,
    bos: String,
    eos: String,
}

/// Rendered prompt plus generation-mode facts derived from its tail.
pub struct Rendered {
    pub text: String,
    /// The prompt ends inside an opened `<think>` block, so output starts as reasoning.
    pub opens_reasoning: bool,
}

/// ChatML, used when the model ships no template or its template fails to render.
const CHATML: &str = r##"
{%- if tools %}
{{- '<|im_start|>system\n' }}
{%- if messages[0].role == 'system' %}{{- messages[0].content + '\n\n' }}{%- endif %}
{{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
{%- for tool in tools %}{{- '\n' }}{{- tool | tojson }}{%- endfor %}
{{- '\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n' }}
{%- elif messages[0].role == 'system' %}
{{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
{%- endif %}
{%- for m in messages %}
{%- if m.role == 'system' and loop.first %}
{%- elif m.role == 'tool' %}
{{- '<|im_start|>user\n<tool_response>\n' + m.content + '\n</tool_response><|im_end|>\n' }}
{%- else %}
{{- '<|im_start|>' + m.role + '\n' + (m.content or '') }}
{%- for c in (m.tool_calls or []) %}
{{- '\n<tool_call>\n{"name": "' + c.function.name + '", "arguments": ' + (c.function.arguments | tojson) + '}\n</tool_call>' }}
{%- endfor %}
{{- '<|im_end|>\n' }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}
"##;

fn raise_exception(msg: String) -> std::result::Result<Value, Error> {
    Err(Error::new(ErrorKind::InvalidOperation, msg))
}

fn strftime_now(fmt: String) -> String {
    // Templates use this only for a "current date" line; render a fixed-format UTC date.
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let days = (secs / 86_400) as i64;
    let (y, m, d) = civil_from_days(days);
    const MONTHS: [&str; 12] = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
    fmt.replace("%Y", &y.to_string())
        .replace("%m", &format!("{m:02}"))
        .replace("%d", &format!("{d:02}"))
        .replace("%B", MONTHS[(m - 1) as usize])
        .replace("%b", &MONTHS[(m - 1) as usize][..3])
}

/// Days since 1970-01-01 to a proleptic Gregorian date.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

impl ChatTemplate {
    /// Compile the model's template, falling back to ChatML.
    /// @param source - `tokenizer.chat_template`, when present.
    /// @param bos - BOS token text.
    /// @param eos - EOS token text.
    pub fn new(source: Option<String>, bos: String, eos: String) -> Result<Self> {
        let mut env = Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", raise_exception);
        env.add_function("strftime_now", strftime_now);
        env.add_template_owned("chatml", CHATML).context("built-in ChatML template")?;
        let has_template = match source {
            Some(src) => env.add_template_owned("model", src).is_ok(),
            None => false,
        };
        Ok(Self { env, has_template, bos, eos })
    }

    /// Render a conversation for generation.
    /// @param messages - OpenAI-style message objects.
    /// @param tools - OpenAI-style tool declarations.
    /// @param thinking - reasoning switch passed as `enable_thinking`.
    pub fn render(&self, messages: &Json, tools: &Json, thinking: Option<bool>) -> Result<Rendered> {
        let has_tools = tools.as_array().is_some_and(|t| !t.is_empty());
        let ctx = minijinja::context! {
            messages => Value::from_serialize(messages),
            tools => if has_tools { Value::from_serialize(tools) } else { Value::from(()) },
            add_generation_prompt => true,
            bos_token => self.bos.clone(),
            eos_token => self.eos.clone(),
            enable_thinking => thinking.unwrap_or(true),
        };
        let text = if self.has_template {
            match self.env.get_template("model")?.render(&ctx) {
                Ok(t) => t,
                Err(_) => self.env.get_template("chatml")?.render(&ctx)?,
            }
        } else {
            self.env.get_template("chatml")?.render(&ctx)?
        };
        let tail = text.trim_end();
        let opens_reasoning = match (tail.rfind("<think>"), tail.rfind("</think>")) {
            (Some(o), Some(c)) => o > c,
            (Some(_), None) => true,
            _ => false,
        };
        Ok(Rendered { text, opens_reasoning })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chatml_fallback_renders_tools_and_generation_prompt() {
        let t = ChatTemplate::new(None, "<s>".into(), "</s>".into()).unwrap();
        let msgs = serde_json::json!([{"role":"system","content":"S"},{"role":"user","content":"hi"}]);
        let tools = serde_json::json!([{"type":"function","function":{"name":"f","parameters":{}}}]);
        let r = t.render(&msgs, &tools, None).unwrap();
        assert!(r.text.contains("<tools>"));
        assert!(r.text.ends_with("<|im_start|>assistant\n"));
        assert!(!r.opens_reasoning);
    }

    #[test]
    fn detects_open_reasoning_tail() {
        let src = "{% for m in messages %}{{ m.content }}{% endfor %}<think>\n".to_string();
        let t = ChatTemplate::new(Some(src), String::new(), String::new()).unwrap();
        let r = t.render(&serde_json::json!([{"role":"user","content":"x"}]), &Json::Null, None).unwrap();
        assert!(r.opens_reasoning);
    }

    #[test]
    fn civil_dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(20_000), (2024, 10, 4));
    }
}
