//! Conditions on tool-call arguments: `amount <= 100`,
//! `currency in ["USD", "EUR"]`, `to matches "@example\\.com$"`.

use std::{fmt, str::FromStr};

use regex::Regex;
use serde_json::Value;

use crate::{decision::to_pointer, GovernError};

/// A comparison applied to one argument.
#[derive(Debug, Clone)]
pub enum Op {
    /// Equal to this JSON value (numbers compare by value: `1` equals `1.0`).
    Eq(Value),
    /// Not equal (a missing field is not equal).
    Ne(Value),
    /// Equal to one of these values.
    In(Vec<Value>),
    /// Present and equal to none of these values.
    NotIn(Vec<Value>),
    /// A string matching this regular expression (unanchored).
    Matches(Regex),
    /// A number lower than.
    Lt(f64),
    /// A number lower than or equal to.
    Le(f64),
    /// A number greater than.
    Gt(f64),
    /// A number greater than or equal to.
    Ge(f64),
    /// The field is present (any value, `null` included).
    Exists,
    /// The field is absent.
    Missing,
}

/// One condition on the request's arguments: a field (JSON pointer such as
/// `/payee/country`, or a dotted path such as `payee.country`) and an
/// [`Op`]. A condition on a field that is absent is false, except
/// [`Op::Missing`] (and [`Op::Ne`], which is false too: write
/// `exists` alongside it if absence should not match).
///
/// Parse the text form with [`Condition::parse`] (also used in policy
/// files):
///
/// | Text | Meaning |
/// |---|---|
/// | `amount <= 100` | number comparisons: `<`, `<=`, `>`, `>=` |
/// | `currency == "USD"` / `currency != "USD"` | JSON equality |
/// | `currency in ["USD", "EUR"]` / `not in` | membership |
/// | `to matches "@example\\.com$"` | regular expression on a string |
/// | `customer.ssn exists` / `customer.ssn missing` | presence |
///
/// Values are JSON; a bare word that is not JSON is read as a string
/// (`status == open`).
///
/// ```
/// use serde_json::json;
/// use sinew_govern::Condition;
///
/// let c = Condition::parse("amount <= 100").unwrap();
/// assert!(c.matches(&json!({"amount": 40})));
/// assert!(!c.matches(&json!({"amount": 400})));
/// assert!(!c.matches(&json!({})));
/// assert!(Condition::parse("amount <== 1").is_err());
/// ```
#[derive(Debug, Clone)]
pub struct Condition {
    field: String,
    pointer: String,
    op: Op,
    source: String,
}

impl Condition {
    /// A condition on `field` (pointer or dotted path).
    pub fn new(field: impl Into<String>, op: Op) -> Self {
        let field = field.into();
        let pointer = to_pointer(&field);
        let source = format!("{field} {}", op_text(&op));
        Self {
            field,
            pointer,
            op,
            source,
        }
    }

    /// Parses the text form (see the type docs).
    pub fn parse(text: &str) -> Result<Self, GovernError> {
        let err = |msg: String| GovernError::Condition {
            condition: text.to_string(),
            message: msg,
        };
        let text_t = text.trim();
        let (field, rest) = text_t
            .split_once(char::is_whitespace)
            .ok_or_else(|| err("expected `<field> <operator> <value>`".into()))?;
        let rest = rest.trim_start();
        let field = field.trim();
        if field.is_empty() {
            return Err(err("missing field".into()));
        }
        let (op_word, value) = split_op(rest);
        let need_value = |v: &str| -> Result<Value, GovernError> {
            if v.is_empty() {
                return Err(err(format!("operator `{op_word}` needs a value")));
            }
            Ok(parse_value(v))
        };
        let number = |v: &str| -> Result<f64, GovernError> {
            need_value(v)?
                .as_f64()
                .ok_or_else(|| err(format!("operator `{op_word}` needs a number, got `{v}`")))
        };
        let list = |v: &str| -> Result<Vec<Value>, GovernError> {
            match need_value(v)? {
                Value::Array(items) => Ok(items),
                _ => Err(err(format!(
                    "operator `{op_word}` needs a JSON list such as [\"a\", \"b\"], got `{v}`"
                ))),
            }
        };
        let op = match op_word {
            "==" | "=" => Op::Eq(need_value(value)?),
            "!=" => Op::Ne(need_value(value)?),
            "<" => Op::Lt(number(value)?),
            "<=" => Op::Le(number(value)?),
            ">" => Op::Gt(number(value)?),
            ">=" => Op::Ge(number(value)?),
            "in" => Op::In(list(value)?),
            "not in" => Op::NotIn(list(value)?),
            "matches" | "~" => {
                let pattern = match need_value(value)? {
                    Value::String(s) => s,
                    other => other.to_string(),
                };
                Op::Matches(
                    Regex::new(&pattern)
                        .map_err(|e| err(format!("invalid regular expression: {e}")))?,
                )
            }
            "exists" | "missing" if !value.is_empty() => {
                return Err(err(format!("operator `{op_word}` takes no value")))
            }
            "exists" => Op::Exists,
            "missing" => Op::Missing,
            other => {
                return Err(err(format!(
                    "unknown operator `{other}` (expected ==, !=, <, <=, >, >=, in, not in, matches, exists or missing)"
                )))
            }
        };
        Ok(Self {
            field: field.to_string(),
            pointer: to_pointer(field),
            op,
            source: text_t.to_string(),
        })
    }

    /// The field as written.
    pub fn field(&self) -> &str {
        &self.field
    }

    pub fn op(&self) -> &Op {
        &self.op
    }

    /// Whether `arguments` satisfy the condition.
    pub fn matches(&self, arguments: &Value) -> bool {
        let v = arguments.pointer(&self.pointer);
        match (&self.op, v) {
            (Op::Missing, v) => v.is_none(),
            (_, None) => false,
            (Op::Exists, Some(_)) => true,
            (Op::Eq(want), Some(v)) => json_eq(v, want),
            (Op::Ne(want), Some(v)) => !json_eq(v, want),
            (Op::In(set), Some(v)) => set.iter().any(|w| json_eq(v, w)),
            (Op::NotIn(set), Some(v)) => !set.iter().any(|w| json_eq(v, w)),
            (Op::Matches(re), Some(v)) => v.as_str().is_some_and(|s| re.is_match(s)),
            (Op::Lt(n), Some(v)) => num(v).is_some_and(|x| x < *n),
            (Op::Le(n), Some(v)) => num(v).is_some_and(|x| x <= *n),
            (Op::Gt(n), Some(v)) => num(v).is_some_and(|x| x > *n),
            (Op::Ge(n), Some(v)) => num(v).is_some_and(|x| x >= *n),
        }
    }
}

impl fmt::Display for Condition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.source)
    }
}

impl FromStr for Condition {
    type Err = GovernError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

fn split_op(rest: &str) -> (&str, &str) {
    for word in ["not in", "matches", "exists", "missing", "in"] {
        if let Some(after) = rest.strip_prefix(word) {
            if after.is_empty() || after.starts_with(char::is_whitespace) {
                return (word, after.trim());
            }
        }
    }
    for sym in ["==", "!=", "<=", ">=", "<", ">", "=", "~"] {
        if let Some(after) = rest.strip_prefix(sym) {
            // `<==` and the like are typos, not `<=` followed by `=1`.
            if after.starts_with(['=', '<', '>', '!']) {
                break;
            }
            return (sym, after.trim());
        }
    }
    let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
    (&rest[..end], rest[end..].trim())
}

fn parse_value(text: &str) -> Value {
    serde_json::from_str(text).unwrap_or_else(|_| Value::String(text.to_string()))
}

/// A number, or a string holding one (tools often receive amounts as
/// strings).
fn num(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

fn json_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64(),
        _ => a == b,
    }
}

fn op_text(op: &Op) -> String {
    match op {
        Op::Eq(v) => format!("== {v}"),
        Op::Ne(v) => format!("!= {v}"),
        Op::In(v) => format!("in {}", Value::Array(v.clone())),
        Op::NotIn(v) => format!("not in {}", Value::Array(v.clone())),
        Op::Matches(re) => format!("matches {}", Value::String(re.as_str().into())),
        Op::Lt(n) => format!("< {n}"),
        Op::Le(n) => format!("<= {n}"),
        Op::Gt(n) => format!("> {n}"),
        Op::Ge(n) => format!(">= {n}"),
        Op::Exists => "exists".into(),
        Op::Missing => "missing".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn check(c: &str, args: Value) -> bool {
        Condition::parse(c).unwrap().matches(&args)
    }

    #[test]
    fn operators() {
        let a = json!({"amount": 100, "currency": "USD", "to": "ana@example.com",
                       "payee": {"country": "CA"}, "qty": "7", "flag": null});
        assert!(check("amount <= 100", a.clone()));
        assert!(!check("amount < 100", a.clone()));
        assert!(check("amount >= 100.0", a.clone()));
        assert!(!check("amount > 100", a.clone()));
        assert!(check("amount == 100.0", a.clone()));
        assert!(check("currency == \"USD\"", a.clone()));
        assert!(check("currency == USD", a.clone()));
        assert!(check("currency != EUR", a.clone()));
        assert!(check("currency in [\"USD\", \"EUR\"]", a.clone()));
        assert!(!check("currency not in [\"USD\"]", a.clone()));
        assert!(check("to matches \"@example\\\\.com$\"", a.clone()));
        assert!(check("payee.country == CA", a.clone()));
        assert!(check("/payee/country in [\"CA\", \"US\"]", a.clone()));
        assert!(check("qty > 5", a.clone()));
        assert!(check("flag exists", a.clone()));
        assert!(check("ssn missing", a.clone()));
        assert!(!check("ssn exists", a.clone()));
        // Absent fields never satisfy comparisons.
        assert!(!check("missing_field < 5", a.clone()));
        assert!(!check("missing_field != 5", a.clone()));
        assert!(!check("currency < 5", a));
    }

    #[test]
    fn parse_errors_are_clear() {
        let msg = |c: &str| Condition::parse(c).unwrap_err().to_string();
        assert!(msg("amount").contains("expected `<field> <operator> <value>`"));
        assert!(msg("amount <== 1").contains("unknown operator"));
        assert!(msg("amount <= lots").contains("needs a number"));
        assert!(msg("currency in USD").contains("needs a JSON list"));
        assert!(msg("to matches \"(\"").contains("invalid regular expression"));
        assert!(msg("ssn exists now").contains("takes no value"));
        assert!(msg("amount ==").contains("needs a value"));
        assert_eq!(
            Condition::parse(" amount <= 1 ").unwrap().to_string(),
            "amount <= 1"
        );
        assert_eq!(Condition::new("a.b", Op::Le(3.0)).to_string(), "a.b <= 3");
    }
}
