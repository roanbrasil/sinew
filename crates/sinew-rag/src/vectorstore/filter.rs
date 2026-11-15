use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{Error, Metadata, Result};

/// Metadata filter for vector search. Keys are top-level metadata fields.
///
/// ```
/// use sinew_rag::Filter;
/// use serde_json::json;
///
/// let f = Filter::eq("lang", "en").and(Filter::gte("year", 2024.0));
/// // The same, LangChain / Chroma style:
/// let g = Filter::from_json(&json!({"lang": "en", "year": {"$gte": 2024}})).unwrap();
/// assert_eq!(f, g);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Filter {
    /// Field equals the value (numbers compare numerically).
    Eq(String, Value),
    /// Field is missing or differs.
    Ne(String, Value),
    /// Field equals one of the values.
    In(String, Vec<Value>),
    /// Field is missing or equals none of the values.
    Nin(String, Vec<Value>),
    Gt(String, f64),
    Gte(String, f64),
    Lt(String, f64),
    Lte(String, f64),
    /// Field is present (and not null).
    Exists(String),
    And(Vec<Filter>),
    Or(Vec<Filter>),
    Not(Box<Filter>),
}

impl Filter {
    pub fn eq(key: impl Into<String>, value: impl Into<Value>) -> Self {
        Filter::Eq(key.into(), value.into())
    }
    pub fn ne(key: impl Into<String>, value: impl Into<Value>) -> Self {
        Filter::Ne(key.into(), value.into())
    }
    pub fn is_in<I, V>(key: impl Into<String>, values: I) -> Self
    where
        I: IntoIterator<Item = V>,
        V: Into<Value>,
    {
        Filter::In(key.into(), values.into_iter().map(Into::into).collect())
    }
    pub fn gt(key: impl Into<String>, v: f64) -> Self {
        Filter::Gt(key.into(), v)
    }
    pub fn gte(key: impl Into<String>, v: f64) -> Self {
        Filter::Gte(key.into(), v)
    }
    pub fn lt(key: impl Into<String>, v: f64) -> Self {
        Filter::Lt(key.into(), v)
    }
    pub fn lte(key: impl Into<String>, v: f64) -> Self {
        Filter::Lte(key.into(), v)
    }
    pub fn exists(key: impl Into<String>) -> Self {
        Filter::Exists(key.into())
    }

    /// Both filters (flattens nested `And`s).
    pub fn and(self, other: Filter) -> Self {
        match self {
            Filter::And(mut v) => {
                v.push(other);
                Filter::And(v)
            }
            f => Filter::And(vec![f, other]),
        }
    }

    pub fn or(self, other: Filter) -> Self {
        match self {
            Filter::Or(mut v) => {
                v.push(other);
                Filter::Or(v)
            }
            f => Filter::Or(vec![f, other]),
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn not(self) -> Self {
        Filter::Not(Box::new(self))
    }

    /// Parses a LangChain / Chroma / Mongo-style filter object:
    /// `{"field": value}` (equality), `{"field": {"$eq" | "$ne" | "$gt" |
    /// "$gte" | "$lt" | "$lte" | "$in" | "$nin" | "$exists": ...}}`,
    /// `{"$and": [...]}`, `{"$or": [...]}`, `{"$not": {...}}`. Several keys
    /// in one object are combined with AND.
    pub fn from_json(v: &Value) -> Result<Self> {
        let obj = v
            .as_object()
            .ok_or_else(|| Error::Invalid("filter must be a JSON object".into()))?;
        let mut parts = Vec::new();
        for (k, val) in obj {
            parts.push(match k.as_str() {
                "$and" | "$or" => {
                    let arr = val
                        .as_array()
                        .ok_or_else(|| Error::Invalid(format!("`{k}` needs an array")))?;
                    let subs = arr.iter().map(Filter::from_json).collect::<Result<_>>()?;
                    if k == "$and" {
                        Filter::And(subs)
                    } else {
                        Filter::Or(subs)
                    }
                }
                "$not" => Filter::Not(Box::new(Filter::from_json(val)?)),
                field => match val.as_object() {
                    Some(ops) if ops.keys().all(|o| o.starts_with('$')) => {
                        let mut conds = Vec::new();
                        for (op, arg) in ops {
                            conds.push(field_op(field, op, arg)?);
                        }
                        single_or_and(conds)
                    }
                    _ => Filter::Eq(field.to_string(), val.clone()),
                },
            });
        }
        Ok(single_or_and(parts))
    }

    /// Whether `metadata` passes the filter.
    pub fn matches(&self, metadata: &Metadata) -> bool {
        match self {
            Filter::Eq(k, v) => metadata.get(k).is_some_and(|m| value_eq(m, v)),
            Filter::Ne(k, v) => !metadata.get(k).is_some_and(|m| value_eq(m, v)),
            Filter::In(k, vs) => metadata
                .get(k)
                .is_some_and(|m| vs.iter().any(|v| value_eq(m, v))),
            Filter::Nin(k, vs) => !metadata
                .get(k)
                .is_some_and(|m| vs.iter().any(|v| value_eq(m, v))),
            Filter::Gt(k, x) => num(metadata, k).is_some_and(|m| m > *x),
            Filter::Gte(k, x) => num(metadata, k).is_some_and(|m| m >= *x),
            Filter::Lt(k, x) => num(metadata, k).is_some_and(|m| m < *x),
            Filter::Lte(k, x) => num(metadata, k).is_some_and(|m| m <= *x),
            Filter::Exists(k) => metadata.get(k).is_some_and(|v| !v.is_null()),
            Filter::And(fs) => fs.iter().all(|f| f.matches(metadata)),
            Filter::Or(fs) => fs.iter().any(|f| f.matches(metadata)),
            Filter::Not(f) => !f.matches(metadata),
        }
    }

    /// The filter in Qdrant's JSON filter language (payload fields under
    /// `prefix`, e.g. `"metadata."`).
    pub fn to_qdrant(&self, prefix: &str) -> Value {
        let key = |k: &str| format!("{prefix}{k}");
        let matcher = |k: &str, v: &Value| match v {
            Value::Number(n) if n.as_i64().is_none() && n.as_u64().is_none() => {
                let f = n.as_f64().unwrap_or_default();
                json!({"key": key(k), "range": {"gte": f, "lte": f}})
            }
            _ => json!({"key": key(k), "match": {"value": v}}),
        };
        let any = |k: &str, vs: &[Value]| -> Value {
            let all_simple = vs.iter().all(|v| v.is_string() || v.is_i64() || v.is_u64());
            if all_simple {
                json!({"key": key(k), "match": {"any": vs}})
            } else {
                json!({"should": vs.iter().map(|v| matcher(k, v)).collect::<Vec<_>>()})
            }
        };
        match self {
            Filter::Eq(k, v) => json!({"must": [matcher(k, v)]}),
            Filter::Ne(k, v) => json!({"must_not": [matcher(k, v)]}),
            Filter::In(k, vs) => json!({"must": [any(k, vs)]}),
            Filter::Nin(k, vs) => json!({"must_not": [any(k, vs)]}),
            Filter::Gt(k, x) => json!({"must": [{"key": key(k), "range": {"gt": x}}]}),
            Filter::Gte(k, x) => json!({"must": [{"key": key(k), "range": {"gte": x}}]}),
            Filter::Lt(k, x) => json!({"must": [{"key": key(k), "range": {"lt": x}}]}),
            Filter::Lte(k, x) => json!({"must": [{"key": key(k), "range": {"lte": x}}]}),
            Filter::Exists(k) => {
                json!({"must_not": [{"is_empty": {"key": key(k)}}, {"is_null": {"key": key(k)}}]})
            }
            Filter::And(fs) => {
                json!({"must": fs.iter().map(|f| f.to_qdrant(prefix)).collect::<Vec<_>>()})
            }
            Filter::Or(fs) => {
                json!({"should": fs.iter().map(|f| f.to_qdrant(prefix)).collect::<Vec<_>>()})
            }
            Filter::Not(f) => json!({"must_not": [f.to_qdrant(prefix)]}),
        }
    }
}

fn single_or_and(mut v: Vec<Filter>) -> Filter {
    if v.len() == 1 {
        v.pop().unwrap()
    } else {
        Filter::And(v)
    }
}

fn field_op(field: &str, op: &str, arg: &Value) -> Result<Filter> {
    let k = field.to_string();
    let number = || {
        arg.as_f64()
            .ok_or_else(|| Error::Invalid(format!("`{op}` on `{field}` needs a number")))
    };
    let list = || {
        arg.as_array()
            .cloned()
            .ok_or_else(|| Error::Invalid(format!("`{op}` on `{field}` needs an array")))
    };
    Ok(match op {
        "$eq" => Filter::Eq(k, arg.clone()),
        "$ne" => Filter::Ne(k, arg.clone()),
        "$gt" => Filter::Gt(k, number()?),
        "$gte" => Filter::Gte(k, number()?),
        "$lt" => Filter::Lt(k, number()?),
        "$lte" => Filter::Lte(k, number()?),
        "$in" => Filter::In(k, list()?),
        "$nin" => Filter::Nin(k, list()?),
        "$exists" => {
            if arg.as_bool().unwrap_or(true) {
                Filter::Exists(k)
            } else {
                Filter::Not(Box::new(Filter::Exists(k)))
            }
        }
        other => return Err(Error::Invalid(format!("unknown filter operator `{other}`"))),
    })
}

fn num(m: &Metadata, k: &str) -> Option<f64> {
    m.get(k).and_then(Value::as_f64)
}

pub(crate) fn value_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64(),
        _ => a == b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(v: Value) -> Metadata {
        v.as_object().unwrap().clone()
    }

    #[test]
    fn evaluates_filters() {
        let m = meta(json!({"lang": "en", "year": 2024, "draft": false, "n": null}));
        assert!(Filter::eq("lang", "en").matches(&m));
        assert!(Filter::eq("year", 2024.0).matches(&m));
        assert!(!Filter::eq("lang", "pt").matches(&m));
        assert!(Filter::ne("missing", 1).matches(&m));
        assert!(Filter::is_in("lang", ["pt", "en"]).matches(&m));
        assert!(Filter::gt("year", 2000.0).matches(&m));
        assert!(!Filter::lt("year", 2000.0).matches(&m));
        assert!(Filter::exists("draft").matches(&m));
        assert!(!Filter::exists("n").matches(&m));
        assert!(Filter::eq("lang", "pt")
            .or(Filter::eq("draft", false))
            .matches(&m));
        assert!(!Filter::eq("lang", "en").not().matches(&m));
    }

    #[test]
    fn parses_mongo_style() {
        let f = Filter::from_json(&json!({
            "$or": [{"lang": "pt"}, {"year": {"$gte": 2020, "$lt": 2030}}],
            "tag": {"$nin": ["x"]}
        }))
        .unwrap();
        let m = meta(json!({"lang": "en", "year": 2024, "tag": "y"}));
        assert!(f.matches(&m));
        assert!(Filter::from_json(&json!({"a": {"$bogus": 1}})).is_err());
        assert!(Filter::from_json(&json!([1])).is_err());
        // An object value without operators is plain equality.
        let f = Filter::from_json(&json!({"a": {"b": 1}})).unwrap();
        assert_eq!(f, Filter::eq("a", json!({"b": 1})));
    }

    #[test]
    fn translates_to_qdrant() {
        let q = Filter::eq("lang", "en")
            .and(Filter::gte("year", 2024.0))
            .to_qdrant("metadata.");
        assert_eq!(
            q,
            json!({"must": [
                {"must": [{"key": "metadata.lang", "match": {"value": "en"}}]},
                {"must": [{"key": "metadata.year", "range": {"gte": 2024.0}}]}
            ]})
        );
    }
}
