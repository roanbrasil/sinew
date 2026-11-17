//! `calculator`: evaluates arithmetic expressions with a small parser.
//!
//! No code is executed: the input is tokenized and evaluated by a
//! recursive-descent parser over `f64` that knows numbers, `+ - * / %`, `^`
//! (or `**`) for powers, `!` for factorials, parentheses, the constants `pi`,
//! `e` and `tau`, and a fixed list of functions. Anything else is rejected
//! with an error that points at the offending position.
//!
//! ```
//! use sinew_tools::calculator::evaluate;
//!
//! assert_eq!(evaluate("2 + 3 * 4").unwrap(), 14.0);
//! assert_eq!(evaluate("-2^2").unwrap(), -4.0);
//! assert_eq!(evaluate("sqrt(16) + log(1000) + 5!").unwrap(), 4.0 + 3.0 + 120.0);
//! assert!(evaluate("std::process::exit(1)").is_err());
//! ```

use serde::Serialize;

use crate::ToolError;

/// Longest accepted expression, in characters.
pub const MAX_EXPRESSION_LEN: usize = 1_000;
/// Deepest accepted nesting of parentheses, unary operators and calls.
pub const MAX_DEPTH: usize = 64;

/// Result of the [`calculator`](fn@calculator) tool.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Calculation {
    pub expression: String,
    pub result: f64,
}

/// Evaluate an arithmetic expression and return the number. Use it for any
/// arithmetic instead of computing in your head. Supports + - * / % (remainder),
/// ^ or ** (power), ! (factorial), parentheses, scientific notation (1.5e3),
/// constants pi, e, tau, and functions sqrt, cbrt, abs, exp, ln, log (base 10,
/// or log(x, base)), log2, log10, sin, cos, tan, asin, acos, atan, atan2(y, x),
/// sinh, cosh, tanh, floor, ceil, round, trunc, min, max, pow(x, y), hypot(x, y).
/// Trigonometric functions use radians. Example: "(17.5 * 3 - 2) / 4".
#[sinew_core::tool]
pub fn calculator(
    /// The expression, e.g. "2 * (3 + 4) ^ 2" or "sqrt(2) * pi".
    expression: String,
) -> Result<Calculation, ToolError> {
    let result = evaluate(&expression)?;
    Ok(Calculation { expression, result })
}

/// Evaluates `expr`; see the [module docs](self) for the grammar.
pub fn evaluate(expr: &str) -> Result<f64, ToolError> {
    if expr.chars().count() > MAX_EXPRESSION_LEN {
        return Err(ToolError::Invalid(format!(
            "expression longer than {MAX_EXPRESSION_LEN} characters"
        )));
    }
    let tokens = tokenize(expr)?;
    if tokens.is_empty() {
        return Err(ToolError::Invalid("empty expression".into()));
    }
    let mut p = Parser {
        tokens,
        pos: 0,
        depth: 0,
    };
    let v = p.expr()?;
    if let Some(t) = p.peek() {
        return Err(p.unexpected(t.clone()));
    }
    if !v.is_finite() {
        return Err(ToolError::Invalid(
            "the result is not a finite number (overflow or undefined operation)".into(),
        ));
    }
    // Avoid printing `-0`.
    Ok(if v == 0.0 { 0.0 } else { v })
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    Ident(String),
    Op(char),
    LParen,
    RParen,
    Comma,
}

#[derive(Debug, Clone)]
struct Token {
    tok: Tok,
    at: usize,
}

fn tokenize(s: &str) -> Result<Vec<Token>, ToolError> {
    let chars: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let at = i + 1;
        match c {
            c if c.is_whitespace() => i += 1,
            '0'..='9' | '.' => {
                let start = i;
                while i < chars.len()
                    && (chars[i].is_ascii_digit() || chars[i] == '.' || chars[i] == '_')
                {
                    i += 1;
                }
                if i < chars.len() && (chars[i] == 'e' || chars[i] == 'E') {
                    let mut j = i + 1;
                    if j < chars.len() && (chars[j] == '+' || chars[j] == '-') {
                        j += 1;
                    }
                    if j < chars.len() && chars[j].is_ascii_digit() {
                        i = j;
                        while i < chars.len() && chars[i].is_ascii_digit() {
                            i += 1;
                        }
                    }
                }
                let text: String = chars[start..i].iter().filter(|c| **c != '_').collect();
                let n: f64 = text.parse().map_err(|_| {
                    ToolError::Invalid(format!("bad number `{text}` at position {at}"))
                })?;
                out.push(Token {
                    tok: Tok::Num(n),
                    at,
                });
            }
            c if c.is_ascii_alphabetic() => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let name: String = chars[start..i]
                    .iter()
                    .collect::<String>()
                    .to_ascii_lowercase();
                out.push(Token {
                    tok: Tok::Ident(name),
                    at,
                });
            }
            '*' if chars.get(i + 1) == Some(&'*') => {
                out.push(Token {
                    tok: Tok::Op('^'),
                    at,
                });
                i += 2;
            }
            '+' | '-' | '*' | '/' | '%' | '^' | '!' => {
                out.push(Token {
                    tok: Tok::Op(c),
                    at,
                });
                i += 1;
            }
            '×' => {
                out.push(Token {
                    tok: Tok::Op('*'),
                    at,
                });
                i += 1;
            }
            '÷' => {
                out.push(Token {
                    tok: Tok::Op('/'),
                    at,
                });
                i += 1;
            }
            '−' => {
                out.push(Token {
                    tok: Tok::Op('-'),
                    at,
                });
                i += 1;
            }
            '(' => {
                out.push(Token {
                    tok: Tok::LParen,
                    at,
                });
                i += 1;
            }
            ')' => {
                out.push(Token {
                    tok: Tok::RParen,
                    at,
                });
                i += 1;
            }
            ',' => {
                out.push(Token {
                    tok: Tok::Comma,
                    at,
                });
                i += 1;
            }
            other => {
                return Err(ToolError::Invalid(format!(
                    "unexpected character `{other}` at position {at}"
                )))
            }
        }
    }
    Ok(out)
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    depth: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn next_if(&mut self, tok: &Tok) -> bool {
        if self.peek().is_some_and(|t| &t.tok == tok) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn unexpected(&self, t: Token) -> ToolError {
        let what = match t.tok {
            Tok::Num(n) => format!("number {n}"),
            Tok::Ident(s) => format!("`{s}`"),
            Tok::Op(c) => format!("`{c}`"),
            Tok::LParen => "`(`".into(),
            Tok::RParen => "`)`".into(),
            Tok::Comma => "`,`".into(),
        };
        ToolError::Invalid(format!("unexpected {what} at position {}", t.at))
    }

    fn enter(&mut self) -> Result<(), ToolError> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(ToolError::Invalid(format!(
                "expression nested deeper than {MAX_DEPTH} levels"
            )));
        }
        Ok(())
    }

    // expr := term (('+' | '-') term)*
    fn expr(&mut self) -> Result<f64, ToolError> {
        let mut v = self.term()?;
        loop {
            if self.next_if(&Tok::Op('+')) {
                v += self.term()?;
            } else if self.next_if(&Tok::Op('-')) {
                v -= self.term()?;
            } else {
                return Ok(v);
            }
        }
    }

    // term := unary (('*' | '/' | '%') unary)*
    fn term(&mut self) -> Result<f64, ToolError> {
        let mut v = self.unary()?;
        loop {
            if self.next_if(&Tok::Op('*')) {
                v *= self.unary()?;
            } else if self.next_if(&Tok::Op('/')) {
                let d = self.unary()?;
                if d == 0.0 {
                    return Err(ToolError::Invalid("division by zero".into()));
                }
                v /= d;
            } else if self.next_if(&Tok::Op('%')) {
                let d = self.unary()?;
                if d == 0.0 {
                    return Err(ToolError::Invalid("remainder by zero".into()));
                }
                v %= d;
            } else {
                return Ok(v);
            }
        }
    }

    // unary := ('-' | '+') unary | power
    fn unary(&mut self) -> Result<f64, ToolError> {
        self.enter()?;
        let v = if self.next_if(&Tok::Op('-')) {
            -self.unary()?
        } else if self.next_if(&Tok::Op('+')) {
            self.unary()?
        } else {
            self.power()?
        };
        self.depth -= 1;
        Ok(v)
    }

    // power := postfix ('^' unary)?   (right-associative; -2^2 = -4)
    fn power(&mut self) -> Result<f64, ToolError> {
        let base = self.postfix()?;
        if self.next_if(&Tok::Op('^')) {
            let exp = self.unary()?;
            return Ok(base.powf(exp));
        }
        Ok(base)
    }

    // postfix := primary '!'*
    fn postfix(&mut self) -> Result<f64, ToolError> {
        let mut v = self.primary()?;
        while self.next_if(&Tok::Op('!')) {
            v = factorial(v)?;
        }
        Ok(v)
    }

    fn primary(&mut self) -> Result<f64, ToolError> {
        let Some(t) = self.peek().cloned() else {
            return Err(ToolError::Invalid("expression ends too early".into()));
        };
        self.pos += 1;
        match t.tok {
            Tok::Num(n) => Ok(n),
            Tok::LParen => {
                self.enter()?;
                let v = self.expr()?;
                if !self.next_if(&Tok::RParen) {
                    return Err(match self.peek().cloned() {
                        Some(t) => self.unexpected(t),
                        None => ToolError::Invalid("missing `)`".into()),
                    });
                }
                self.depth -= 1;
                Ok(v)
            }
            Tok::Ident(name) => {
                if self.next_if(&Tok::LParen) {
                    self.enter()?;
                    let mut args = Vec::new();
                    if !self.next_if(&Tok::RParen) {
                        loop {
                            args.push(self.expr()?);
                            if self.next_if(&Tok::Comma) {
                                continue;
                            }
                            if self.next_if(&Tok::RParen) {
                                break;
                            }
                            return Err(match self.peek().cloned() {
                                Some(t) => self.unexpected(t),
                                None => ToolError::Invalid(format!("missing `)` after {name}(")),
                            });
                        }
                    }
                    self.depth -= 1;
                    call(&name, &args)
                } else {
                    match name.as_str() {
                        "pi" => Ok(std::f64::consts::PI),
                        "e" => Ok(std::f64::consts::E),
                        "tau" => Ok(std::f64::consts::TAU),
                        _ => Err(ToolError::Invalid(format!(
                            "unknown name `{name}` at position {}; constants are pi, e, tau",
                            t.at
                        ))),
                    }
                }
            }
            _ => Err(self.unexpected(t)),
        }
    }
}

fn factorial(n: f64) -> Result<f64, ToolError> {
    if n < 0.0 || n.fract() != 0.0 {
        return Err(ToolError::Invalid(
            "factorial needs a non-negative integer".into(),
        ));
    }
    if n > 170.0 {
        return Err(ToolError::Invalid("factorial overflows above 170!".into()));
    }
    Ok((1..=n as u64).fold(1.0, |acc, k| acc * k as f64))
}

fn call(name: &str, args: &[f64]) -> Result<f64, ToolError> {
    let arity = |n: usize| -> Result<(), ToolError> {
        if args.len() == n {
            Ok(())
        } else {
            Err(ToolError::Invalid(format!(
                "{name}() takes {n} argument{}, got {}",
                if n == 1 { "" } else { "s" },
                args.len()
            )))
        }
    };
    let one = |f: fn(f64) -> f64| -> Result<f64, ToolError> {
        arity(1)?;
        Ok(f(args[0]))
    };
    let two = |f: fn(f64, f64) -> f64| -> Result<f64, ToolError> {
        arity(2)?;
        Ok(f(args[0], args[1]))
    };
    match name {
        "sqrt" => {
            arity(1)?;
            if args[0] < 0.0 {
                return Err(ToolError::Invalid("sqrt() of a negative number".into()));
            }
            Ok(args[0].sqrt())
        }
        "cbrt" => one(f64::cbrt),
        "abs" => one(f64::abs),
        "exp" => one(f64::exp),
        "ln" => log_checked(name, args, f64::ln),
        "log10" => log_checked(name, args, f64::log10),
        "log2" => log_checked(name, args, f64::log2),
        "log" => match args.len() {
            1 => log_checked(name, args, f64::log10),
            2 => {
                if args[0] <= 0.0 || args[1] <= 0.0 || args[1] == 1.0 {
                    return Err(ToolError::Invalid(
                        "log(x, base) needs x > 0 and a positive base other than 1".into(),
                    ));
                }
                Ok(args[0].log(args[1]))
            }
            n => Err(ToolError::Invalid(format!(
                "log() takes 1 or 2 arguments, got {n}"
            ))),
        },
        "sin" => one(f64::sin),
        "cos" => one(f64::cos),
        "tan" => one(f64::tan),
        "asin" => one(f64::asin),
        "acos" => one(f64::acos),
        "atan" => one(f64::atan),
        "atan2" => two(f64::atan2),
        "sinh" => one(f64::sinh),
        "cosh" => one(f64::cosh),
        "tanh" => one(f64::tanh),
        "floor" => one(f64::floor),
        "ceil" => one(f64::ceil),
        "round" => one(f64::round),
        "trunc" => one(f64::trunc),
        "pow" => two(f64::powf),
        "hypot" => two(f64::hypot),
        "min" | "max" => {
            if args.is_empty() {
                return Err(ToolError::Invalid(format!(
                    "{name}() needs at least one argument"
                )));
            }
            let pick = if name == "min" { f64::min } else { f64::max };
            Ok(args[1..].iter().fold(args[0], |a, b| pick(a, *b)))
        }
        _ => Err(ToolError::Invalid(format!("unknown function `{name}`"))),
    }
}

fn log_checked(name: &str, args: &[f64], f: fn(f64) -> f64) -> Result<f64, ToolError> {
    if args.len() != 1 {
        return Err(ToolError::Invalid(format!(
            "{name}() takes 1 argument, got {}",
            args.len()
        )));
    }
    if args[0] <= 0.0 {
        return Err(ToolError::Invalid(format!(
            "{name}() needs a positive number"
        )));
    }
    Ok(f(args[0]))
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use sinew_core::Tool;

    use super::*;

    fn close(expr: &str, want: f64) {
        let got = evaluate(expr).unwrap_or_else(|e| panic!("{expr}: {e}"));
        assert!(
            (got - want).abs() < 1e-9 * want.abs().max(1.0),
            "{expr} = {got}, want {want}"
        );
    }

    #[test]
    fn evaluates_correctly() {
        close("1 + 2 * 3", 7.0);
        close("(1 + 2) * 3", 9.0);
        close("10 - 4 - 3", 3.0);
        close("100 / 10 / 5", 2.0);
        close("7 % 3", 1.0);
        close("-7 % 3", -1.0);
        close("2 ^ 3 ^ 2", 512.0);
        close("2 ** 10", 1024.0);
        close("-2 ^ 2", -4.0);
        close("(-2) ^ 2", 4.0);
        close("2 ^ -1", 0.5);
        close("--3", 3.0);
        close("+4", 4.0);
        close("3!", 6.0);
        close("0!", 1.0);
        close("3!!", 720.0);
        close("1.5e3 + 2E-1", 1500.2);
        close("1_000_000 / 4", 250_000.0);
        close(".5 + 0.25", 0.75);
        close("pi", std::f64::consts::PI);
        close("2 * PI", std::f64::consts::TAU);
        close("e ^ 1", std::f64::consts::E);
        close("sqrt(16)", 4.0);
        close("cbrt(27)", 3.0);
        close("abs(-3.5)", 3.5);
        close("ln(e)", 1.0);
        close("log(1000)", 3.0);
        close("log(8, 2)", 3.0);
        close("log2(1024)", 10.0);
        close("sin(pi / 2)", 1.0);
        close("cos(0)", 1.0);
        close("atan2(1, 1)", std::f64::consts::FRAC_PI_4);
        close("min(3, 1, 2) + max(3, 1, 2)", 4.0);
        close("pow(2, 0.5) * hypot(3, 4)", 2f64.sqrt() * 5.0);
        close(
            "round(2.5) + floor(-1.5) + ceil(1.2) + trunc(-1.7)",
            3.0 - 2.0 + 2.0 - 1.0,
        );
        close("12 × 3 ÷ 4 − 1", 8.0);
        close("((((1))))", 1.0);
        close("0.1 + 0.2", 0.30000000000000004);
        assert_eq!(evaluate("-0").unwrap().to_string(), "0");
    }

    #[test]
    fn rejects_garbage() {
        for bad in [
            "",
            "   ",
            "1 +",
            "* 2",
            "(1 + 2",
            "1 + 2)",
            "1 2",
            "2 (3)",
            "foo",
            "foo(1)",
            "x + 1",
            "1 / 0",
            "5 % 0",
            "sqrt(-1)",
            "ln(0)",
            "log(-1)",
            "log(8, 1)",
            "sqrt()",
            "sqrt(1, 2)",
            "min()",
            "(-1)!",
            "2.5!",
            "171!",
            "10 ^ 400",
            "1..2",
            "1e",
            "std::process::exit(1)",
            "__import__('os').system('ls')",
            "1; drop table",
            "\"1\" + 1",
            "{1}",
            "1 = 1",
            "a.b",
            "$HOME",
        ] {
            assert!(evaluate(bad).is_err(), "`{bad}` should be rejected");
        }
        let long = "1+".repeat(600) + "1";
        assert!(evaluate(&long).is_err());
        let deep = "(".repeat(200) + "1" + &")".repeat(200);
        assert!(matches!(evaluate(&deep), Err(ToolError::Invalid(m)) if m.contains("nested")));
        let neg = "-".repeat(500) + "1";
        assert!(evaluate(&neg).is_err());
    }

    #[test]
    fn error_messages_point_at_the_problem() {
        let e = evaluate("2 + $").unwrap_err().to_string();
        assert!(e.contains("`$` at position 5"), "{e}");
        let e = evaluate("2 + foo").unwrap_err().to_string();
        assert!(e.contains("unknown name `foo`"), "{e}");
    }

    #[tokio::test]
    async fn works_as_a_tool() {
        let t = calculator::tool();
        assert!(!t.side_effects());
        let spec = t.spec();
        assert_eq!(spec.name, "calculator");
        assert!(spec.description.contains("radians"));
        assert_eq!(spec.parameters["required"], json!(["expression"]));
        let out = t.call(json!({"expression": "6 * 7"})).await.unwrap();
        assert_eq!(out, json!({"expression": "6 * 7", "result": 42.0}));
        let err = t.call(json!({"expression": "6 *"})).await.unwrap_err();
        assert!(err.to_string().contains("calculator"), "{err}");
        assert!(t.call(json!({"expr": "1"})).await.is_err());
    }
}
