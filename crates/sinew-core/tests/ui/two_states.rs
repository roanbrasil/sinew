/// Echo.
#[sinew_core::tool]
fn echo(#[state] a: u8, #[state] b: u8, x: String) -> String { let _ = (a, b); x }
fn main() {}
