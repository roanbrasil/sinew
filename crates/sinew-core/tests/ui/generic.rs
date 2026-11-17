/// Echo.
#[sinew_core::tool]
fn echo<T: serde::Serialize>(x: T) -> T { x }
fn main() {}
