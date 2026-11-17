struct S;
impl S {
    /// Echo.
    #[sinew_core::tool]
    fn echo(&self, x: String) -> String { x }
}
fn main() {}
