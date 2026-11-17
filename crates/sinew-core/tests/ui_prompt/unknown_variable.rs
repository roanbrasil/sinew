fn main() {
    let _ = sinew_core::prompt!("Summarize {text}").txt("doc").format();
}
