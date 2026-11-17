fn main() {
    let _ = sinew_core::prompt!("Summarize {text} in {lang}").text("doc").format();
}
