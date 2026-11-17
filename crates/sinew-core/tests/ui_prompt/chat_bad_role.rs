fn main() {
    let _ = sinew_core::chat_prompt!(system: "Be brief.", tool: "{x}").format();
}
