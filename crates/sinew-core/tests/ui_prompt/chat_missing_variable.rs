fn main() {
    let _ = sinew_core::chat_prompt!(system: "You are {persona}.", user: "{question}")
        .question("why?")
        .format();
}
