#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| sinew_fuzz::streams::openai(data));
