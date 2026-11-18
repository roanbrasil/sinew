#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| sinew_fuzz::otlp::proto(data));
