#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| sinew_fuzz::json_rpc::mcp(data));
