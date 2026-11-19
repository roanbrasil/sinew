//! `sinew-govern verify-audit`, `export-audit` and `keygen`; see
//! `sinew-govern --help`.

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map(|rt| {
            rt.block_on(sinew_govern::audit::cli::run(
                &args,
                &mut std::io::stdout(),
                &mut std::io::stderr(),
            ))
        })
        .unwrap_or(2);
    std::process::exit(code);
}
