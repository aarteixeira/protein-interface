fn main() {
    if let Err(error) = _core::hbplus::cli::run(&std::env::args().skip(1).collect::<Vec<_>>()) {
        eprintln!("protein-interface-hbplus: {error}");
        std::process::exit(2);
    }
}
