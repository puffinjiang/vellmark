fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("vellmark-cli v{}", env!("CARGO_PKG_VERSION"));
        println!("Usage:");
        println!("  vellmark <file.md>                  Open file in Vellmark");
        println!("  vellmark export <file.md> -o <out>  Headless export (HTML / PDF)");
        return;
    }
    println!("vellmark-cli v{} ready.", env!("CARGO_PKG_VERSION"));
}
