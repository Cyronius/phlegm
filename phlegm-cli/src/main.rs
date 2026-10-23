use phlegm_core::mock_backend::MockBackend;
use phlegm_core::server::{self, ServerConfig};

mod bench;

const USAGE: &str = "usage: phlegm <command>

commands:
  bench     benchmark protocol, machine metadata, and result validation
  serve     OpenAI-compatible server (mock backend until the engine lands)
  version   print the version";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("bench") => {
            if let Err(e) = bench::command(&args[1..]) {
                eprintln!("phlegm bench: {e}");
                std::process::exit(1);
            }
        }
        Some("serve") => {
            let cfg = ServerConfig::from_env();
            if let Err(e) = server::serve(Box::new(MockBackend::new()), cfg) {
                eprintln!("phlegm serve: {e}");
                std::process::exit(1);
            }
        }
        Some("version") | Some("--version") => println!("phlegm {}", env!("CARGO_PKG_VERSION")),
        _ => {
            eprintln!("{USAGE}");
            std::process::exit(2);
        }
    }
}
