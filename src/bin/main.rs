use std::io;

use compiler::repl::repl;

fn main() {
  if let Err(e) = repl(io::stdin(), io::stdout()) {
    println!("REPL exited with {e:?}");
  }
}
