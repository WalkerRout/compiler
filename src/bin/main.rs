use std::env;

use compiler::repl;

fn main() {
  let args: Vec<_> = env::args().skip(1).collect();

  loop {
    match repl::run(&args) {
      Ok(()) => break,
      Err(_) => continue,
    }
  }
}
