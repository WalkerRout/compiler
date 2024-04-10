
use std::env;

use compiler::repl;

fn main() -> Result<(), anyhow::Error> {
  let args: Vec<_> = env::args()
    .skip(1)
    .collect();
    
  loop {
    match repl::run(&args) {
      Ok(_) => break,
      Err(_) => continue,
    }
  }

  Ok(())
}
