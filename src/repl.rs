
#[allow(unused_imports)]
use anyhow::anyhow;

pub enum Target {
  File,
  CommandLine,
}

pub fn interpret(target: Target) -> Result<(), anyhow::Error> {
  match target {
    Target::File => (),
    Target::CommandLine => (),
  }
  
  Ok(())
}