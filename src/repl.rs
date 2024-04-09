
use std::fs;
use std::io::{self, Write};

#[allow(unused_imports)]
use anyhow::anyhow;

use crate::vm::VirtualMachine;
use crate::compiler::Compiler;

/// Run the compiler with either a REPL or a file, based on args count
///
/// # Errors
///
/// Will return `Err` if the following return an error;
/// - `run_repl`
/// - `run_file`
pub fn run(args: &[String]) -> Result<(), anyhow::Error> {
  match args.len() {
    0 => run_repl()?,
    1 => run_file(&args[0])?,
    _ => eprintln!("Usage: compiler <file_name>"),
  }
  
  Ok(())
}

/// Run compiler in REPL
///
/// # Errors
///
/// Will return `Err` if the following return an error;
/// - `flush()`
/// - `interpret`
fn run_repl() -> Result<(), anyhow::Error> {
  loop {
    print!("> ");
    io::stdout().flush()?;
    if let Some(Ok(input)) = io::stdin().lines().next() {
      if input.trim().is_empty() {
        println!("exiting...");
        break;
      }
      interpret(&input)?;
    }
  }
  Ok(())
}

/// Run compiler over file
///
/// # Errors
///
/// Will return `Err` if the following return an error;
/// - `read_to_string`
fn run_file<A: AsRef<str>>(file_path: A) -> Result<(), anyhow::Error> {
  let file_string = fs::read_to_string(file_path.as_ref())?;
  interpret(file_string)
}

/// Run compiler in REPL
///
/// # Errors
///
/// Returns result of compile in either variant
fn interpret<A: AsRef<str>>(command: A) -> Result<(), anyhow::Error> {
  let mut compiler = Compiler::new(command.as_ref());
  let chunk = compiler.compile()?;
  
  let mut vm = VirtualMachine::new(chunk);
  vm.run()
}
