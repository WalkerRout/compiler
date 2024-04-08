
use std::fs;
use std::io::{self, Write};

#[allow(unused_imports)]
use anyhow::anyhow;

use crate::compiler;

pub enum Target {
  File,
  CommandLine,
}

pub fn run(args: &[String]) -> Result<(), anyhow::Error> {
  match args.len() {
    0 => run_repl()?,
    1 => run_file(&args[0])?,
    _ => eprintln!("Usage: compiler <file_name>"), 
  }
  
  Ok(())
}

fn run_repl() -> Result<(), anyhow::Error> {
  loop {
    print!("> ");
    io::stdout().flush()?;
    match io::stdin().lines().next() {
      Some(Ok(input)) => {
        if input.trim().is_empty() {
          println!("exiting...");
          break;
        }
        interpret(&input)?;
      },
      _ => {},
    }
  }

  Ok(())
}

fn run_file<A: AsRef<str>>(file_path: A) -> Result<(), anyhow::Error> {
  let file_string = fs::read_to_string(file_path.as_ref())?;
  interpret(file_string)
}

fn interpret<A: AsRef<str>>(command: A) -> Result<(), anyhow::Error> {
  println!("Interpret Command:\n{}", command.as_ref());
  compiler::compile(command)
}