use std::io::{BufRead, BufReader, Read, Write};

use crate::ast::Visitor;
use crate::interpreter::{Environment, Interpreter};
use crate::lexer::Lexer;
use crate::parser::Parser;

/// Starts a simple read-eval-print-loop (REPL)
///
/// # Errors
///
/// Returns an [`anyhow::Error`] if:
/// - The prompt cannot be flushed to the output
/// - Reading from the input fails
/// - Any IO-related error occurs during the REPL loop
///
pub fn repl<R: Read, W: Write>(r#in: R, out: W) -> Result<(), anyhow::Error> {
  let mut reader = BufReader::new(r#in);
  let mut writer = out;

  let environment = Environment::new();
  loop {
    print!("-> ");
    writer.flush()?;

    let mut buffer = String::new();
    reader.read_line(&mut buffer)?;
    let input = buffer.trim_end();

    if input.trim().is_empty() {
      println!("exiting...");
      break;
    }

    let lexer = Lexer::new(input);
    let parser = Parser::new(lexer);
    let program = match parser.parse() {
      Ok(program) => program,
      Err(errors) => {
        println!("parser errors: {errors:?}");
        continue;
      }
    };
    let mut interpreter = Interpreter::new(environment.clone());
    println!("ast: {program:?}");
    let value = interpreter.visit_program(&program);
    println!("{value:?}");
  }

  Ok(())
}
