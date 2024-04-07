
use std::io;
use std::fmt;

#[allow(unused_imports)]
use anyhow::anyhow;

use crate::chunk::Value;
use crate::chunk::Chunk;
use crate::chunk::Opcode;

#[derive(Debug)]
pub enum VirtualMachineError {
  CompileError,
  RuntimeError,
}

impl fmt::Display for VirtualMachineError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::CompileError => write!(f, "VirtualMachineError::CompileError"),
      Self::RuntimeError => write!(f, "VirtualMachineError::RuntimeError"),
    }
  }
}

const STACK_INITIAL: usize = 256;

#[derive(Debug, Clone)]
struct Stack {
  stack: Vec<Value>,
}

impl Stack {
  fn new() -> Self {
    Self {
      stack: Vec::with_capacity(STACK_INITIAL),
    }
  }

  fn push(&mut self, value: Value) {
    self.stack.push(value);
  }

  fn pop(&mut self) -> Value {
    self.stack.pop().expect("stack underflow")
  }

  #[allow(dead_code)]
  fn reset(&mut self) {
    self.stack.clear();
  }
}

#[derive(Debug, Clone)]
pub struct VirtualMachine {
  chunk: Chunk,
  instruction_ptr: usize,
  stack: Stack,
}

impl VirtualMachine {
  #[allow(clippy::must_use_candidate)]
  pub fn new(chunk: Chunk) -> Self {
    Self {
      chunk,
      instruction_ptr: 0,
      stack: Stack::new(),
    }
  }

  /// Run the virtual machine over self.chunk
  ///
  /// # Errors
  ///
  /// Will return `Err` if the following return an error;
  /// - `writeln!()`
  /// - `chunk.disassemble_opcodes`
  pub fn run(&mut self) -> Result<(), anyhow::Error> {
    if cfg!(debug_assertions) {
      println!("== VirtualMachine ==");
    }

    'run: loop {
      if cfg!(debug_assertions) {
        print!("\t");
        print!("stk:[");
        for value in &self.stack.stack {
          print!(" {value} ");
        }
        println!("]");
        self.chunk.disassemble_opcodes(&mut io::stdout(), self.instruction_ptr)?;
      }

      let instruction = self.read_byte();
      if let Ok(byte) = Opcode::try_from(instruction) {
        match byte {
          Opcode::Return => {
            println!("\tret:{}", self.stack.pop());
            break 'run;
          },
          Opcode::Constant => {
            let c = self.read_constant();
            self.stack.push(c);
          },
          Opcode::ConstantLong => {
            let c = self.read_constant_long();
            self.stack.push(c);
          },
          Opcode::Negate => {
            // update in place to avoid pop/push cycle
            self.stack.stack.last_mut().map(|v| *v = -*v);
          },
          Opcode::Add => {
            let b = self.stack.pop();
            let a = self.stack.pop();
            self.stack.push(a + b);
          },
          Opcode::Subtract => {
            let b = self.stack.pop();
            let a = self.stack.pop();
            self.stack.push(a - b);
          },
          Opcode::Multiply => {
            let b = self.stack.pop();
            let a = self.stack.pop();
            self.stack.push(a * b);
          },
          Opcode::Divide => {
            let b = self.stack.pop();
            let a = self.stack.pop();
            self.stack.push(a / b);
          },
        }
      } else {
        // add to error list
      }
    }

    Ok(())
  }

  #[allow(dead_code)]
  fn reset_stack(&mut self) {
    self.stack.reset();
  }

  fn read_byte(&mut self) -> u8 {
    let byte = self.chunk.code[self.instruction_ptr];
    self.instruction_ptr += 1;
    byte
  }

  fn read_constant(&mut self) -> Value {
    let byte = self.read_byte();
    self.chunk.constants[byte as usize]
  }

  fn read_constant_long(&mut self) -> Value {
    let part_a = self.read_byte() as usize;
    let part_b = self.read_byte() as usize;
    let index = part_a << 8 | part_b;
    self.chunk.constants[index]
  }
}

#[cfg(test)]
mod tests {
  use rstest::*;

  use super::*;

  mod virtual_machine_error {
    use super::*;

    #[rstest]
    fn compile_error() {
      todo!()
    }
  }

  mod stack {
    use super::*;
  }

  mod virtual_machine {
    use super::*;

    #[rstest]
    fn new() {
      todo!()
    }

    #[rstest]
    fn run() {
      todo!()
    }
  }

}