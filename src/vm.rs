
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

#[derive(Debug, Clone, PartialEq)]
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

#[derive(Debug, Clone, PartialEq)]
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
            #[allow(clippy::option_map_unit_fn)]
            { self.stack.stack.last_mut().map(|v| *v = -*v); }
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
    fn display_fmt() {
      todo!()
    }
  }

  mod stack {
    use super::*;

    #[fixture]
    fn stack_empty() -> Stack {
      Stack {
        stack: Vec::new(),
      }
    }

    #[fixture]
    fn stack_1_value() -> Stack {
      Stack {
        stack: vec![1.0],
      }
    }

    #[fixture]
    fn stack_4_values() -> Stack {
      Stack {
        stack: vec![1.0, 2.0, 3.0, 4.0],
      }
    }

    #[rstest]
    fn new() {
      let stack = Stack::new();
      assert_eq!(stack, stack_empty());
      assert_eq!(stack.stack.capacity(), STACK_INITIAL);
    }

    #[rstest]
    #[case(stack_empty(), 0.0, &[0.0])]
    #[case(stack_1_value(), 0.0, &[1.0, 0.0])]
    #[case(stack_4_values(), 0.0, &[1.0, 2.0, 3.0, 4.0, 0.0])]
    fn push(
      #[case] mut stack: Stack,
      #[case] value: Value,
      #[case] expected_values: &[Value],
    ) {
      stack.push(value);
      assert_eq!(&stack.stack, expected_values);
    }

    #[rstest]
    #[should_panic]
    #[case(stack_empty(), 0.0, &[])]
    #[case(stack_1_value(), 1.0, &[])]
    #[case(stack_4_values(), 4.0, &[1.0, 2.0, 3.0])]
    fn pop(
      #[case] mut stack: Stack,
      #[case] expected_value: Value,
      #[case] expected_values: &[Value],
    ) {
      assert_eq!(stack.pop(), expected_value);
      assert_eq!(&stack.stack, expected_values);
    }

    #[rstest]
    #[case(stack_empty())]
    #[case(stack_1_value())]
    #[case(stack_4_values())]
    fn reset(
      #[case] mut stack: Stack,
    ) {
      stack.reset();
      assert_eq!(&stack.stack, &[]);
    }
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

    #[rstest]
    fn reset_stack() {
      todo!()
    }

    #[rstest]
    fn read_byte() {
      todo!()
    }

    #[rstest]
    fn read_constant() {
      todo!()
    }

    #[rstest]
    fn read_constant_long() {
      todo!()
    }
  }

}