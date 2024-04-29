use std::fmt;
use std::io;

#[allow(unused_imports)]
use anyhow::anyhow;

use crate::chunk::{Chunk, Opcode, Value};

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
  #[allow(clippy::too_many_lines)]
  pub fn run(&mut self) -> Result<(), anyhow::Error> {
    if cfg!(debug_assertions) {
      eprintln!("== VirtualMachine ==");
    }

    'run: loop {
      if cfg!(debug_assertions) {
        eprint!("\t");
        eprint!("stk:[");
        for value in &self.stack.stack {
          eprint!(" {value} ");
        }
        eprintln!("]");
        self
          .chunk
          .disassemble_opcodes(&mut io::stdout(), self.instruction_ptr)?;
      }

      let instruction = self.read_byte();
      if let Ok(byte) = Opcode::try_from(instruction) {
        match byte {
          Opcode::Return => {
            //println!("\tret:{}", self.stack.pop());
            //break 'run;
            todo!()
          }
          Opcode::Constant => {
            let c = self.read_constant();
            self.stack.push(c);
          }
          Opcode::ConstantLong => {
            let c = self.read_constant_long();
            self.stack.push(c);
          }
          Opcode::Negate => {
            let mut was_invalid = false;
            // update in place to avoid pop/push cycle
            #[allow(clippy::option_map_unit_fn)]
            {
              self.stack.stack.last_mut().map(|v| {
                if let Value::Number(ref mut n) = v {
                  *n = -*n;
                } else {
                  was_invalid = true;
                }
              });
            }
            if was_invalid {
              self.runtime_error("operands must be a number");
              return Err(anyhow!(VirtualMachineError::RuntimeError));
            }
          }
          Opcode::Add => {
            let b = self.stack.pop();
            let a = self.stack.pop();
            #[allow(clippy::single_match_else)]
            match (a, b) {
              (Value::String(a), Value::String(b)) => self.stack.push((a + &b).into()),
              (Value::Number(a), Value::Number(b)) => self.stack.push((a + b).into()),
              _ => {
                self.runtime_error("operands must be numbers");
                return Err(anyhow!(VirtualMachineError::RuntimeError));
              }
            }
          }
          Opcode::Subtract => {
            let b = self.stack.pop();
            let a = self.stack.pop();
            #[allow(clippy::single_match_else)]
            match (a, b) {
              (Value::Number(a), Value::Number(b)) => self.stack.push((a - b).into()),
              _ => {
                self.runtime_error("operands must be numbers");
                return Err(anyhow!(VirtualMachineError::RuntimeError));
              }
            }
          }
          Opcode::Multiply => {
            let b = self.stack.pop();
            let a = self.stack.pop();
            #[allow(clippy::single_match_else)]
            match (a, b) {
              (Value::Number(a), Value::Number(b)) => self.stack.push((a * b).into()),
              _ => {
                self.runtime_error("operands must be numbers");
                return Err(anyhow!(VirtualMachineError::RuntimeError));
              }
            }
          }
          Opcode::Divide => {
            let b = self.stack.pop();
            let a = self.stack.pop();
            #[allow(clippy::single_match_else)]
            match (a, b) {
              (Value::Number(a), Value::Number(b)) => self.stack.push((a / b).into()),
              _ => {
                self.runtime_error("operands must be numbers");
                return Err(anyhow!(VirtualMachineError::RuntimeError));
              }
            }
          }
          Opcode::Nil => self.stack.push(Value::Nil),
          Opcode::True => self.stack.push(Value::Bool(true)),
          Opcode::False => self.stack.push(Value::Bool(false)),
          Opcode::Not => {
            let v = self.stack.pop();
            let is_falsy = v.is_falsy();
            self.stack.push(Value::Bool(is_falsy));
          }
          Opcode::Equal => {
            let b = self.stack.pop();
            let a = self.stack.pop();
            self.stack.push(Value::Bool(a == b));
          }
          Opcode::Less => {
            let b = self.stack.pop();
            let a = self.stack.pop();
            #[allow(clippy::single_match_else)]
            match (a, b) {
              (Value::Number(a), Value::Number(b)) => self.stack.push((a < b).into()),
              _ => {
                self.runtime_error("operands must be numbers");
                return Err(anyhow!(VirtualMachineError::RuntimeError));
              }
            }
          }
          Opcode::Greater => {
            let b = self.stack.pop();
            let a = self.stack.pop();
            #[allow(clippy::single_match_else)]
            match (a, b) {
              (Value::Number(a), Value::Number(b)) => self.stack.push((a > b).into()),
              _ => {
                self.runtime_error("operands must be numbers");
                return Err(anyhow!(VirtualMachineError::RuntimeError));
              }
            }
          }
          Opcode::Print => println!("\"{}\"", self.stack.pop()),
          Opcode::Pop => {
            self.stack.pop();
          }
        }
      } else {
        // add to error list
      }
    }

    Ok(())
  }

  fn read_byte(&mut self) -> u8 {
    let byte = self.chunk.code[self.instruction_ptr];
    self.instruction_ptr += 1;
    byte
  }

  fn read_constant(&mut self) -> Value {
    let byte = self.read_byte();
    self.chunk.constants[byte as usize].clone()
  }

  fn read_constant_long(&mut self) -> Value {
    let part_a = self.read_byte() as usize;
    let part_b = self.read_byte() as usize;
    let index = part_a << 8 | part_b;
    self.chunk.constants[index].clone()
  }

  fn runtime_error(&mut self, msg: &str) {
    let ip = self.instruction_ptr;
    let line = self.chunk.lines.line(ip);
    eprintln!("{msg}\n\t[line {line}] in script");
    self.stack.reset();
  }

  fn compile_error(&mut self, _msg: &str) {
    todo!()
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use rstest::*;

  mod virtual_machine_error {
    use super::*;

    #[rstest]
    #[case(VirtualMachineError::RuntimeError, "RuntimeError")]
    #[case(VirtualMachineError::CompileError, "CompileError")]
    fn display_fmt(#[case] variant: VirtualMachineError, #[case] expected_debug_fmt: &str) {
      assert_eq!(format!("{variant:?}"), expected_debug_fmt);
    }
  }

  mod stack {
    use super::*;
    use Value as v;

    #[fixture]
    fn stack_empty() -> Stack {
      Stack { stack: Vec::new() }
    }

    #[fixture]
    fn stack_1_value() -> Stack {
      Stack {
        stack: vec![1.0.into()],
      }
    }

    #[fixture]
    fn stack_4_values() -> Stack {
      Stack {
        stack: vec![1.0.into(), 2.0.into(), 3.0.into(), 4.0.into()],
      }
    }

    #[rstest]
    fn new() {
      let stack = Stack::new();
      assert_eq!(stack, stack_empty());
      assert_eq!(stack.stack.capacity(), STACK_INITIAL);
    }

    #[rstest]
    #[case(stack_empty(), 0.0.into(), &[0.0.into()])]
    #[case(stack_1_value(), 0.0.into(), &[1.0.into(), 0.0.into()])]
    #[case(stack_4_values(), 0.0.into(), &[1.0.into(), 2.0.into(), 3.0.into(), 4.0.into(), 0.0.into()])]
    fn push(#[case] mut stack: Stack, #[case] value: Value, #[case] expected_values: &[Value]) {
      stack.push(value);
      assert_eq!(&stack.stack, expected_values);
    }

    #[rstest]
    #[should_panic]
    #[case(stack_empty(), 0.0.into(), &[])]
    #[case(stack_1_value(), 1.0.into(), &[])]
    #[case(stack_4_values(), 4.0.into(), &[1.0.into(), 2.0.into(), 3.0.into()])]
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
    fn reset(#[case] mut stack: Stack) {
      stack.reset();
      assert_eq!(&stack.stack, &[]);
    }
  }

  mod virtual_machine {
    use super::*;
    use crate::chunk::{Line, Lines};
    use Value as v;

    #[fixture]
    fn chunk_empty() -> Chunk {
      Chunk::new()
    }

    #[fixture]
    fn chunk_add_2() -> Chunk {
      Chunk {
        code: vec![
          Opcode::Constant as u8,
          0,
          Opcode::Constant as u8,
          1,
          Opcode::Add as u8,
          Opcode::Return as u8,
        ],
        constants: vec![1.0.into(), 2.0.into()],
        lines: Lines {
          lines: vec![
            Line { line: 1, count: 2 },
            Line { line: 2, count: 2 },
            Line { line: 3, count: 1 },
            Line { line: 4, count: 1 },
          ],
        },
      }
    }

    #[fixture]
    fn stack_empty() -> Stack {
      Stack::new()
    }

    #[fixture]
    fn virtual_machine_empty() -> VirtualMachine {
      VirtualMachine {
        chunk: chunk_empty(),
        instruction_ptr: 0,
        stack: stack_empty(),
      }
    }

    #[fixture]
    fn virtual_machine_add_2() -> VirtualMachine {
      VirtualMachine {
        chunk: chunk_add_2(),
        instruction_ptr: 0,
        stack: stack_empty(),
      }
    }

    #[rstest]
    fn new() {
      let chunk = chunk_empty();
      let vm = VirtualMachine::new(chunk);
      assert_eq!(vm, virtual_machine_empty());
    }

    // main tests for application logic
    #[rstest]
    fn run() {
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

    #[rstest]
    fn runtime_error() {
      todo!()
    }

    #[rstest]
    fn compile_error() {
      todo!()
    }
  }
}
