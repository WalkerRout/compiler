
use std::io;
use std::fmt;

use anyhow::anyhow;

pub type Value = f64;

/// Assert certain trait implementations exist on Value at compile time:
///
/// - `Display`
const _: () = {
  const fn assert() {
    assert_display::<Value>();
  }
  const fn assert_display<T: fmt::Display>() {}
};

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Opcode {
  Return = 0,
  Constant = 1,
  ConstantLong = 2,
  Negate = 3,
  Add = 4,
  Subtract = 5,
  Multiply = 6,
  Divide = 7,
}

impl TryFrom<u8> for Opcode {
  type Error = anyhow::Error;

  fn try_from(value: u8) -> Result<Self, Self::Error> {
    let result = match value {
      0 => Self::Return,
      1 => Self::Constant,
      2 => Self::ConstantLong,
      3 => Self::Negate,
      4 => Self::Add,
      5 => Self::Subtract,
      6 => Self::Multiply,
      7 => Self::Divide,
      _ => return Err(anyhow!("invalid opcode")),
    };
    Ok(result)
  }
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
struct Line {
  line: usize,
  count: usize,
}

#[derive(Debug, Default, Clone, PartialEq)]
struct Lines {
  lines: Vec<Line>,
}

impl Lines {
  const fn new() -> Self {
    Self { lines: Vec::new(), }
  }

  fn line(&self, offset: usize) -> usize {
    let mut result = 0;
    let mut prev_count = 0;
    for line in &self.lines {
      let curr_count = line.count + prev_count;
      if prev_count <= offset && offset < curr_count {
        result = line.line;
      }
      prev_count = curr_count;
    }
    result
  }

  fn write_line(&mut self, line: usize) {
    let previous_line = self.lines.last().map(|l| l.line);
    // check if line is same as previous, otherwise write a new line
    if Some(line) == previous_line {
      // safe to unwrap, we know previous line in self.lines is Some(line)
      self.lines.last_mut().unwrap().count += 1;
    } else {
      self.lines.push(Line { line, count: 1 });
    }
  }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Chunk {
  pub code: Vec<u8>,
  pub constants: Vec<Value>,
  lines: Lines,
}

impl Chunk {
  #[allow(clippy::must_use_candidate)]
  pub fn new() -> Self {
    Self { 
      code: Vec::new(),
      constants: Vec::new(),
      lines: Lines::new(),
    }
  }

  pub fn add_constant(&mut self, value: Value) -> usize {
    self.constants.push(value);
    self.constants.len() - 1
  }

  pub fn write_byte(&mut self, byte: u8, line: usize) {
    self.code.push(byte);
    self.lines.write_line(line);
  }

  /// Disassemble print chunk to stdout
  ///
  /// # Errors
  ///
  /// Will return `Err` if the following return an error;
  /// - `writeln!()`
  /// - `self.disassemble_opcodes`
  pub fn disassemble<A: AsRef<str>>(
    &self, 
    w: &mut impl io::Write, 
    name: A
  ) -> Result<(), anyhow::Error> {
    let name = name.as_ref();
    writeln!(w, "== {name} ==")?;

    let mut offset = 0;
    while offset < self.code.len() {
      write!(w, "{offset:04} ")?;
      if offset > 0 && self.lines.line(offset) == self.lines.line(offset - 1) {
        write!(w, "  | ")?;
      } else {
        write!(w, "{} ", self.lines.line(offset))?;
      }
      offset = self.disassemble_opcodes(w, offset)?;
    }

    Ok(())
  }

  /// Disassemble single opcode of chunk and advance offset
  ///
  /// # Errors
  ///
  /// Will return `Err` if the following return an error;
  /// - `writeln!()`
  /// - `self.disassemble_single`
  /// - `self.disassemble_constant`
  /// - `self.disassemble_constant_long`
  pub fn disassemble_opcodes(
    &self, 
    w: &mut impl io::Write, 
    offset: usize
  ) -> Result<usize, anyhow::Error> {
    let new_offset = {
      if let Ok(byte) = Opcode::try_from(self.code[offset]) {
        match byte {
          Opcode::Return => self.disassemble_single(w, offset, "OP_RETURN")?,
          Opcode::Constant => self.disassemble_constant(w, offset, "OP_CONSTANT")?,
          Opcode::ConstantLong => self.disassemble_constant_long(w, offset, "OP_CONSTANT_LONG")?,
          Opcode::Negate => self.disassemble_single(w, offset, "OP_NEGATE")?,
          Opcode::Add => self.disassemble_single(w, offset, "OP_ADD")?,
          Opcode::Subtract => self.disassemble_single(w, offset, "OP_SUBTRACT")?,
          Opcode::Multiply => self.disassemble_single(w, offset, "OP_MULTIPLY")?,
          Opcode::Divide => self.disassemble_single(w, offset, "OP_DIVIDE")?,
        }
      } else {
        writeln!(w, "OP_INVALID @offset:{offset}")?;
        offset + 1
      }
    };

    Ok(new_offset)
  }

  /// Print name of single instruction and advance offset by 1
  ///
  /// # Errors
  ///
  /// Will return `Err` if the following return an error;
  /// - `writeln!()`
  #[allow(clippy::unused_self)]
  fn disassemble_single(
    &self,
    w: &mut impl io::Write, 
    offset: usize, 
    name: &str
  ) -> Result<usize, anyhow::Error> {
    writeln!(w, "{name}")?;
    Ok(offset + 1)
  }

  /// Print a constant at index specified by offset+1
  ///
  /// # Errors
  ///
  /// Will return `Err` if the following return an error;
  /// - `writeln!()`
  fn disassemble_constant(
    &self, 
    w: &mut impl io::Write, 
    offset: usize, 
    name: &str
  ) -> Result<usize, anyhow::Error> {
    let index = self.code[offset+1] as usize;
    let constant = self.constants[index];
    writeln!(w, "{name} @constants:{index} {constant}")?;
    Ok(offset + 2)
  }

  /// Print a constant at index specified by (offset+1 << 8 | offset+2)
  ///
  /// # Errors
  ///
  /// Will return `Err` if the following return an error;
  /// - `writeln!()`
  fn disassemble_constant_long(
    &self, 
    w: &mut impl io::Write, 
    offset: usize,
    name: &str
  ) -> Result<usize, anyhow::Error> {
    let part_a = self.code[offset+1] as usize;
    let part_b = self.code[offset+2] as usize;
    let index = part_a << 8 | part_b;
    let constant = self.constants[index];
    writeln!(w, "{name} @constants:{index} {constant}")?;
    Ok(offset + 3)
  }
}

#[cfg(test)]
mod tests {
  use rstest::*;

  use super::*;

  mod opcode {
    use super::*;

    #[rstest]
    #[case(0, Opcode::Return)]
    #[case(1, Opcode::Constant)]
    #[case(2, Opcode::ConstantLong)]
    #[case(3, Opcode::Negate)]
    #[case(4, Opcode::Add)]
    #[case(5, Opcode::Subtract)]
    #[case(6, Opcode::Multiply)]
    #[case(7, Opcode::Divide)]
    fn try_from_u8(
      #[case] _byte: u8,
      #[case] _expected: Opcode,
    ) {
      assert_eq!(Opcode::try_from(_byte).unwrap(), _expected);
    }
  }

  mod lines {
    use super::*;

    #[fixture]
    fn lines_empty() -> Lines {
      Lines { lines: Vec::new(), }
    }

    #[fixture]
    fn lines_4_separate() -> Lines {
      let mut lines = Lines { lines: Vec::new(), };
      lines.lines = vec![
        Line { line: 1, count: 1, },
        Line { line: 2, count: 1, },
        Line { line: 3, count: 1, },
        Line { line: 4, count: 1, },
      ];
      lines
    }

    #[fixture]
    fn lines_4_2_joined() -> Lines {
      let mut lines = Lines { lines: Vec::new(), };
      lines.lines = vec![
        Line { line: 1, count: 2, },
        Line { line: 2, count: 2, },
      ];
      lines
    }

    #[fixture]
    fn lines_7_3_joined() -> Lines {
      let mut lines = Lines { lines: Vec::new(), };
      lines.lines = vec![
        Line { line: 1, count: 2, },
        Line { line: 2, count: 1, },
        Line { line: 3, count: 2, },
        Line { line: 4, count: 2, },
      ];
      lines
    }

    #[rstest]
    fn new(lines_empty: Lines) {
      let lines = Lines::new();
      assert_eq!(lines, lines_empty);
    }

    #[rstest]
    #[case(lines_empty(), 0, 0)]
    #[case(lines_4_separate(), 0, 1)]
    #[case(lines_4_separate(), 1, 2)]
    #[case(lines_4_separate(), 2, 3)]
    #[case(lines_4_separate(), 3, 4)]
    #[case(lines_4_2_joined(), 0, 1)]
    #[case(lines_4_2_joined(), 1, 1)]
    #[case(lines_4_2_joined(), 2, 2)]
    #[case(lines_4_2_joined(), 3, 2)]
    #[case(lines_7_3_joined(), 0, 1)]
    #[case(lines_7_3_joined(), 1, 1)]
    #[case(lines_7_3_joined(), 2, 2)]
    #[case(lines_7_3_joined(), 3, 3)]
    #[case(lines_7_3_joined(), 4, 3)]
    #[case(lines_7_3_joined(), 5, 4)]
    #[case(lines_7_3_joined(), 6, 4)]
    fn line(
      #[case] lines: Lines,
      #[case] offset: usize,
      #[case] expected_line: usize,
    ) {
      assert_eq!(lines.line(offset), expected_line);
    }

    #[rstest]
    #[case(lines_empty(), 1, 1)]
    #[case(lines_4_separate(), 4, 2)]
    #[case(lines_4_separate(), 5, 1)]
    #[case(lines_4_2_joined(), 2, 3)]
    #[case(lines_4_2_joined(), 3, 1)]
    #[case(lines_7_3_joined(), 4, 3)]
    #[case(lines_7_3_joined(), 5, 1)]
    fn write_line(
      #[case] mut lines: Lines,
      #[case] new_line: usize,
      #[case] expected_last_count: usize,
    ) {
      lines.write_line(new_line);
      let last_count = lines.lines.last().unwrap().count as usize;
      assert_eq!(last_count, expected_last_count);
    }
  }

  mod chunk {
    use super::*;

    #[fixture]
    fn chunk_empty() -> Chunk {
      Chunk { 
        code: Vec::new(),
        constants: Vec::new(),
        lines: Lines {
          lines: Vec::new(),
        },
      }
    }
    
    #[rstest]
    fn new() {
      todo!()
    }

    #[rstest]
    fn add_constant() {
      todo!()
    }

    #[rstest]
    fn write_byte() {
      todo!()
    }

    #[rstest]
    fn disassemble() {
      todo!()
    }

    #[rstest]
    fn disassemble_opcodes() {
      todo!()
    }

    #[rstest]
    fn disassemble_single() {
      todo!()
    }

    #[rstest]
    fn disassemble_constant() {
      todo!()
    }

    #[rstest]
    fn disassemble_constant_long() {
      todo!()
    }
  }
}