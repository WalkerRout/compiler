
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
        write!(w, "| ")?;
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

    #[fixture]
    fn chunk_1_constant() -> Chunk {
      Chunk { 
        code: vec![Opcode::Constant as u8, 0],
        constants: vec![1.0],
        lines: Lines {
          lines: vec![Line { line: 1, count: 2 }],
        },
      }
    }

    #[fixture]
    fn chunk_1_constant_long() -> Chunk {
      Chunk {
        code: vec![Opcode::ConstantLong as u8, 0, 0],
        constants: vec![1.0],
        lines: Lines {
          lines: vec![Line { line: 1, count: 3 }]
        }
      }
    }

    #[fixture]
    fn chunk_2_constant() -> Chunk {
      Chunk { 
        code: vec![
          Opcode::Constant as u8, 0,
          Opcode::Constant as u8, 1,
        ],
        constants: vec![1.0, 2.0],
        lines: Lines {
          lines: vec![Line { line: 1, count: 4 }],
        },
      }
    }

    #[fixture]
    fn chunk_3_constant_3_line() -> Chunk {
      Chunk { 
        code: vec![
          Opcode::Constant as u8, 0,
          Opcode::Constant as u8, 1,
          Opcode::Constant as u8, 2,
        ],
        constants: vec![1.0, 2.0, 3.0],
        lines: Lines {
          lines: vec![
            Line { line: 1, count: 2 },
            Line { line: 2, count: 2 },
            Line { line: 3, count: 2 },
          ],
        },
      }
    }

    #[fixture]
    fn chunk_2_constant_add_return_4_line() -> Chunk {
      Chunk { 
        code: vec![
          Opcode::Constant as u8, 0,
          Opcode::Constant as u8, 1,
          Opcode::Add as u8,
          Opcode::Return as u8,
        ],
        constants: vec![1.0, 2.0],
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
    
    #[rstest]
    fn new() {
      let chunk = Chunk::new();
      assert_eq!(chunk, chunk_empty());
    }

    #[rstest]
    #[case(chunk_empty(), 0.0, &[0.0])]
    #[case(chunk_1_constant(), 0.0, &[1.0, 0.0])]
    #[case(chunk_2_constant(), 0.0, &[1.0, 2.0, 0.0])]
    #[case(chunk_3_constant_3_line(), 0.0, &[1.0, 2.0, 3.0, 0.0])]
    #[case(chunk_2_constant_add_return_4_line(), 0.0, &[1.0, 2.0, 0.0])]
    fn add_constant(
      #[case] mut chunk: Chunk,
      #[case] new_constant: Value,
      #[case] expected_constants: &[Value],
    ) {
      let c = chunk.add_constant(new_constant);
      assert_eq!(c, expected_constants.len()-1);
      assert_eq!(&chunk.constants, expected_constants);
    }

    #[rstest]
    #[case(chunk_empty(), 0, 1, Line { line: 1, count: 1 })]
    #[case(chunk_1_constant(), 0, 1, Line { line: 1, count: 3 })]
    #[case(chunk_1_constant(), 0, 2, Line { line: 2, count: 1 })]
    #[case(chunk_2_constant(), 0, 1, Line { line: 1, count: 5 })]
    #[case(chunk_2_constant(), 0, 2, Line { line: 2, count: 1 })]
    #[case(chunk_3_constant_3_line(), 0, 3, Line { line: 3, count: 3 })]
    #[case(chunk_3_constant_3_line(), 0, 4, Line { line: 4, count: 1 })]
    #[case(chunk_2_constant_add_return_4_line(), 0, 4, Line { line: 4, count: 2 })]
    #[case(chunk_2_constant_add_return_4_line(), 0, 5, Line { line: 5, count: 1 })]
    fn write_byte(
      #[case] mut chunk: Chunk,
      #[case] byte: u8,
      #[case] line_n: usize,
      #[case] expected_line: Line,
    ) {
      chunk.write_byte(byte, line_n);
      assert_eq!(chunk.code.last(), Some(byte).as_ref());
      assert_eq!(chunk.lines.lines.last(), Some(expected_line).as_ref());
    }

    #[derive(Debug, Default, Clone)]
    struct StringWriter(String);
    impl io::Write for StringWriter {
      fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let s = String::from_utf8_lossy(buf);
        self.0.push_str(&s);
        Ok(buf.len())
      }

      fn flush(&mut self) -> io::Result<()> {
        Ok(())
      }
    }

    #[rstest]
    #[case(chunk_empty(), r"== chunk ==
")]
    #[case(chunk_1_constant(), r"== chunk ==
0000 1 OP_CONSTANT @constants:0 1
")]
    #[case(chunk_2_constant(), r"== chunk ==
0000 1 OP_CONSTANT @constants:0 1
0002 | OP_CONSTANT @constants:1 2
")]
    #[case(chunk_3_constant_3_line(), r"== chunk ==
0000 1 OP_CONSTANT @constants:0 1
0002 2 OP_CONSTANT @constants:1 2
0004 3 OP_CONSTANT @constants:2 3
")]
    fn disassemble(
      #[case] chunk: Chunk,
      #[case] expected_disassembly: &str,
    ) {
      let mut buffer = StringWriter::default();
      // panic on failure
      chunk.disassemble(&mut buffer, "chunk").unwrap();
      assert_eq!(buffer.0, expected_disassembly);
    }

    #[rstest]
    #[should_panic]
    #[case(chunk_empty(), 0, "")] // no opcodes in empty chunk, invalid offset
    #[case(chunk_1_constant(), 0, "OP_CONSTANT @constants:0 1\n")]
    #[case(chunk_2_constant(), 0, "OP_CONSTANT @constants:0 1\n")]
    #[case(chunk_2_constant(), 2, "OP_CONSTANT @constants:1 2\n")]
    #[case(chunk_3_constant_3_line(), 0, "OP_CONSTANT @constants:0 1\n")]
    #[case(chunk_3_constant_3_line(), 2, "OP_CONSTANT @constants:1 2\n")]
    #[case(chunk_3_constant_3_line(), 4, "OP_CONSTANT @constants:2 3\n")]
    #[case(chunk_2_constant_add_return_4_line(), 0, "OP_CONSTANT @constants:0 1\n")]
    #[case(chunk_2_constant_add_return_4_line(), 2, "OP_CONSTANT @constants:1 2\n")]
    #[case(chunk_2_constant_add_return_4_line(), 4, "OP_ADD\n")]
    #[case(chunk_2_constant_add_return_4_line(), 5, "OP_RETURN\n")]
    fn disassemble_opcodes(
      #[case] chunk: Chunk,
      #[case] offset: usize,
      #[case] expected_disassembly: &str,
    ) {
      let mut buffer = StringWriter::default();
      chunk.disassemble_opcodes(&mut buffer, offset).unwrap();
      assert_eq!(buffer.0, expected_disassembly);
    }

    #[rstest]
    // does not use self in disassemble_single, empty chunk irrelevant
    #[case(chunk_empty(), 0, "OP_RETURN", "OP_RETURN\n")]
    #[case(chunk_empty(), 1, "OP_CONSTANT", "OP_CONSTANT\n")]
    #[case(chunk_empty(), 2, "OP_CONSTANT_LONG", "OP_CONSTANT_LONG\n")]
    #[case(chunk_empty(), 5, "OP_NEGATE", "OP_NEGATE\n")]
    #[case(chunk_empty(), 10, "OP_ADD", "OP_ADD\n")]
    fn disassemble_single(
      #[case] chunk: Chunk,
      #[case] offset: usize,
      #[case] instruction_name: &str,
      #[case] expected_disassembly: &str,
    ) {
      let mut buffer = StringWriter::default();
      let new_offset = chunk.disassemble_single(&mut buffer, offset, instruction_name).unwrap();
      assert_eq!(new_offset, offset as usize + 1);
      assert_eq!(buffer.0, expected_disassembly);
    }

    #[rstest]
    #[should_panic]
    #[case(chunk_empty(), 0, "")]
    #[case(chunk_1_constant(), 0, "OP_CONSTANT @constants:0 1\n")]
    #[case(chunk_2_constant(), 0, "OP_CONSTANT @constants:0 1\n")]
    #[case(chunk_3_constant_3_line(), 0, "OP_CONSTANT @constants:0 1\n")]
    fn disassemble_constant(
      #[case] chunk: Chunk,
      #[case] offset: usize,
      #[case] expected_disassembly: &str,
    ) {
      let mut buffer = StringWriter::default();
      let new_offset = chunk.disassemble_constant(&mut buffer, offset, "OP_CONSTANT").unwrap();
      assert_eq!(new_offset, offset + 2);
      assert_eq!(buffer.0, expected_disassembly);
    }

    #[rstest]
    #[should_panic]
    #[case(chunk_empty(), 0, "")]
    #[case(chunk_1_constant_long(), 0, "OP_CONSTANT_LONG @constants:0 1\n")]
    fn disassemble_constant_long(
      #[case] chunk: Chunk,
      #[case] offset: usize,
      #[case] expected_disassembly: &str,
    ) {
      let mut buffer = StringWriter::default();
      let new_offset = chunk.disassemble_constant_long(&mut buffer, offset, "OP_CONSTANT_LONG").unwrap();
      assert_eq!(new_offset, offset + 3);
      assert_eq!(buffer.0, expected_disassembly);
    }
  }
}