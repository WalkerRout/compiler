
use compiler::{
  vm::VirtualMachine,
  chunk::{
    Chunk,
    Opcode,
  },
};

fn main() -> anyhow::Result<()> {
  let mut chunk = Chunk::new();
  let a = chunk.add_constant(1.0);
  let b = chunk.add_constant(2.0);
  chunk.write_byte(Opcode::Constant as u8, 123);
  chunk.write_byte(u8::try_from(a).unwrap(), 123);
  chunk.write_byte(Opcode::Negate as u8, 123);
  chunk.write_byte(Opcode::Constant as u8, 124);
  chunk.write_byte(u8::try_from(b).unwrap(), 124);
  chunk.write_byte(Opcode::Add as u8, 125);
  chunk.write_byte(Opcode::Constant as u8, 126);
  chunk.write_byte(u8::try_from(b).unwrap(), 126);
  chunk.write_byte(Opcode::Multiply as u8, 127);
  chunk.write_byte(Opcode::Negate as u8, 127);
  chunk.write_byte(Opcode::Return as u8, 129);
  chunk.disassemble(&mut std::io::stdout(), "Chunkster McGee").unwrap();
  println!();
  
  let mut vm = VirtualMachine::new(chunk);
  vm.run()?;
  
  Ok(())
}