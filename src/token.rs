use std::collections::HashMap;
use std::fmt;
use std::sync::OnceLock;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
  Unknown = 0,
  Eof,

  // Identifiers + literals
  Ident,
  Int,
  String,

  // Operators
  Assign,
  Plus,
  Minus,
  Bang,
  Asterisk,
  Slash,

  Lt,
  Gt,

  Eq,
  NotEq,

  // Delimiters
  Dot,
  Comma,
  Semicolon,
  Colon,

  LParen,
  RParen,
  LBrace,
  RBrace,
  LBracket,
  RBracket,

  // Keywords
  Function,
  Let,
  True,
  False,
  If,
  Else,
  Return,
}

impl fmt::Display for Kind {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let s = match self {
      Self::Unknown => "<unknown>",
      Self::Eof => "<eof>",

      // Identifiers + literals
      Self::Ident => "<ident>",
      Self::Int => "<int>",
      Self::String => "<string>",

      // Operators
      Self::Assign => "=",
      Self::Plus => "+",
      Self::Minus => "-",
      Self::Bang => "!",
      Self::Asterisk => "*",
      Self::Slash => "/",

      Self::Lt => "<",
      Self::Gt => ">",

      Self::Eq => "==",
      Self::NotEq => "!=",

      // Delimiters
      Self::Dot => ".",
      Self::Comma => ",",
      Self::Semicolon => ";",
      Self::Colon => ":",

      Self::LParen => "(",
      Self::RParen => ")",
      Self::LBrace => "{",
      Self::RBrace => "}",
      Self::LBracket => "[",
      Self::RBracket => "]",

      // Keywords
      Self::Function => "<function>",
      Self::Let => "<let>",
      Self::True => "<true>",
      Self::False => "<false>",
      Self::If => "<if>",
      Self::Else => "<else>",
      Self::Return => "<return>",
    };
    write!(f, "{s}")
  }
}

static KEYWORDS: OnceLock<HashMap<&'static str, Kind>> = OnceLock::new();

fn init_keywords() -> HashMap<&'static str, Kind> {
  let mut m = HashMap::new();
  m.insert("fn", Kind::Function);
  m.insert("let", Kind::Let);
  m.insert("true", Kind::True);
  m.insert("false", Kind::False);
  m.insert("if", Kind::If);
  m.insert("else", Kind::Else);
  m.insert("return", Kind::Return);
  m
}

pub fn lookup_identifier(s: &str) -> Kind {
  let map = KEYWORDS.get_or_init(init_keywords);
  *map.get(s).unwrap_or(&Kind::Ident)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Token {
  pub kind: Kind,
  pub literal: String,
}

impl Token {
  pub fn new(kind: Kind, literal: impl Into<String>) -> Self {
    Self {
      kind,
      literal: literal.into(),
    }
  }
}
