
use std::fmt;
use std::iter::Peekable;

use anyhow::anyhow;

use crate::chunk::Chunk;

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub enum TokenType {
  LeftParen,
  RightParen,
  LeftBrace,
  RightBrace,
  Comma,
  Dot,
  Minus,
  Plus,
  Semicolon,
  Slash,
  Star,
  Bang,
  BangEqual,
  Equal,
  EqualEqual,
  Greater,
  GreaterEqual,
  Less,
  LessEqual,
  Identifier(String),
  StringLiteral(String),
  Number(f64),
  And,
  Class,
  Else,
  False,
  For,
  Fun,
  If,
  Nil,
  Or,
  Print,
  Return,
  Super,
  This,
  True,
  Var,
  While,
  Invalid(String), // Todo: store err enum variants
}

impl TokenType {
  fn parse_keyword<A: AsRef<str>>(section: A) -> Result<Self, anyhow::Error> {
    let ty = match section.as_ref() {
      "and" => Self::And,
      "class" => Self::Class,
      "else" => Self::Else,
      "for" => Self::For,
      "false" => Self::False,
      "fun" => Self::Fun,
      "if" => Self::If,
      "nil" => Self::Nil,
      "or" => Self::Or,
      "print" => Self::Print,
      "return" => Self::Return,
      "super" => Self::Super,
      "true" => Self::True,
      "this" => Self::This,
      "var" => Self::Var,
      "while" => Self::While,
      _ => return Err(anyhow!("not a keyword")),
    };
    Ok(ty)
  }
}

impl fmt::Display for TokenType {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::LeftParen => write!(f, "("),
      Self::RightParen => write!(f, ")"),
      Self::LeftBrace => write!(f, "{{"),
      Self::RightBrace => write!(f, "}}"),
      Self::Comma => write!(f, ","),
      Self::Dot => write!(f, "."),
      Self::Minus => write!(f, "-"),
      Self::Plus => write!(f, "+"),
      Self::Semicolon => write!(f, ";"),
      Self::Slash => write!(f, "/"),
      Self::Star => write!(f, "*"),
      Self::Bang => write!(f, "!"),
      Self::BangEqual => write!(f, "!="),
      Self::Equal => write!(f, "="),
      Self::EqualEqual => write!(f, "=="),
      Self::Greater => write!(f, ">"),
      Self::GreaterEqual => write!(f, ">="),
      Self::Less => write!(f, "<"),
      Self::LessEqual => write!(f, "<="),
      Self::Identifier(s) => write!(f, "{s}"),
      Self::StringLiteral(s) => write!(f, "\"{s}\""),
      Self::Number(n) => write!(f, "{n}"),
      Self::And => write!(f, "and"),
      Self::Class => write!(f, "class"),
      Self::Else => write!(f, "else"),
      Self::False => write!(f, "false"),
      Self::For => write!(f, "for"),
      Self::Fun => write!(f, "fun"),
      Self::If => write!(f, "if"),
      Self::Nil => write!(f, "nil"),
      Self::Or => write!(f, "or"),
      Self::Print => write!(f, "print"),
      Self::Return => write!(f, "return"),
      Self::Super => write!(f, "super"),
      Self::This => write!(f, "this"),
      Self::True => write!(f, "true"),
      Self::Var => write!(f, "var"),
      Self::While => write!(f, "while"),
      Self::Invalid(s) => write!(f, "invalid@{s}"),
    }
  }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
  pub line: usize,
  pub token_type: TokenType,
}

impl fmt::Display for Token {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}: {}", self.line, self.token_type)
  }
}

#[derive(Debug, Default)]
struct Scanner<'src> {
  source: &'src str,
  line: usize,
}

impl<'src> Scanner<'src> {
  const fn new(source: &'src str) -> Self {
    Self {
      source,
      line: 1,
    }
  }

  fn scan_tokens(&mut self) -> Vec<Token> {
    let mut char_indices = self.source.char_indices().peekable();
    let mut tokens: Vec<Token> = Vec::new();

    while let Some((pos, ch)) = char_indices.next() {
      let token_type = match ch {
        '(' => TokenType::LeftParen,
        ')' => TokenType::RightParen,
        '{' => TokenType::LeftBrace,
        '}' => TokenType::RightBrace,
        ',' => TokenType::Comma,
        '.' => TokenType::Dot,
        '-' => TokenType::Minus,
        '+' => TokenType::Plus,
        ';' => TokenType::Semicolon,
        '/' => TokenType::Slash,
        '*' => TokenType::Star,
        // multi-part tokens
        '!' => match char_indices.next_if_eq(&(pos + 1, '=')) {
          Some(_) => TokenType::BangEqual,
          None => TokenType::Bang,
        },
        '=' => match char_indices.next_if_eq(&(pos + 1, '=')) {
          Some(_) => TokenType::EqualEqual,
          None => TokenType::Equal,
        },
        '>' => match char_indices.next_if_eq(&(pos + 1, '=')) {
          Some(_) => TokenType::GreaterEqual,
          None => TokenType::Greater,
        },
        '<' => match char_indices.next_if_eq(&(pos + 1, '=')) {
          Some(_) => TokenType::LessEqual,
          None => TokenType::Less,
        },
        '"' => Self::string_type(&mut char_indices),
        ch if ch.is_ascii_digit() => Self::number_type(ch, &mut char_indices),
        ch if ch.is_alphabetic() => Self::identifier_type(ch, &mut char_indices),
        // whitespace
        '\n' => {
          self.line += 1;
          continue;
        },
        ch if ch.is_ascii_whitespace() => continue,
        _ => TokenType::Invalid(ch.into()),
      };
      tokens.push(Token { line: self.line, token_type });
    }

    tokens
  }

  fn read_string<I>(last_matched: &mut char, it: &mut Peekable<I>) -> String 
    where I: Iterator<Item = (usize, char)>, {
    it
      .by_ref()
      .take_while(|(_, c)| {
        *last_matched = *c;
        *c != '"'
      })
      .map(|(_, c)| c)
      .collect()
  }

  fn string_type<I>(it: &mut Peekable<I>) -> TokenType
    where I: Iterator<Item = (usize, char)>, {
    let mut last_matched: char = '\0';
    let s = Self::read_string(&mut last_matched, it);
    match last_matched {
      '"' => TokenType::StringLiteral(s),
      _ => TokenType::Invalid(format!("nonterminated literal \"{s}\"")),
    }
  }

  fn read_number<I>(start: char, it: &mut Peekable<I>) -> String 
    where I: Iterator<Item = (usize, char)>, {
    Self::read_while(start, it, |(_, c)| c.is_ascii_digit() || *c == '.')
  }

  fn number_type<I>(start: char, it: &mut Peekable<I>) -> TokenType
    where I: Iterator<Item = (usize, char)>, {
    let num = Self::read_number(start, it);
    #[allow(clippy::option_if_let_else)]
    match num.parse() {
      Ok(n) => TokenType::Number(n),
      Err(_) => TokenType::Invalid(format!("invalid number {num}")),
    }
  }

  fn read_identifier<I>(start: char, it: &mut Peekable<I>) -> String
    where I: Iterator<Item = (usize, char)>, {
    Self::read_while(start, it, |(_, c)| c.is_alphanumeric() || *c == '_')
  }

  fn identifier_type<I>(start: char, it: &mut Peekable<I>) -> TokenType
    where I: Iterator<Item = (usize, char)>, {
    let ident = Self::read_identifier(start, it);
    TokenType::parse_keyword(&ident).map_or(TokenType::Identifier(ident), |t| t)
  }

  fn read_while<I>(
    start: char,
    it: &mut Peekable<I>,
    f: impl Fn(&(usize, char)) -> bool
  ) -> String
    where I: Iterator<Item = (usize, char)>, {
    let mut compose = String::new();
    let mut ch = start;
    loop {
      compose.push(ch);
      match it.next_if(&f) {
        Some((_, next)) => ch = next,
        None => break,
      }
    }
    compose
  }
}

#[derive(Debug, Clone)]
struct Parser {
  current: Token,
  previous: Token,
  tokens: Vec<Token>,
}

impl Parser {
  fn new(tokens: Vec<Token>) -> Self {
    Self {
      current: tokens[0].clone(),
      previous: tokens[0].clone(),
      tokens,
    }
  }
}

#[derive(Debug, Clone)]
pub struct Compiler<'src> {
  source: &'src str,
}

impl<'src> Compiler<'src> {
  #[allow(clippy::must_use_candidate)]
  pub const fn new(source: &'src str) -> Self {
    Self {
      source,
    }
  }

  /// Compile source code
  ///
  /// # Errors
  ///
  /// No errors, possible, Result is a placeholder
  pub fn compile(&mut self, _chunk: &mut Chunk) -> Result<(), anyhow::Error> {
    let mut scanner = Scanner::new(self.source);
    let mut parser = Parser::new(scanner.scan_tokens());
    let tokens = parser.tokens;

    let mut prev_line = 0;
    for token in tokens {
      if token.line == prev_line {
        print!("|\t");
      } else {
        print!("{}\t", token.line);
        prev_line = token.line;
      }
      println!("'{:?}'", token.token_type); 
    }

    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use rstest::*;

  use super::*;

  mod token_type {
    use super::*;

    #[rstest]
    fn parse_keyword() {
      todo!()
    }

    #[rstest]
    fn display_fmt() {
      todo!()
    }
  }

  mod token {
    use super::*;

    #[rstest]
    fn display_fmt() {
      todo!()
    }
  }

  mod scanner {
    use super::*;

    #[rstest]
    fn new() {
      todo!()
    }

    #[rstest]
    fn scan_tokens() {
      todo!()
    }

    #[rstest]
    fn read_string() {
      todo!()
    }

    #[rstest]
    fn string_type() {
      todo!()
    }

    #[rstest]
    fn read_number() {
      todo!()
    }

    #[rstest]
    fn number_type() {
      todo!()
    }

    #[rstest]
    fn read_identifier() {
      todo!()
    }

    #[rstest]
    fn identifier_type() {
      todo!()
    }

    #[rstest]
    fn read_while() {
      todo!()
    }
  }

  mod parser {
    use super::*;

    #[rstest]
    fn new() {
      todo!()
    }
  }

  mod compiler {
    use super::*;

    #[rstest]
    fn new() {
      todo!()
    }

    #[rstest]
    fn compile() {
      todo!()
    }
  }

}