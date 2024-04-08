
use std::fmt;

#[allow(unused_imports)]
use anyhow::anyhow;

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
enum TokenType {
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
      Self::Identifier(s) => write!(f, "{}", s),
      Self::StringLiteral(s) => write!(f, "\"{}\"", s),
      Self::Number(n) => write!(f, "{}", n),
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
      Self::Invalid(s) => write!(f, "invalid@{}", s),
    }
  }
}

#[derive(Debug, Clone, PartialEq)]
struct Token {
  line: usize,
  token_type: TokenType,
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
  fn new(source: &'src str) -> Self {
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
        // strings
        '"' => {
          let mut last_matched: char = '\0';
          let s: String = char_indices
            .by_ref()
            .take_while(|(_, c)| {
              last_matched = *c;
              *c != '"'
            })
            .map(|(_, c)| c)
            .collect();
          match last_matched {
            '"' => TokenType::StringLiteral(s),
            _ => TokenType::Invalid(format!("nonterminated literal \"{s}\"")),
          }
        },
        // numbers
        mut ch if ch.is_ascii_digit() => {
          let mut s = String::new();
          loop {
            s.push(ch);
            match char_indices.next_if(|(_, c)| c.is_ascii_digit() || *c == '.') {
              Some((_, next)) => ch = next,
              None => break,
            }
          }
          match s.parse() {
            Ok(n) => TokenType::Number(n),
            Err(_) => TokenType::Invalid(format!("{s}")),
          }
        }
        // identifiers
        mut ch if ch.is_alphabetic() => {
          TokenType::Identifier(ch.to_string())
        }
        // whitespace
        '\n' => {
          self.line += 1;
          continue;
        },
        ch if ch.is_ascii_whitespace() => continue,
        _ => TokenType::Invalid(format!("{}", ch)),
      };
      tokens.push(Token { line: self.line, token_type });
    }

    tokens
  }
}

pub fn compile<A: AsRef<str>>(source: A) -> Result<(), anyhow::Error> {
  let mut scanner = Scanner::new(source.as_ref());
  let tokens = scanner.scan_tokens();
  
  let mut prev_line = 0;
  for token in tokens {
    if token.line == prev_line {
      print!("|\t");
    } else {
      print!("{}\t", token.line);
      prev_line = token.line;
    }
    println!("'{}'", token.token_type); 
  }

  Ok(())
}