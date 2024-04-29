use std::cell::Cell;
use std::fmt;
use std::io;
use std::iter::Peekable;
use std::mem;

use anyhow::anyhow;

use crate::chunk::{Chunk, Opcode, Value};

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
  Eof,
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
      Self::Eof => write!(f, "EOF"),
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
    Self { source, line: 1 }
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
        }
        ch if ch.is_ascii_whitespace() => continue,
        _ => TokenType::Invalid(ch.into()),
      };
      tokens.push(Token {
        line: self.line,
        token_type,
      });
    }
    tokens.push(Token {
      line: self.line,
      token_type: TokenType::Eof,
    });
    tokens
  }

  fn read_string<I>(last_matched: &mut char, it: &mut Peekable<I>) -> String
  where
    I: Iterator<Item = (usize, char)>,
  {
    it.by_ref()
      .take_while(|(_, c)| {
        *last_matched = *c;
        *c != '"'
      })
      .map(|(_, c)| c)
      .collect()
  }

  fn string_type<I>(it: &mut Peekable<I>) -> TokenType
  where
    I: Iterator<Item = (usize, char)>,
  {
    let mut last_matched: char = '\0';
    let s = Self::read_string(&mut last_matched, it);
    match last_matched {
      '"' => TokenType::StringLiteral(s),
      _ => TokenType::Invalid(format!("nonterminated literal \"{s}\"")),
    }
  }

  fn read_number<I>(start: char, it: &mut Peekable<I>) -> String
  where
    I: Iterator<Item = (usize, char)>,
  {
    Self::read_while(start, it, |(_, c)| c.is_ascii_digit() || *c == '.')
  }

  fn number_type<I>(start: char, it: &mut Peekable<I>) -> TokenType
  where
    I: Iterator<Item = (usize, char)>,
  {
    let num = Self::read_number(start, it);
    #[allow(clippy::option_if_let_else)]
    match num.parse() {
      Ok(n) => TokenType::Number(n),
      Err(_) => TokenType::Invalid(format!("invalid number {num}")),
    }
  }

  fn read_identifier<I>(start: char, it: &mut Peekable<I>) -> String
  where
    I: Iterator<Item = (usize, char)>,
  {
    Self::read_while(start, it, |(_, c)| c.is_alphanumeric() || *c == '_')
  }

  fn identifier_type<I>(start: char, it: &mut Peekable<I>) -> TokenType
  where
    I: Iterator<Item = (usize, char)>,
  {
    let ident = Self::read_identifier(start, it);
    TokenType::parse_keyword(&ident).map_or(TokenType::Identifier(ident), |t| t)
  }

  fn read_while<I>(start: char, it: &mut Peekable<I>, f: impl Fn(&(usize, char)) -> bool) -> String
  where
    I: Iterator<Item = (usize, char)>,
  {
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

#[repr(u8)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Precedence {
  #[default]
  None,
  Assignment, // =
  Or,         // or
  And,        // and
  Equality,   // == !=
  Comparison, // < > <= >=
  Term,       // + -
  Factor,     // * /
  Unary,      // ! -
  Call,       // . ()
  Primary,
}

impl Precedence {
  const fn into_higher(self) -> Self {
    match self {
      Self::None => Self::Assignment,
      Self::Assignment => Self::Or,
      Self::Or => Self::And,
      Self::And => Self::Equality,
      Self::Equality => Self::Comparison,
      Self::Comparison => Self::Term,
      Self::Term => Self::Factor,
      Self::Factor => Self::Unary,
      Self::Unary => Self::Call,
      Self::Call | Self::Primary => Self::Primary,
    }
  }
}

/// TODO: change fn pointer to fn item
/// - Benchmark `type ParserFn<'cnk, I> = Box<dyn Fn(&mut Parser<'cnk, I>)>;`
type ParserFn<'cnk, I> = fn(&mut Parser<'cnk, I>);

#[derive(Default)]
struct ParserRule<'cnk, I> {
  prefix: Option<ParserFn<'cnk, I>>,
  infix: Option<ParserFn<'cnk, I>>,
  precedence: Precedence,
}

impl<'cnk, I> ParserRule<'cnk, I> {
  fn new(
    prefix: Option<ParserFn<'cnk, I>>,
    infix: Option<ParserFn<'cnk, I>>,
    precedence: Precedence,
  ) -> Self {
    Self {
      prefix,
      infix,
      precedence,
    }
  }
}

impl<'cnk, I> fmt::Debug for ParserRule<'cnk, I> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "ParseRule {{ ..., ..., {:?}, }} ", self.precedence)
  }
}

#[derive(Debug)]
struct Parser<'cnk, It> {
  current: Token,
  previous: Token,
  tokens: It,
  had_error: Cell<bool>,
  panic_mode: Cell<bool>,
  chunk: &'cnk mut Chunk,
}

impl<'cnk, It> Parser<'cnk, It>
where
  It: Iterator<Item = Token>,
{
  fn new<T>(tokens: T, chunk: &'cnk mut Chunk) -> Self
  where
    T: IntoIterator<Item = It::Item, IntoIter = It>,
  {
    let mut this = Self {
      current: Token {
        line: 1,
        token_type: TokenType::Eof,
      },
      previous: Token {
        line: 1,
        token_type: TokenType::Eof,
      },
      tokens: tokens.into_iter(),
      had_error: Cell::new(false),
      panic_mode: Cell::new(false),
      chunk,
    };
    this.advance();
    this
  }

  fn advance(&mut self) {
    self.previous = self.current.clone();
    loop {
      if let Some(t) = self.tokens.next() {
        self.current = t;
      }
      match self.current.token_type {
        TokenType::Invalid(_) => {}
        _ => break,
      }
      self.error_at_current(&self.current.to_string());
    }
  }

  fn consume_if_eq(&mut self, token_type: &TokenType, msg: &str) {
    if mem::discriminant(token_type) == mem::discriminant(&self.current.token_type) {
      self.advance();
      return;
    }
    self.error_at_current(msg);
  }

  fn parse_expression(&mut self) {
    self.parse_precedence(Precedence::Assignment);
  }

  fn parse_number(&mut self) {
    if let TokenType::Number(n) = self.previous.token_type {
      let value = Value::Number(n);
      self.emit_constant(value);
    } else {
      self.error("not a number");
    }
  }

  fn parse_grouping(&mut self) {
    self.parse_expression();
    self.consume_if_eq(&TokenType::RightParen, "expect ')' after expression");
  }

  fn parse_unary(&mut self) {
    let token_type = self.previous.token_type.clone();

    self.parse_precedence(Precedence::Unary);

    match token_type {
      TokenType::Minus => self.emit_byte(Opcode::Negate as u8),
      TokenType::Bang => self.emit_byte(Opcode::Not as u8),
      _ => self.error("not a unary operator"),
    }
  }

  fn parse_binary(&mut self) {
    let token_type = self.previous.token_type.clone();
    let token_parse_rule = Self::get_rule(&token_type);

    self.parse_precedence(token_parse_rule.precedence.into_higher());

    match token_type {
      TokenType::Plus => self.emit_byte(Opcode::Add as u8),
      TokenType::Minus => self.emit_byte(Opcode::Subtract as u8),
      TokenType::Star => self.emit_byte(Opcode::Multiply as u8),
      TokenType::Slash => self.emit_byte(Opcode::Divide as u8),
      TokenType::BangEqual => self.emit_2_bytes(Opcode::Equal as u8, Opcode::Not as u8),
      TokenType::EqualEqual => self.emit_byte(Opcode::Equal as u8),
      TokenType::Less => self.emit_byte(Opcode::Less as u8),
      TokenType::LessEqual => self.emit_2_bytes(Opcode::Greater as u8, Opcode::Not as u8),
      TokenType::Greater => self.emit_byte(Opcode::Greater as u8),
      TokenType::GreaterEqual => self.emit_2_bytes(Opcode::Less as u8, Opcode::Not as u8),
      _ => self.error("not a binary operator"),
    }
  }

  fn parse_precedence(&mut self, precedence: Precedence) {
    self.advance();

    let prefix_rule = Self::get_rule(&self.previous.token_type).prefix;
    if let Some(rule) = prefix_rule {
      rule(self);
    } else {
      self.error("expect expression, no rule defined");
      return;
    }

    while precedence as u8 <= Self::get_rule(&self.current.token_type).precedence as u8 {
      self.advance();
      let infix_rule = Self::get_rule(&self.previous.token_type).infix;
      if let Some(rule) = infix_rule {
        rule(self);
      }
    }
  }

  fn parse_literal(&mut self) {
    match self.previous.token_type {
      TokenType::Nil => self.emit_byte(Opcode::Nil as u8),
      TokenType::True => self.emit_byte(Opcode::True as u8),
      TokenType::False => self.emit_byte(Opcode::False as u8),
      _ => self.error("expect literal"),
    }
  }

  fn parse_string(&mut self) {
    if let TokenType::StringLiteral(ref mut s) = self.previous.token_type {
      let s = std::mem::take(s);
      self.emit_constant(Value::String(s));
    } else {
      self.error("expect string");
    }
  }

  fn parse_declaration(&mut self) {
    if self.match_token(&TokenType::Var) {
      self.parse_var_declaration();
    } else {
      self.parse_statement();
    }
    if self.panic_mode.get() {
      self.synchronize();
    }
  }

  fn parse_statement(&mut self) {
    if self.match_token(&TokenType::Print) {
      self.parse_print_statement();
    } else {
      self.parse_expression_statement();
    }
  }

  fn parse_print_statement(&mut self) {
    self.parse_expression();
    self.consume_if_eq(&TokenType::Semicolon, "expect ';' after value");
    self.emit_byte(Opcode::Print as u8);
  }

  fn parse_expression_statement(&mut self) {
    self.parse_expression();
    self.consume_if_eq(&TokenType::Semicolon, "expect ';' after expression");
    self.emit_byte(Opcode::Pop as u8);
  }

  fn parse_var_declaration(&mut self) {
    let global = self.parse_variable("expect variable name");

    if self.match_token(&TokenType::Equal) {
      self.parse_expression();
    } else {
      self.emit_byte(Opcode::Nil as u8);
    }

    self.consume_if_eq(
      &TokenType::Semicolon,
      "expect ';' after variable declaration",
    );
    self.define_variable(global);
  }

  fn parse_variable(&mut self, panic_message: &str) -> u8 {
    self.consume_if_eq(&TokenType::Identifier(String::new()), panic_message);
    return self.parse_identifier_constant();
  }

  fn define_variable(&mut self, index: u8) {
    self.emit_2_bytes(Opcode::DefineGlobal as u8, index);
  }

  fn parse_identifier_constant(&mut self) -> u8 {
    let value = Value::String(format!("{}", self.previous.token_type));
    self.make_constant(value)
  }

  fn synchronize(&mut self) {
    self.panic_mode.set(false);
    while self.current.token_type != TokenType::Eof {
      if self.previous.token_type == TokenType::Semicolon {
        return;
      }
      match self.current.token_type {
        TokenType::Class
        | TokenType::Fun
        | TokenType::Var
        | TokenType::For
        | TokenType::If
        | TokenType::While
        | TokenType::Print
        | TokenType::Return => return,
        _ => (),
      }
      self.advance();
    }
  }

  fn emit_return(&mut self) {
    self.emit_byte(Opcode::Return as u8);
  }

  fn emit_constant(&mut self, value: Value) {
    let constant = self.make_constant(value);
    self.emit_2_bytes(Opcode::Constant as u8, constant);
  }

  fn make_constant(&mut self, value: Value) -> u8 {
    let constant = self.chunk.add_constant(value);
    if constant > u8::MAX.into() {
      self.error("too many constants in one chunk");
      return 0;
    }
    // safe to unwrap, we check that constant is <= u8::MAX
    u8::try_from(constant).unwrap()
  }

  fn emit_byte(&mut self, byte: u8) {
    self.chunk.write_byte(byte, self.previous.line);
  }

  fn emit_2_bytes(&mut self, byte_a: u8, byte_b: u8) {
    self.emit_byte(byte_a);
    self.emit_byte(byte_b);
  }

  fn match_token(&mut self, token_type: &TokenType) -> bool {
    if !self.check_token(token_type) {
      return false;
    }
    self.advance();
    true
  }

  fn check_token(&mut self, token_type: &TokenType) -> bool {
    self.current.token_type == *token_type
  }

  fn error(&self, msg: &str) {
    self.error_at(&self.previous, msg);
  }

  fn error_at_current(&self, msg: &str) {
    self.error_at(&self.current, msg);
  }

  fn error_at(&self, token: &Token, msg: &str) {
    if self.panic_mode.get() {
      return;
    }
    self.panic_mode.set(true);

    eprint!("[line {}] Error", token.line);

    match token.token_type {
      TokenType::Eof => eprint!(" at end"),
      _ => eprint!(" at '{token:?}'"),
    }

    eprintln!(": {msg}");
    self.had_error.set(true);
  }

  fn get_rule(token_type: &TokenType) -> ParserRule<'cnk, It> {
    #[allow(clippy::match_same_arms)]
    match token_type {
      TokenType::LeftParen => ParserRule::new(Some(Self::parse_grouping), None, Precedence::None),
      TokenType::RightParen => ParserRule::new(None, None, Precedence::None),
      TokenType::LeftBrace => ParserRule::new(None, None, Precedence::None),
      TokenType::RightBrace => ParserRule::new(None, None, Precedence::None),
      TokenType::Comma => ParserRule::new(None, None, Precedence::None),
      TokenType::Dot => ParserRule::new(None, None, Precedence::None),
      TokenType::Minus => ParserRule::new(
        Some(Self::parse_unary),
        Some(Self::parse_binary),
        Precedence::Term,
      ),
      TokenType::Plus => ParserRule::new(None, Some(Self::parse_binary), Precedence::Term),
      TokenType::Semicolon => ParserRule::new(None, None, Precedence::None),
      TokenType::Slash => ParserRule::new(None, Some(Self::parse_binary), Precedence::Factor),
      TokenType::Star => ParserRule::new(None, Some(Self::parse_binary), Precedence::Factor),
      TokenType::Bang => ParserRule::new(Some(Self::parse_unary), None, Precedence::None),
      TokenType::Equal => ParserRule::new(None, None, Precedence::None),
      TokenType::BangEqual => ParserRule::new(None, Some(Self::parse_binary), Precedence::Equality),
      TokenType::EqualEqual => {
        ParserRule::new(None, Some(Self::parse_binary), Precedence::Equality)
      }
      TokenType::Greater => ParserRule::new(None, Some(Self::parse_binary), Precedence::Comparison),
      TokenType::GreaterEqual => {
        ParserRule::new(None, Some(Self::parse_binary), Precedence::Comparison)
      }
      TokenType::Less => ParserRule::new(None, Some(Self::parse_binary), Precedence::Comparison),
      TokenType::LessEqual => {
        ParserRule::new(None, Some(Self::parse_binary), Precedence::Comparison)
      }
      TokenType::Identifier(_) => ParserRule::new(None, None, Precedence::None),
      TokenType::StringLiteral(_) => {
        ParserRule::new(Some(Self::parse_string), None, Precedence::None)
      }
      TokenType::Number(_) => ParserRule::new(Some(Self::parse_number), None, Precedence::None),
      TokenType::And => ParserRule::new(None, None, Precedence::None),
      TokenType::Class => ParserRule::new(None, None, Precedence::None),
      TokenType::Else => ParserRule::new(None, None, Precedence::None),
      TokenType::False => ParserRule::new(Some(Self::parse_literal), None, Precedence::None),
      TokenType::For => ParserRule::new(None, None, Precedence::None),
      TokenType::Fun => ParserRule::new(None, None, Precedence::None),
      TokenType::If => ParserRule::new(None, None, Precedence::None),
      TokenType::Nil => ParserRule::new(Some(Self::parse_literal), None, Precedence::None),
      TokenType::Or => ParserRule::new(None, None, Precedence::None),
      TokenType::Print => ParserRule::new(None, None, Precedence::None),
      TokenType::Return => ParserRule::new(None, None, Precedence::None),
      TokenType::Super => ParserRule::new(None, None, Precedence::None),
      TokenType::This => ParserRule::new(None, None, Precedence::None),
      TokenType::True => ParserRule::new(Some(Self::parse_literal), None, Precedence::None),
      TokenType::Var => ParserRule::new(None, None, Precedence::None),
      TokenType::While => ParserRule::new(None, None, Precedence::None),
      TokenType::Eof => ParserRule::new(None, None, Precedence::None),
      TokenType::Invalid(_) => ParserRule::new(None, None, Precedence::None),
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
    Self { source }
  }

  /// Compile source code
  ///
  /// # Errors
  ///
  /// No errors, possible, Result is a placeholder
  pub fn compile(&mut self) -> Result<Chunk, anyhow::Error> {
    let mut chunk = Chunk::new();
    let mut scanner = Scanner::new(self.source);
    let mut parser = Parser::new(scanner.scan_tokens(), &mut chunk);

    while !parser.match_token(&TokenType::Eof) {
      parser.parse_declaration();
    }

    let parser_had_error = parser.had_error.get();

    if cfg!(debug_assertions) && !parser_had_error {
      chunk.disassemble(&mut io::stdout(), "code")?;
    }

    if parser_had_error {
      Err(anyhow!("parser had error"))
    } else {
      Ok(chunk)
    }
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
