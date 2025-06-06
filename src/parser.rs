use std::iter::Peekable;
use std::mem;

use crate::ast::{
  Array, Block, Boolean, Dangling, Expression, FunctionCall, FunctionLiteral, Identifier, If,
  Index, Infix, InfixOperator, InfixOperatorError, Integer, Let, Prefix, PrefixOperator,
  PrefixOperatorError, Program, Return, Statement, Str,
};
use crate::token::{Kind as TokenKind, Token};

#[derive(thiserror::Error, Debug, Clone)]
pub enum PrecedenceError {
  #[error("unable to convert {0} into a valid precedence")]
  UnableToConvert(u8),
}

#[repr(u8)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd)]
pub enum Precedence {
  #[default]
  Lowest = 0,
  Equals,
  LessGreater,
  Sum,
  Product,
  Prefix,
  Call,
}

impl Precedence {
  #[must_use]
  pub const fn of(kind: TokenKind) -> Self {
    match kind {
      TokenKind::Eq | TokenKind::NotEq => Self::Equals,
      TokenKind::Lt | TokenKind::Gt => Self::LessGreater,
      TokenKind::Plus | TokenKind::Minus => Self::Sum,
      TokenKind::Asterisk | TokenKind::Slash => Self::Product,
      TokenKind::LParen | TokenKind::LBracket => Self::Call,
      _ => Self::Lowest,
    }
  }

  #[must_use]
  pub fn lower(self) -> Self {
    let val = self as u8;
    let new_val = val.saturating_sub(1);
    Self::try_from(new_val).unwrap_or_else(|_| Self::default())
  }

  #[must_use]
  pub fn higher(self) -> Self {
    let val = self as u8;
    let new_val = val.saturating_add(1).min(Self::Call as u8);
    Self::try_from(new_val).unwrap_or_else(|_| Self::default())
  }
}

impl TryFrom<u8> for Precedence {
  type Error = PrecedenceError;

  fn try_from(byte: u8) -> Result<Self, Self::Error> {
    let precedence = match byte {
      0 => Self::Lowest,
      1 => Self::Equals,
      2 => Self::LessGreater,
      3 => Self::Sum,
      4 => Self::Product,
      5 => Self::Prefix,
      6 => Self::Call,
      _ => return Err(PrecedenceError::UnableToConvert(byte)),
    };
    Ok(precedence)
  }
}

#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum ParserError {
  #[error("ran out of input while parsing")]
  UnexpectedEof,

  #[error("ran into unexpected token: {0:?}")]
  UnexpectedToken(Token),

  #[error("token does not have a prefix parse function: {0}")]
  MissingPrefix(#[from] PrefixOperatorError),

  #[error("token does not have an infix parse function: {0}")]
  MissingInfix(#[from] InfixOperatorError),

  #[error("illegal integer representation: {0:?}")]
  InvalidInteger(Token),
}

#[derive(Debug, Clone)]
pub struct Parser<I>
where
  I: Iterator<Item = Token>,
{
  // this sucks, but for some reason Peekable requires concrete trait bounds
  // on definition... this seems silly, considering all the other std library
  // structures try to avoid this poor decision (see -XDatatypeContexts...)
  tokens: Peekable<I>,
  errors: Vec<ParserError>,
}

trait PrefixParse: Sized {
  fn parse<I>(parser: &mut Parser<I>) -> Result<Self, ParserError>
  where
    I: Iterator<Item = Token>;
}

trait InfixParse: Sized {
  fn parse<I>(parser: &mut Parser<I>, left: Expression) -> Result<Self, ParserError>
  where
    I: Iterator<Item = Token>;
}

impl PrefixParse for Let {
  fn parse<I>(parser: &mut Parser<I>) -> Result<Self, ParserError>
  where
    I: Iterator<Item = Token>,
  {
    let token = parser.eat(TokenKind::Let)?;
    let name = Identifier::parse(parser)?;
    let _ = parser.eat(TokenKind::Assign)?;
    let value = parser.parse_expression(Precedence::Lowest)?;
    let _ = parser.eat(TokenKind::Semicolon)?;
    Ok(Self {
      token,
      name,
      value: Box::new(value),
    })
  }
}

impl PrefixParse for Return {
  fn parse<I>(parser: &mut Parser<I>) -> Result<Self, ParserError>
  where
    I: Iterator<Item = Token>,
  {
    let token = parser.eat(TokenKind::Return)?;
    let value = if parser.peek_is(TokenKind::Semicolon) {
      None
    } else {
      Some(Box::new(parser.parse_expression(Precedence::Lowest)?))
    };
    let _ = parser.eat(TokenKind::Semicolon)?;
    Ok(Self { token, value })
  }
}

impl PrefixParse for Dangling {
  fn parse<I>(parser: &mut Parser<I>) -> Result<Self, ParserError>
  where
    I: Iterator<Item = Token>,
  {
    let token = parser.peek_eof()?.clone();
    let expr = parser.parse_expression(Precedence::Lowest)?;
    if parser.peek_is(TokenKind::Semicolon) {
      let _ = parser.eat(TokenKind::Semicolon)?;
    }
    Ok(Self {
      token,
      expr: Box::new(expr),
    })
  }
}

impl PrefixParse for Block {
  fn parse<I>(parser: &mut Parser<I>) -> Result<Self, ParserError>
  where
    I: Iterator<Item = Token>,
  {
    let token = parser.eat(TokenKind::LBrace)?;
    let mut statements = Vec::new();
    while parser.peek().is_some() && !parser.peek_is(TokenKind::RBrace) {
      statements.push(parser.parse_statement()?);
    }
    let _ = parser.eat(TokenKind::RBrace)?;
    Ok(Self { token, statements })
  }
}

impl PrefixParse for Identifier {
  fn parse<I>(parser: &mut Parser<I>) -> Result<Self, ParserError>
  where
    I: Iterator<Item = Token>,
  {
    let token = parser.eat(TokenKind::Ident)?;
    Ok(Self {
      value: token.literal.clone(),
      token,
    })
  }
}

impl PrefixParse for Integer {
  fn parse<I>(parser: &mut Parser<I>) -> Result<Self, ParserError>
  where
    I: Iterator<Item = Token>,
  {
    let token = parser.eat(TokenKind::Int)?;
    let value = token
      .literal
      .parse::<i64>()
      .map_err(|_| ParserError::InvalidInteger(token.clone()))?;
    Ok(Self { token, value })
  }
}

impl PrefixParse for Boolean {
  fn parse<I>(parser: &mut Parser<I>) -> Result<Self, ParserError>
  where
    I: Iterator<Item = Token>,
  {
    let token = parser.next_eof()?;
    let value = token.kind == TokenKind::True;
    Ok(Self { token, value })
  }
}

impl PrefixParse for Str {
  fn parse<I>(parser: &mut Parser<I>) -> Result<Self, ParserError>
  where
    I: Iterator<Item = Token>,
  {
    let token = parser.eat(TokenKind::String)?;
    let value = token.literal.clone();
    Ok(Self { token, value })
  }
}

impl PrefixParse for Array {
  fn parse<I>(parser: &mut Parser<I>) -> Result<Self, ParserError>
  where
    I: Iterator<Item = Token>,
  {
    let token = parser.eat(TokenKind::LBracket)?;
    let elements = parser.parse_comma_separated(TokenKind::RBracket, |parser| {
      // parse each element as an expression with default lowest precedence
      parser.parse_expression(Precedence::default())
    })?;
    let _ = parser.eat(TokenKind::RBracket)?;
    Ok(Self { token, elements })
  }
}

impl PrefixParse for Prefix {
  fn parse<I>(parser: &mut Parser<I>) -> Result<Self, ParserError>
  where
    I: Iterator<Item = Token>,
  {
    let token = parser.next_eof()?;
    let operator = PrefixOperator::try_from(token.kind)?;
    let right = parser.parse_expression(Precedence::Prefix)?;
    Ok(Self {
      token,
      operator,
      right: Box::new(right),
    })
  }
}

impl PrefixParse for If {
  fn parse<I>(parser: &mut Parser<I>) -> Result<Self, ParserError>
  where
    I: Iterator<Item = Token>,
  {
    let token = parser.eat(TokenKind::If)?;
    let _ = parser.eat(TokenKind::LParen)?;
    let antecedent = parser.parse_expression(Precedence::default())?;
    let _ = parser.eat(TokenKind::RParen)?;
    let consequent = Block::parse(parser)?;
    let alternative = if parser.peek_is(TokenKind::Else) {
      let _ = parser.eat(TokenKind::Else)?;
      Some(Block::parse(parser)?)
    } else {
      None
    };
    Ok(Self {
      token,
      antecedent: Box::new(antecedent),
      consequent,
      alternative,
    })
  }
}

impl PrefixParse for FunctionLiteral {
  fn parse<I>(parser: &mut Parser<I>) -> Result<Self, ParserError>
  where
    I: Iterator<Item = Token>,
  {
    let token = parser.eat(TokenKind::Function)?;
    let _ = parser.eat(TokenKind::LParen)?;
    let parameters = parser.parse_comma_separated(TokenKind::RParen, Identifier::parse)?;
    let _ = parser.eat(TokenKind::RParen)?;
    let body = Block::parse(parser)?;
    Ok(Self {
      token,
      parameters,
      body,
    })
  }
}

impl InfixParse for FunctionCall {
  fn parse<I>(parser: &mut Parser<I>, function: Expression) -> Result<Self, ParserError>
  where
    I: Iterator<Item = Token>,
  {
    let token = parser.eat(TokenKind::LParen)?;
    let arguments = parser.parse_comma_separated(TokenKind::RParen, |parser| {
      // parse each argument as an expression with default lowest precedence
      parser.parse_expression(Precedence::default())
    })?;
    let _ = parser.eat(TokenKind::RParen)?;
    Ok(Self {
      token,
      function: Box::new(function),
      arguments,
    })
  }
}

impl InfixParse for Index {
  fn parse<I>(parser: &mut Parser<I>, left: Expression) -> Result<Self, ParserError>
  where
    I: Iterator<Item = Token>,
  {
    let token = parser.eat(TokenKind::LBracket)?;
    let index = parser.parse_expression(Precedence::default())?;
    let _ = parser.eat(TokenKind::RBracket)?;
    Ok(Self {
      token,
      left: Box::new(left),
      index: Box::new(index),
    })
  }
}

impl InfixParse for Infix {
  fn parse<I>(parser: &mut Parser<I>, left: Expression) -> Result<Self, ParserError>
  where
    I: Iterator<Item = Token>,
  {
    let token = parser.next_eof()?;
    let operator = InfixOperator::try_from(token.kind)?;
    let precedence = Precedence::of(token.kind);
    let precedence = match token.kind {
      TokenKind::Plus => precedence.lower(),
      _ => precedence,
    };
    let right = parser.parse_expression(precedence)?;
    Ok(Self {
      token,
      operator,
      left: Box::new(left),
      right: Box::new(right),
    })
  }
}

// parser implements general functions to consume a stream of tokens...
impl<I> Parser<I>
where
  I: Iterator<Item = Token>,
{
  pub fn new(tokens: I) -> Self {
    Self {
      tokens: tokens.peekable(),
      errors: Vec::new(),
    }
  }

  /// Parses a stream of tokens into a `Program` AST node
  ///
  /// # Errors
  ///
  /// Returns a list of `ParserError`s if parsing fails due to unexpected tokens,
  /// missing expressions, or malformed syntax...
  pub fn parse(mut self) -> Result<Program, Vec<ParserError>> {
    match self.parse_program() {
      Some(program) => Ok(program),
      None => Err(mem::take(&mut self.errors)),
    }
  }

  fn next(&mut self) -> Option<Token> {
    self.tokens.next()
  }

  fn next_eof(&mut self) -> Result<Token, ParserError> {
    self.next().ok_or(ParserError::UnexpectedEof)
  }

  fn peek(&mut self) -> Option<&Token> {
    self.tokens.peek()
  }

  fn peek_eof(&mut self) -> Result<&Token, ParserError> {
    self.peek().ok_or(ParserError::UnexpectedEof)
  }

  fn peek_is(&mut self, kind: TokenKind) -> bool {
    matches!(self.peek(), Some(t) if t.kind == kind)
  }

  fn eat(&mut self, expected: TokenKind) -> Result<Token, ParserError> {
    let token = self.next_eof()?;
    if token.kind == expected {
      Ok(token)
    } else {
      Err(ParserError::UnexpectedToken(token))
    }
  }

  fn eof(&mut self) -> Result<(), ParserError> {
    self
      .next()
      .map_or(Ok(()), |token| Err(ParserError::UnexpectedToken(token)))
  }

  fn record<T>(&mut self, result: Result<T, ParserError>) -> Option<T> {
    match result {
      Ok(t) => Some(t),
      Err(e) => {
        self.errors.push(e);
        None
      }
    }
  }

  fn synchronize(&mut self) {
    while let Some(token) = self.peek() {
      if token.kind == TokenKind::Semicolon {
        self.next();
        return;
      }
      self.next();
    }
  }

  fn parse_program(&mut self) -> Option<Program> {
    let mut statements = Vec::new();
    // while we have stuff and that stuff isnt an end of file...
    while self.peek().is_some() && !self.peek_is(TokenKind::Eof) {
      let statement = self.parse_statement();
      match self.record(statement) {
        Some(stmt) => statements.push(stmt),
        None => self.synchronize(),
      }
    }

    // eat the end of file; no more tokens left!
    let eof_result = self.eof();
    let _ = self.record(eof_result);

    if self.errors.is_empty() {
      Some(Program { statements })
    } else {
      None
    }
  }

  fn parse_statement(&mut self) -> Result<Statement, ParserError> {
    #[rustfmt::skip]
    let stmt = match self.peek() {
      Some(Token { kind: TokenKind::Let, .. }) =>
        Let::parse(self).map(Into::into),
      Some(Token { kind: TokenKind::Return, .. }) =>
        Return::parse(self).map(Into::into),
      Some(Token { kind: TokenKind::LBrace, .. }) =>
        Block::parse(self).map(Into::into),
      Some(_) =>
        Dangling::parse(self).map(Into::into),
      None => Err(ParserError::UnexpectedEof),
    };
    stmt
  }

  fn parse_expression(&mut self, precedence: Precedence) -> Result<Expression, ParserError> {
    let left = parse_expression_prefix(self)?;
    parse_expression_infix(self, left, precedence)
  }

  fn parse_grouped(&mut self) -> Result<Expression, ParserError> {
    let _ = self.eat(TokenKind::LParen)?;
    let expr = self.parse_expression(Precedence::default())?;
    let _ = self.eat(TokenKind::RParen)?;
    Ok(expr)
  }

  fn parse_comma_separated<T>(
    &mut self,
    terminator: TokenKind,
    mut parse_element: impl FnMut(&mut Self) -> Result<T, ParserError>,
  ) -> Result<Vec<T>, ParserError> {
    let mut elements = Vec::new();
    // is the list empty?
    if self.peek_is(terminator) {
      return Ok(elements);
    }
    // add elements while we have yet to reach infront of the terminator...
    loop {
      elements.push(parse_element(self)?);
      let peek = self.peek();
      match peek {
        Some(Token { kind, .. }) if *kind == terminator => break,
        Some(Token {
          kind: TokenKind::Comma,
          ..
        }) => {
          let _ = self.eat(TokenKind::Comma)?;
          // handle trailing comma
          if self.peek_is(terminator) {
            break;
          }
        }
        _ => return Err(ParserError::UnexpectedToken(self.peek_eof()?.clone())),
      }
    }
    Ok(elements)
  }
}

// map token kinds to prefix/infix parse functions

fn parse_expression_prefix<I>(parser: &mut Parser<I>) -> Result<Expression, ParserError>
where
  I: Iterator<Item = Token>,
{
  let token = parser.peek_eof()?;
  match token.kind {
    TokenKind::Ident => Identifier::parse(parser).map(Into::into),
    TokenKind::Int => Integer::parse(parser).map(Into::into),
    TokenKind::String => Str::parse(parser).map(Into::into),
    TokenKind::LBracket => Array::parse(parser).map(Into::into),
    TokenKind::True | TokenKind::False => Boolean::parse(parser).map(Into::into),
    TokenKind::Bang | TokenKind::Minus => Prefix::parse(parser).map(Into::into),
    TokenKind::If => If::parse(parser).map(Into::into),
    TokenKind::Function => FunctionLiteral::parse(parser).map(Into::into),
    TokenKind::LParen => parser.parse_grouped(),
    _ => Err(PrefixOperatorError::TokenNotPrefix(token.kind).into()),
  }
}

fn parse_expression_infix<I>(
  parser: &mut Parser<I>,
  mut left: Expression,
  precedence: Precedence,
) -> Result<Expression, ParserError>
where
  I: Iterator<Item = Token>,
{
  while let Some(next_token) = parser.peek() {
    let next_precedence = Precedence::of(next_token.kind);
    if next_precedence <= precedence {
      break;
    }
    left = match next_token.kind {
      TokenKind::LParen => FunctionCall::parse(parser, left).map(Into::into)?,
      TokenKind::LBracket => Index::parse(parser, left).map(Into::into)?,
      TokenKind::Plus
      | TokenKind::Minus
      | TokenKind::Asterisk
      | TokenKind::Slash
      | TokenKind::Eq
      | TokenKind::NotEq
      | TokenKind::Lt
      | TokenKind::Gt => Infix::parse(parser, left).map(Into::into)?,
      _ => break,
    };
  }
  Ok(left)
}

#[cfg(test)]
mod tests {
  use super::*;
  use rstest::*;

  fn token(kind: TokenKind, literal: &str) -> Token {
    Token {
      kind,
      literal: literal.to_string(),
    }
  }

  mod parser {
    use super::*;

    mod fixtures {
      use super::*;

      #[fixture]
      pub fn boolean_true() -> (Vec<Token>, Program) {
        let input = vec![
          token(TokenKind::True, "true"),
          token(TokenKind::Semicolon, ";"),
        ];
        let expected = Program {
          statements: vec![Statement::Expr(Dangling {
            token: token(TokenKind::True, "true"),
            expr: Box::new(Expression::Boolean(Boolean {
              token: token(TokenKind::True, "true"),
              value: true,
            })),
          })],
        };
        (input, expected)
      }

      #[fixture]
      pub fn boolean_false() -> (Vec<Token>, Program) {
        let input = vec![
          token(TokenKind::False, "false"),
          token(TokenKind::Semicolon, ";"),
        ];
        let expected = Program {
          statements: vec![Statement::Expr(Dangling {
            token: token(TokenKind::False, "false"),
            expr: Box::new(Expression::Boolean(Boolean {
              token: token(TokenKind::False, "false"),
              value: false,
            })),
          })],
        };
        (input, expected)
      }

      #[fixture]
      pub fn grouped_simple() -> (Vec<Token>, Program) {
        let input = vec![
          token(TokenKind::LParen, "("),
          token(TokenKind::Int, "5"),
          token(TokenKind::RParen, ")"),
          token(TokenKind::Semicolon, ";"),
        ];
        let expected = Program {
          statements: vec![Statement::Expr(Dangling {
            token: token(TokenKind::LParen, "("),
            expr: Box::new(Expression::Integer(Integer {
              token: token(TokenKind::Int, "5"),
              value: 5,
            })),
          })],
        };
        (input, expected)
      }

      #[fixture]
      pub fn grouped_complex() -> (Vec<Token>, Program) {
        let input = vec![
          token(TokenKind::LParen, "("),
          token(TokenKind::Int, "1"),
          token(TokenKind::Plus, "+"),
          token(TokenKind::Int, "2"),
          token(TokenKind::RParen, ")"),
          token(TokenKind::Asterisk, "*"),
          token(TokenKind::Int, "3"),
          token(TokenKind::Semicolon, ";"),
        ];
        let expected = Program {
          statements: vec![Statement::Expr(Dangling {
            token: token(TokenKind::LParen, "("),
            expr: Box::new(Expression::Infix(Infix {
              token: token(TokenKind::Asterisk, "*"),
              operator: InfixOperator::Mul,
              left: Box::new(Expression::Infix(Infix {
                token: token(TokenKind::Plus, "+"),
                operator: InfixOperator::Add,
                left: Box::new(Expression::Integer(Integer {
                  token: token(TokenKind::Int, "1"),
                  value: 1,
                })),
                right: Box::new(Expression::Integer(Integer {
                  token: token(TokenKind::Int, "2"),
                  value: 2,
                })),
              })),
              right: Box::new(Expression::Integer(Integer {
                token: token(TokenKind::Int, "3"),
                value: 3,
              })),
            })),
          })],
        };
        (input, expected)
      }

      #[fixture]
      pub fn precedence_add_mul() -> (Vec<Token>, Program) {
        let input = vec![
          token(TokenKind::Ident, "a"),
          token(TokenKind::Plus, "+"),
          token(TokenKind::Ident, "b"),
          token(TokenKind::Asterisk, "*"),
          token(TokenKind::Ident, "c"),
          token(TokenKind::Semicolon, ";"),
        ];
        let expected = Program {
          statements: vec![Statement::Expr(Dangling {
            token: token(TokenKind::Ident, "a"),
            expr: Box::new(Expression::Infix(Infix {
              token: token(TokenKind::Plus, "+"),
              operator: InfixOperator::Add,
              left: Box::new(Expression::Identifier(Identifier {
                token: token(TokenKind::Ident, "a"),
                value: "a".to_string(),
              })),
              right: Box::new(Expression::Infix(Infix {
                token: token(TokenKind::Asterisk, "*"),
                operator: InfixOperator::Mul,
                left: Box::new(Expression::Identifier(Identifier {
                  token: token(TokenKind::Ident, "b"),
                  value: "b".to_string(),
                })),
                right: Box::new(Expression::Identifier(Identifier {
                  token: token(TokenKind::Ident, "c"),
                  value: "c".to_string(),
                })),
              })),
            })),
          })],
        };
        (input, expected)
      }

      #[fixture]
      pub fn let_statement() -> (Vec<Token>, Program) {
        let input = vec![
          token(TokenKind::Let, "let"),
          token(TokenKind::Ident, "x"),
          token(TokenKind::Assign, "="),
          token(TokenKind::Int, "5"),
          token(TokenKind::Semicolon, ";"),
        ];
        let expected = Program {
          statements: vec![Statement::Let(Let {
            token: token(TokenKind::Let, "let"),
            name: Identifier {
              token: token(TokenKind::Ident, "x"),
              value: "x".to_string(),
            },
            value: Box::new(Expression::Integer(Integer {
              token: token(TokenKind::Int, "5"),
              value: 5,
            })),
          })],
        };
        (input, expected)
      }

      #[fixture]
      pub fn return_statement() -> (Vec<Token>, Program) {
        let input = vec![
          token(TokenKind::Return, "return"),
          token(TokenKind::Int, "10"),
          token(TokenKind::Semicolon, ";"),
        ];
        let expected = Program {
          statements: vec![Statement::Return(Return {
            token: token(TokenKind::Return, "return"),
            value: Some(Box::new(Expression::Integer(Integer {
              token: token(TokenKind::Int, "10"),
              value: 10,
            }))),
          })],
        };
        (input, expected)
      }

      #[fixture]
      pub fn if_without_else() -> (Vec<Token>, Program) {
        let input = vec![
          token(TokenKind::If, "if"),
          token(TokenKind::LParen, "("),
          token(TokenKind::Ident, "x"),
          token(TokenKind::Lt, "<"),
          token(TokenKind::Ident, "y"),
          token(TokenKind::RParen, ")"),
          token(TokenKind::LBrace, "{"),
          token(TokenKind::Ident, "x"),
          token(TokenKind::Semicolon, ";"),
          token(TokenKind::RBrace, "}"),
        ];
        let expected = Program {
          statements: vec![Statement::Expr(Dangling {
            token: token(TokenKind::If, "if"),
            expr: Box::new(Expression::If(If {
              token: token(TokenKind::If, "if"),
              antecedent: Box::new(Expression::Infix(Infix {
                token: token(TokenKind::Lt, "<"),
                operator: InfixOperator::Lt,
                left: Box::new(Expression::Identifier(Identifier {
                  token: token(TokenKind::Ident, "x"),
                  value: "x".to_string(),
                })),
                right: Box::new(Expression::Identifier(Identifier {
                  token: token(TokenKind::Ident, "y"),
                  value: "y".to_string(),
                })),
              })),
              consequent: Block {
                token: token(TokenKind::LBrace, "{"),
                statements: vec![Statement::Expr(Dangling {
                  token: token(TokenKind::Ident, "x"),
                  expr: Box::new(Expression::Identifier(Identifier {
                    token: token(TokenKind::Ident, "x"),
                    value: "x".to_string(),
                  })),
                })],
              },
              alternative: None,
            })),
          })],
        };
        (input, expected)
      }

      #[fixture]
      pub fn function_literal_no_params() -> (Vec<Token>, Program) {
        let input = vec![
          token(TokenKind::Function, "fn"),
          token(TokenKind::LParen, "("),
          token(TokenKind::RParen, ")"),
          token(TokenKind::LBrace, "{"),
          token(TokenKind::Ident, "x"),
          token(TokenKind::Semicolon, ";"),
          token(TokenKind::RBrace, "}"),
        ];
        let expected = Program {
          statements: vec![Statement::Expr(Dangling {
            token: token(TokenKind::Function, "fn"),
            expr: Box::new(Expression::FunctionLiteral(FunctionLiteral {
              token: token(TokenKind::Function, "fn"),
              parameters: vec![],
              body: Block {
                token: token(TokenKind::LBrace, "{"),
                statements: vec![Statement::Expr(Dangling {
                  token: token(TokenKind::Ident, "x"),
                  expr: Box::new(Expression::Identifier(Identifier {
                    token: token(TokenKind::Ident, "x"),
                    value: "x".to_string(),
                  })),
                })],
              },
            })),
          })],
        };
        (input, expected)
      }

      #[fixture]
      pub fn function_call_no_args() -> (Vec<Token>, Program) {
        let input = vec![
          token(TokenKind::Ident, "add"),
          token(TokenKind::LParen, "("),
          token(TokenKind::RParen, ")"),
          token(TokenKind::Semicolon, ";"),
        ];
        let expected = Program {
          statements: vec![Statement::Expr(Dangling {
            token: token(TokenKind::Ident, "add"),
            expr: Box::new(Expression::FunctionCall(FunctionCall {
              token: token(TokenKind::LParen, "("),
              function: Box::new(Expression::Identifier(Identifier {
                token: token(TokenKind::Ident, "add"),
                value: "add".to_string(),
              })),
              arguments: vec![],
            })),
          })],
        };
        (input, expected)
      }
    }

    #[rstest]
    #[case::booleans_true(fixtures::boolean_true())]
    #[case::booleans_false(fixtures::boolean_false())]
    #[case::grouping_simple(fixtures::grouped_simple())]
    #[case::grouping_complex(fixtures::grouped_complex())]
    #[case::precedence(fixtures::precedence_add_mul())]
    #[case::let_stmt(fixtures::let_statement())]
    #[case::return_stmt(fixtures::return_statement())]
    #[case::if_expr(fixtures::if_without_else())]
    #[case::function_literal(fixtures::function_literal_no_params())]
    #[case::function_call(fixtures::function_call_no_args())]
    fn parse_program(#[case] input: (Vec<Token>, Program)) {
      let (tokens, expected) = input;
      let parser = Parser::new(tokens.into_iter());
      let program = parser.parse().unwrap();
      assert_eq!(program, expected);
    }

    #[rstest]
    #[case(TokenKind::Plus, InfixOperator::Add, "+", 5, 5)]
    #[case(TokenKind::Minus, InfixOperator::Sub, "-", 10, 3)]
    #[case(TokenKind::Asterisk, InfixOperator::Mul, "*", 4, 6)]
    #[case(TokenKind::Slash, InfixOperator::Div, "/", 12, 3)]
    #[case(TokenKind::Eq, InfixOperator::Eq, "==", 7, 7)]
    #[case(TokenKind::NotEq, InfixOperator::Neq, "!=", 8, 9)]
    #[case(TokenKind::Lt, InfixOperator::Lt, "<", 3, 5)]
    #[case(TokenKind::Gt, InfixOperator::Gt, ">", 10, 2)]
    fn infix_operators(
      #[case] token_kind: TokenKind,
      #[case] op: InfixOperator,
      #[case] op_literal: &str,
      #[case] left_val: i64,
      #[case] right_val: i64,
    ) {
      let input = vec![
        token(TokenKind::Int, &left_val.to_string()),
        token(token_kind, op_literal),
        token(TokenKind::Int, &right_val.to_string()),
        token(TokenKind::Semicolon, ";"),
      ];
      let parser = Parser::new(input.into_iter());
      let program = parser.parse().unwrap();

      let expected = Program {
        statements: vec![Statement::Expr(Dangling {
          token: token(TokenKind::Int, &left_val.to_string()),
          expr: Box::new(Expression::Infix(Infix {
            token: token(token_kind, op_literal),
            operator: op,
            left: Box::new(Expression::Integer(Integer {
              token: token(TokenKind::Int, &left_val.to_string()),
              value: left_val,
            })),
            right: Box::new(Expression::Integer(Integer {
              token: token(TokenKind::Int, &right_val.to_string()),
              value: right_val,
            })),
          })),
        })],
      };

      assert_eq!(program, expected);
    }

    #[rstest]
    #[case(
      vec![
        token(TokenKind::Let, "let"),
        token(TokenKind::Int, "5"),
        token(TokenKind::Assign, "="),
        token(TokenKind::Int, "10"),
        token(TokenKind::Semicolon, ";"),
      ],
      vec![ParserError::UnexpectedToken(token(TokenKind::Int, "5"))]
    )]
    #[case(
      vec![
        token(TokenKind::Let, "let"),
        token(TokenKind::Ident, "x"),
        token(TokenKind::Int, "5"),
        token(TokenKind::Semicolon, ";"),
      ],
      vec![ParserError::UnexpectedToken(token(TokenKind::Int, "5"))]
    )]
    fn parsing_errors(#[case] tokens: Vec<Token>, #[case] expected_errors: Vec<ParserError>) {
      let parser = Parser::new(tokens.into_iter());
      let result = parser.parse();
      assert!(result.is_err());
      if let Err(errors) = result {
        assert_eq!(errors, expected_errors);
      }
    }
  }
}
