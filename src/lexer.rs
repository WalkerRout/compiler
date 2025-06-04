use std::iter::Peekable;
use std::str::Chars;

use crate::token::{lookup_identifier, Kind as TokenKind, Token};

#[derive(Debug, Clone)]
pub struct Lexer<'src> {
  peekable: Peekable<Chars<'src>>,
}

impl<'src> Lexer<'src> {
  #[must_use]
  pub fn new(source: &'src str) -> Self {
    Self {
      peekable: source.chars().peekable(),
    }
  }

  fn next_token(&mut self) -> Option<Token> {
    self.skip_whitespace();
    let ch = self.peek_char()?;
    match ch {
      '=' => self.eat_next_as_or('=', TokenKind::Eq, TokenKind::Assign),
      '!' => self.eat_next_as_or('=', TokenKind::NotEq, TokenKind::Bang),
      '+' => self.eat(TokenKind::Plus),
      '-' => self.eat(TokenKind::Minus),
      '*' => self.eat(TokenKind::Asterisk),
      '/' => self.eat(TokenKind::Slash),
      '<' => self.eat(TokenKind::Lt),
      '>' => self.eat(TokenKind::Gt),
      ';' => self.eat(TokenKind::Semicolon),
      ':' => self.eat(TokenKind::Colon),
      ',' => self.eat(TokenKind::Comma),
      '.' => self.eat(TokenKind::Dot),
      '{' => self.eat(TokenKind::LBrace),
      '}' => self.eat(TokenKind::RBrace),
      '(' => self.eat(TokenKind::LParen),
      ')' => self.eat(TokenKind::RParen),
      '[' => self.eat(TokenKind::LBracket),
      ']' => self.eat(TokenKind::RBracket),
      '"' => self.eat_string(),
      '0'..='9' => self.eat_number(), // important this comes before eat_word
      c if c.is_alphabetic() || c == '_' => self.eat_word(),
      _ => self.eat(TokenKind::Unknown),
    }
  }

  fn eat(&mut self, kind: TokenKind) -> Option<Token> {
    let c = self.peekable.next()?;
    Some(Token::new(kind, c))
  }

  fn eat_next_as_or(&mut self, next: char, r#as: TokenKind, or: TokenKind) -> Option<Token> {
    let mut kind = or;
    let mut buffer = String::new();
    buffer.push(self.peekable.next()?);
    if let Some(&peek) = self.peekable.peek() {
      if next == peek {
        buffer.push(peek);
        kind = r#as;
        self.peekable.next(); // advance, since we already took peek
      }
    }
    Some(Token::new(kind, buffer))
  }

  fn eat_string(&mut self) -> Option<Token> {
    let mut buffer = String::new();
    self.peekable.next(); // eat '"' char
    for ch in self.peekable.by_ref() {
      if ch == '"' {
        return Some(Token::new(TokenKind::String, buffer));
      }
      buffer.push(ch);
    }
    None
  }

  fn eat_number(&mut self) -> Option<Token> {
    let mut buffer = String::new();
    while self.peek_char().is_some_and(|c| c.is_ascii_digit()) {
      buffer.push(self.peekable.next()?);
    }
    Some(Token::new(TokenKind::Int, buffer))
  }

  fn eat_word(&mut self) -> Option<Token> {
    let mut buffer = String::new();
    while self.peek_char().is_some_and(char::is_alphanumeric) {
      buffer.push(self.peekable.next()?);
    }
    let kind = lookup_identifier(buffer.as_str());
    Some(Token::new(kind, buffer))
  }

  fn peek_char(&mut self) -> Option<char> {
    self.peekable.peek().copied()
  }

  fn skip_whitespace(&mut self) {
    while self.peek_char().is_some_and(char::is_whitespace) {
      self.peekable.next();
    }
  }
}

impl Iterator for Lexer<'_> {
  type Item = Token;

  fn next(&mut self) -> Option<Self::Item> {
    self.next_token()
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use rstest::*;

  mod lexer {
    use super::*;

    mod fixtures {
      use super::*;

      #[fixture]
      pub fn invalid_characters() -> (&'static str, Vec<Token>) {
        (
          r"let a = f @ j | k;",
          vec![
            Token::new(TokenKind::Let, "let"),
            Token::new(TokenKind::Ident, "a"),
            Token::new(TokenKind::Assign, "="),
            Token::new(TokenKind::Ident, "f"),
            Token::new(TokenKind::Unknown, "@"),
            Token::new(TokenKind::Ident, "j"),
            Token::new(TokenKind::Unknown, "|"),
            Token::new(TokenKind::Ident, "k"),
            Token::new(TokenKind::Semicolon, ";"),
          ],
        )
      }

      #[fixture]
      pub fn variable_bindings() -> (&'static str, Vec<Token>) {
        (
          r"let five = 5;
let ten = 10;",
          vec![
            Token::new(TokenKind::Let, "let"),
            Token::new(TokenKind::Ident, "five"),
            Token::new(TokenKind::Assign, "="),
            Token::new(TokenKind::Int, "5"),
            Token::new(TokenKind::Semicolon, ";"),
            Token::new(TokenKind::Let, "let"),
            Token::new(TokenKind::Ident, "ten"),
            Token::new(TokenKind::Assign, "="),
            Token::new(TokenKind::Int, "10"),
            Token::new(TokenKind::Semicolon, ";"),
          ],
        )
      }

      #[fixture]
      pub fn function_binding() -> (&'static str, Vec<Token>) {
        (
          r"let add = fn(x, y) {
  x + y;
};",
          vec![
            Token::new(TokenKind::Let, "let"),
            Token::new(TokenKind::Ident, "add"),
            Token::new(TokenKind::Assign, "="),
            Token::new(TokenKind::Function, "fn"),
            Token::new(TokenKind::LParen, "("),
            Token::new(TokenKind::Ident, "x"),
            Token::new(TokenKind::Comma, ","),
            Token::new(TokenKind::Ident, "y"),
            Token::new(TokenKind::RParen, ")"),
            Token::new(TokenKind::LBrace, "{"),
            Token::new(TokenKind::Ident, "x"),
            Token::new(TokenKind::Plus, "+"),
            Token::new(TokenKind::Ident, "y"),
            Token::new(TokenKind::Semicolon, ";"),
            Token::new(TokenKind::RBrace, "}"),
            Token::new(TokenKind::Semicolon, ";"),
          ],
        )
      }

      #[fixture]
      pub fn function_call_and_operators() -> (&'static str, Vec<Token>) {
        (
          r"let result = add(five, ten);
!-/*5;
5 < 10 > 5;",
          vec![
            Token::new(TokenKind::Let, "let"),
            Token::new(TokenKind::Ident, "result"),
            Token::new(TokenKind::Assign, "="),
            Token::new(TokenKind::Ident, "add"),
            Token::new(TokenKind::LParen, "("),
            Token::new(TokenKind::Ident, "five"),
            Token::new(TokenKind::Comma, ","),
            Token::new(TokenKind::Ident, "ten"),
            Token::new(TokenKind::RParen, ")"),
            Token::new(TokenKind::Semicolon, ";"),
            Token::new(TokenKind::Bang, "!"),
            Token::new(TokenKind::Minus, "-"),
            Token::new(TokenKind::Slash, "/"),
            Token::new(TokenKind::Asterisk, "*"),
            Token::new(TokenKind::Int, "5"),
            Token::new(TokenKind::Semicolon, ";"),
            Token::new(TokenKind::Int, "5"),
            Token::new(TokenKind::Lt, "<"),
            Token::new(TokenKind::Int, "10"),
            Token::new(TokenKind::Gt, ">"),
            Token::new(TokenKind::Int, "5"),
            Token::new(TokenKind::Semicolon, ";"),
          ],
        )
      }

      #[fixture]
      pub fn if_statement() -> (&'static str, Vec<Token>) {
        (
          r"if (5 < 10) {
  return true;
} else {
  return false;
}",
          vec![
            Token::new(TokenKind::If, "if"),
            Token::new(TokenKind::LParen, "("),
            Token::new(TokenKind::Int, "5"),
            Token::new(TokenKind::Lt, "<"),
            Token::new(TokenKind::Int, "10"),
            Token::new(TokenKind::RParen, ")"),
            Token::new(TokenKind::LBrace, "{"),
            Token::new(TokenKind::Return, "return"),
            Token::new(TokenKind::True, "true"),
            Token::new(TokenKind::Semicolon, ";"),
            Token::new(TokenKind::RBrace, "}"),
            Token::new(TokenKind::Else, "else"),
            Token::new(TokenKind::LBrace, "{"),
            Token::new(TokenKind::Return, "return"),
            Token::new(TokenKind::False, "false"),
            Token::new(TokenKind::Semicolon, ";"),
            Token::new(TokenKind::RBrace, "}"),
          ],
        )
      }

      #[fixture]
      pub fn equality_and_data_structures() -> (&'static str, Vec<Token>) {
        (
          r#"10 == 10;
10 != 9;
"foobar"
"foo bar"
[1, 2];
{"foo": "bar"}"#,
          vec![
            Token::new(TokenKind::Int, "10"),
            Token::new(TokenKind::Eq, "=="),
            Token::new(TokenKind::Int, "10"),
            Token::new(TokenKind::Semicolon, ";"),
            Token::new(TokenKind::Int, "10"),
            Token::new(TokenKind::NotEq, "!="),
            Token::new(TokenKind::Int, "9"),
            Token::new(TokenKind::Semicolon, ";"),
            Token::new(TokenKind::String, "foobar"),
            Token::new(TokenKind::String, "foo bar"),
            Token::new(TokenKind::LBracket, "["),
            Token::new(TokenKind::Int, "1"),
            Token::new(TokenKind::Comma, ","),
            Token::new(TokenKind::Int, "2"),
            Token::new(TokenKind::RBracket, "]"),
            Token::new(TokenKind::Semicolon, ";"),
            Token::new(TokenKind::LBrace, "{"),
            Token::new(TokenKind::String, "foo"),
            Token::new(TokenKind::Colon, ":"),
            Token::new(TokenKind::String, "bar"),
            Token::new(TokenKind::RBrace, "}"),
          ],
        )
      }
    }

    #[rstest]
    #[case::mistakes(fixtures::invalid_characters())]
    #[case::variables(fixtures::variable_bindings())]
    #[case::functions(fixtures::function_binding())]
    #[case::operators(fixtures::function_call_and_operators())]
    #[case::control_flow(fixtures::if_statement())]
    #[case::data(fixtures::equality_and_data_structures())]
    fn next_token(#[case] input: (&'static str, Vec<Token>)) {
      let (input, expected_tokens) = input;
      let mut lexer = Lexer::new(input);

      for expected in expected_tokens {
        let actual = lexer.next_token().unwrap();
        dbg!(&actual);
        assert_eq!(actual.kind, expected.kind, "TokenKind mismatch");
        assert_eq!(actual.literal, expected.literal, "Token literal mismatch");
      }
    }
  }
}
