use std::fmt;

use crate::token::{Kind as TokenKind, Token};

/// A visitor trait for traversing and handling nodes of an abstract syntax tree
///
/// Implementors of this trait define what to do with each kind of node
pub trait Visitor<T> {
  /// The error type returned by visitor methods...
  type Error;

  // program

  /// Visit a complete program node
  ///
  /// # Errors
  ///
  /// Returns an error if visiting the program node fails for any reason,
  /// such as unsupported syntax or semantic validation failure
  fn visit_program(&mut self, program: &Program) -> Result<T, Self::Error>;

  // statement nodes

  /// Visit a single statement node
  ///
  /// # Errors
  ///
  /// Returns an error if the statement is malformed or cannot be processed
  fn visit_statement(&mut self, stmt: &Statement) -> Result<T, Self::Error>;

  /// Visit a `let` statement node
  ///
  /// # Errors
  ///
  /// Returns an error if the `let` binding is invalid or unsupported
  fn visit_let(&mut self, let_stmt: &Let) -> Result<T, Self::Error>;

  /// Visit a `return` statement node
  ///
  /// # Errors
  ///
  /// Returns an error if the `return` statement contains invalid expressions
  /// or violates context-specific rules
  fn visit_return(&mut self, return_stmt: &Return) -> Result<T, Self::Error>;

  /// Visit a dangling expression statement node
  ///
  /// # Errors
  ///
  /// Returns an error if the dangling expression cannot be evaluated or is
  /// invalid in context
  fn visit_dangling(&mut self, dangling: &Dangling) -> Result<T, Self::Error>;

  /// Visit a block node (ex. `{ ... }`)
  ///
  /// # Errors
  ///
  /// Returns an error if any statement within the block fails to process
  fn visit_block(&mut self, block: &Block) -> Result<T, Self::Error>;

  // expression nodes

  /// Visit a generic expression node
  ///
  /// # Errors
  ///
  /// Returns an error if the expression type is unknown or unhandled
  fn visit_expression(&mut self, expr: &Expression) -> Result<T, Self::Error>;

  /// Visit an identifier node
  ///
  /// # Errors
  ///
  /// Returns an error if the identifier is undefined or invalid in its usage
  /// context
  fn visit_identifier(&mut self, ident: &Identifier) -> Result<T, Self::Error>;

  /// Visit an integer literal node
  ///
  /// # Errors
  ///
  /// Returns an error if the integer literal is outside valid bounds or
  /// otherwise malformed
  fn visit_integer(&mut self, int: &Integer) -> Result<T, Self::Error>;

  /// Visit a boolean literal node
  ///
  /// # Errors
  ///
  /// Returns an error if the boolean value cannot be used in its context
  fn visit_boolean(&mut self, bool: &Boolean) -> Result<T, Self::Error>;

  /// Visit a string literal node
  ///
  /// # Errors
  ///
  /// Returns an error if the string cannot be processed
  fn visit_string(&mut self, string: &Str) -> Result<T, Self::Error>;

  /// Visit an array literal node
  ///
  /// # Errors
  ///
  /// Returns an error if the array is malformed
  fn visit_array(&mut self, array: &Array) -> Result<T, Self::Error>;

  /// Visit an index expression such as list[0]
  ///
  /// # Errors
  ///
  /// Returns an error if the index access is malformed
  fn visit_index(&mut self, index: &Index) -> Result<T, Self::Error>;

  /// Visit a prefix expression node (ex. `!x`, `-y`)
  ///
  /// # Errors
  ///
  /// Returns an error if the operator or operand is invalid or unsupported
  fn visit_prefix(&mut self, prefix: &Prefix) -> Result<T, Self::Error>;

  /// Visit an infix expression node (ex. `x + y`)
  ///
  /// # Errors
  ///
  /// Returns an error if the operation is not valid for the given operand types
  fn visit_infix(&mut self, infix: &Infix) -> Result<T, Self::Error>;

  /// Visit an `if` expression node.
  ///
  /// # Errors
  ///
  /// Returns an error if the condition or either branch of the `if` is invalid.
  fn visit_if(&mut self, if_expr: &If) -> Result<T, Self::Error>;

  /// Visit a function literal node
  ///
  /// # Errors
  ///
  /// Returns an error if the function declaration is malformed or invalid
  fn visit_function_literal(&mut self, func: &FunctionLiteral) -> Result<T, Self::Error>;

  /// Visit a function call node
  ///
  /// # Errors
  ///
  /// Returns an error if the function call is invalid, such as calling a
  /// non-function, passing incorrect arguments, or failing during resolution
  fn visit_function_call(&mut self, call: &FunctionCall) -> Result<T, Self::Error>;
}

pub trait Visitable<T> {
  /// Accept a visitor, delegating the visit operation to the appropriate
  /// visitor method
  ///
  /// # Errors
  ///
  /// Returns an error if the visitor encounters an error while processing this
  /// node
  fn accept<V: Visitor<T>>(&self, visitor: &mut V) -> Result<T, V::Error>;
}

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
  Let(Let),
  Return(Return),
  Expr(Dangling),
  Block(Block),
}

impl<T> Visitable<T> for Statement {
  fn accept<V>(&self, visitor: &mut V) -> Result<T, V::Error>
  where
    V: Visitor<T>,
  {
    visitor.visit_statement(self)
  }
}

impl From<Let> for Statement {
  fn from(r#let: Let) -> Self {
    Self::Let(r#let)
  }
}

impl From<Return> for Statement {
  fn from(r#return: Return) -> Self {
    Self::Return(r#return)
  }
}

impl From<Dangling> for Statement {
  fn from(dangler: Dangling) -> Self {
    Self::Expr(dangler)
  }
}

impl From<Block> for Statement {
  fn from(block: Block) -> Self {
    Self::Block(block)
  }
}

impl fmt::Display for Statement {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::Let(r#let) => write!(f, "{let}"),
      Self::Return(r#return) => write!(f, "{return}"),
      Self::Expr(dangler) => write!(f, "{dangler}"),
      Self::Block(block) => write!(f, "{block}"),
    }
  }
}

// statement
#[derive(Debug, Clone, PartialEq)]
pub struct Let {
  pub token: Token,
  pub name: Identifier,
  pub value: Box<Expression>,
}

impl<T> Visitable<T> for Let {
  fn accept<V>(&self, visitor: &mut V) -> Result<T, V::Error>
  where
    V: Visitor<T>,
  {
    visitor.visit_let(self)
  }
}

impl fmt::Display for Let {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "let {} = {};", self.name, self.value)
  }
}

// statement
#[derive(Debug, Clone, PartialEq)]
pub struct Return {
  pub token: Token,
  pub value: Option<Box<Expression>>,
}

impl<T> Visitable<T> for Return {
  fn accept<V>(&self, visitor: &mut V) -> Result<T, V::Error>
  where
    V: Visitor<T>,
  {
    visitor.visit_return(self)
  }
}

impl fmt::Display for Return {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "return")?;
    if let Some(value) = &self.value {
      write!(f, " {value}")?;
    }
    write!(f, ";")
  }
}

// statement
#[derive(Debug, Clone, PartialEq)]
pub struct Dangling {
  pub token: Token,
  pub expr: Box<Expression>,
}

impl<T> Visitable<T> for Dangling {
  fn accept<V>(&self, visitor: &mut V) -> Result<T, V::Error>
  where
    V: Visitor<T>,
  {
    visitor.visit_dangling(self)
  }
}

impl fmt::Display for Dangling {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}", self.expr)?;
    // Only add semicolon if not a block expression
    if !matches!(
      *self.expr,
      Expression::If(_) | Expression::FunctionLiteral(_)
    ) {
      write!(f, ";")?;
    }
    Ok(())
  }
}

// statement
#[derive(Debug, Clone, PartialEq)]
pub struct Block {
  pub token: Token,
  pub statements: Vec<Statement>,
}

impl<T> Visitable<T> for Block {
  fn accept<V>(&self, visitor: &mut V) -> Result<T, V::Error>
  where
    V: Visitor<T>,
  {
    visitor.visit_block(self)
  }
}

impl fmt::Display for Block {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    writeln!(f, "{{")?;
    for stmt in &self.statements {
      writeln!(f, "{stmt}")?;
    }
    write!(f, "}}")
  }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
  Identifier(Identifier),
  Integer(Integer),
  Boolean(Boolean),
  String(Str),
  Array(Array),
  Index(Index),
  Prefix(Prefix),
  Infix(Infix),
  If(If),
  FunctionLiteral(FunctionLiteral),
  FunctionCall(FunctionCall),
}

impl<T> Visitable<T> for Expression {
  fn accept<V>(&self, visitor: &mut V) -> Result<T, V::Error>
  where
    V: Visitor<T>,
  {
    visitor.visit_expression(self)
  }
}

impl From<Identifier> for Expression {
  fn from(identifier: Identifier) -> Self {
    Self::Identifier(identifier)
  }
}

impl From<Integer> for Expression {
  fn from(integer: Integer) -> Self {
    Self::Integer(integer)
  }
}

impl From<Boolean> for Expression {
  fn from(boolean: Boolean) -> Self {
    Self::Boolean(boolean)
  }
}

impl From<Str> for Expression {
  fn from(string: Str) -> Self {
    Self::String(string)
  }
}

impl From<Array> for Expression {
  fn from(array: Array) -> Self {
    Self::Array(array)
  }
}

impl From<Index> for Expression {
  fn from(index: Index) -> Self {
    Self::Index(index)
  }
}

impl From<Prefix> for Expression {
  fn from(prefix: Prefix) -> Self {
    Self::Prefix(prefix)
  }
}

impl From<Infix> for Expression {
  fn from(infix: Infix) -> Self {
    Self::Infix(infix)
  }
}

impl From<If> for Expression {
  fn from(r#if: If) -> Self {
    Self::If(r#if)
  }
}

impl From<FunctionLiteral> for Expression {
  fn from(function_literal: FunctionLiteral) -> Self {
    Self::FunctionLiteral(function_literal)
  }
}

impl From<FunctionCall> for Expression {
  fn from(function_call: FunctionCall) -> Self {
    Self::FunctionCall(function_call)
  }
}

impl fmt::Display for Expression {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::Identifier(identifier) => write!(f, "{identifier}"),
      Self::Integer(integer) => write!(f, "{integer}"),
      Self::Boolean(boolean) => write!(f, "{boolean}"),
      Self::String(string) => write!(f, "{string}"),
      Self::Array(array) => write!(f, "{array}"),
      Self::Index(index) => write!(f, "{index}"),
      Self::Prefix(prefix) => write!(f, "{prefix}"),
      Self::Infix(infix) => write!(f, "{infix}"),
      // apparently format strings DO like raw literals...
      Self::If(r#if) => write!(f, "{if}"),
      Self::FunctionLiteral(function_literal) => write!(f, "{function_literal}"),
      Self::FunctionCall(function_call) => write!(f, "{function_call}"),
    }
  }
}

// expression
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Identifier {
  pub token: Token,
  pub value: String,
}

impl<T> Visitable<T> for Identifier {
  fn accept<V>(&self, visitor: &mut V) -> Result<T, V::Error>
  where
    V: Visitor<T>,
  {
    visitor.visit_identifier(self)
  }
}

impl fmt::Display for Identifier {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}", self.value)
  }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Integer {
  pub token: Token,
  pub value: i64,
}

impl<T> Visitable<T> for Integer {
  fn accept<V>(&self, visitor: &mut V) -> Result<T, V::Error>
  where
    V: Visitor<T>,
  {
    visitor.visit_integer(self)
  }
}

impl fmt::Display for Integer {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}", self.value)
  }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Boolean {
  pub token: Token,
  pub value: bool,
}

impl<T> Visitable<T> for Boolean {
  fn accept<V>(&self, visitor: &mut V) -> Result<T, V::Error>
  where
    V: Visitor<T>,
  {
    visitor.visit_boolean(self)
  }
}

impl fmt::Display for Boolean {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}", self.value)
  }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Str {
  pub token: Token,
  pub value: String,
}

impl<T> Visitable<T> for Str {
  fn accept<V>(&self, visitor: &mut V) -> Result<T, V::Error>
  where
    V: Visitor<T>,
  {
    visitor.visit_string(self)
  }
}

impl fmt::Display for Str {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "\"{}\"", self.value)
  }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Array {
  pub token: Token,
  pub elements: Vec<Expression>,
}

impl<T> Visitable<T> for Array {
  fn accept<V>(&self, visitor: &mut V) -> Result<T, V::Error>
  where
    V: Visitor<T>,
  {
    visitor.visit_array(self)
  }
}

impl fmt::Display for Array {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    // todo make this display nice
    write!(f, "{:?}", self.elements)
  }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Index {
  pub token: Token,
  pub left: Box<Expression>,
  pub index: Box<Expression>,
}

impl<T> Visitable<T> for Index {
  fn accept<V>(&self, visitor: &mut V) -> Result<T, V::Error>
  where
    V: Visitor<T>,
  {
    visitor.visit_index(self)
  }
}

impl fmt::Display for Index {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    // todo make this display nice
    write!(f, "{}[{}]", self.left, self.index)
  }
}

#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum PrefixOperatorError {
  #[error("token kind is not a prefix operator: {0}")]
  TokenNotPrefix(TokenKind),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefixOperator {
  Not,
  Negative,
}

impl TryFrom<TokenKind> for PrefixOperator {
  type Error = PrefixOperatorError;

  fn try_from(kind: TokenKind) -> Result<Self, Self::Error> {
    let op = match kind {
      TokenKind::Bang => Self::Not,
      TokenKind::Minus => Self::Negative,
      _ => return Err(Self::Error::TokenNotPrefix(kind)),
    };
    Ok(op)
  }
}

impl fmt::Display for PrefixOperator {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::Not => write!(f, "!"),
      Self::Negative => write!(f, "-"),
    }
  }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Prefix {
  pub token: Token,
  pub operator: PrefixOperator,
  pub right: Box<Expression>,
}

impl<T> Visitable<T> for Prefix {
  fn accept<V>(&self, visitor: &mut V) -> Result<T, V::Error>
  where
    V: Visitor<T>,
  {
    visitor.visit_prefix(self)
  }
}

impl fmt::Display for Prefix {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "({}{})", self.operator, self.right)
  }
}

#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum InfixOperatorError {
  #[error("token kind is not an infix operator: {0}")]
  TokenNotInfix(TokenKind),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InfixOperator {
  Add,
  Sub,
  Mul,
  Div,
  Eq,
  Neq,
  Lt,
  Gt,
}

impl TryFrom<TokenKind> for InfixOperator {
  type Error = InfixOperatorError;

  fn try_from(kind: TokenKind) -> Result<Self, Self::Error> {
    let op = match kind {
      TokenKind::Plus => Self::Add,
      TokenKind::Minus => Self::Sub,
      TokenKind::Asterisk => Self::Mul,
      TokenKind::Slash => Self::Div,
      TokenKind::Eq => Self::Eq,
      TokenKind::NotEq => Self::Neq,
      TokenKind::Lt => Self::Lt,
      TokenKind::Gt => Self::Gt,
      _ => return Err(Self::Error::TokenNotInfix(kind)),
    };
    Ok(op)
  }
}

impl fmt::Display for InfixOperator {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::Add => write!(f, "+"),
      Self::Sub => write!(f, "-"),
      Self::Mul => write!(f, "*"),
      Self::Div => write!(f, "/"),
      Self::Eq => write!(f, "=="),
      Self::Neq => write!(f, "!="),
      Self::Lt => write!(f, "<"),
      Self::Gt => write!(f, ">"),
    }
  }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Infix {
  pub token: Token,
  pub operator: InfixOperator,
  pub left: Box<Expression>,
  pub right: Box<Expression>,
}

impl<T> Visitable<T> for Infix {
  fn accept<V>(&self, visitor: &mut V) -> Result<T, V::Error>
  where
    V: Visitor<T>,
  {
    visitor.visit_infix(self)
  }
}

impl fmt::Display for Infix {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "({} {} {})", self.left, self.operator, self.right)
  }
}

#[derive(Debug, Clone, PartialEq)]
pub struct If {
  pub token: Token,
  pub antecedent: Box<Expression>,
  pub consequent: Block,
  pub alternative: Option<Block>,
}

impl<T> Visitable<T> for If {
  fn accept<V>(&self, visitor: &mut V) -> Result<T, V::Error>
  where
    V: Visitor<T>,
  {
    visitor.visit_if(self)
  }
}

impl fmt::Display for If {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "if ({}) {}", self.antecedent, self.consequent)?;
    if let Some(alt) = &self.alternative {
      write!(f, " else {alt}")?;
    }
    Ok(())
  }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionLiteral {
  pub token: Token,
  pub parameters: Vec<Identifier>,
  pub body: Block,
}

impl<T> Visitable<T> for FunctionLiteral {
  fn accept<V>(&self, visitor: &mut V) -> Result<T, V::Error>
  where
    V: Visitor<T>,
  {
    visitor.visit_function_literal(self)
  }
}

impl fmt::Display for FunctionLiteral {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "fn(")?;
    for (i, param) in self.parameters.iter().enumerate() {
      if i > 0 {
        write!(f, ", ")?;
      }
      write!(f, "{param}")?;
    }
    write!(f, ") {}", self.body)
  }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionCall {
  pub token: Token,
  pub function: Box<Expression>, // Identifier OR FunctionLiteral
  pub arguments: Vec<Expression>,
}

impl<T> Visitable<T> for FunctionCall {
  fn accept<V>(&self, visitor: &mut V) -> Result<T, V::Error>
  where
    V: Visitor<T>,
  {
    visitor.visit_function_call(self)
  }
}

impl fmt::Display for FunctionCall {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}(", self.function)?;
    for (i, arg) in self.arguments.iter().enumerate() {
      if i > 0 {
        write!(f, ", ")?;
      }
      write!(f, "{arg}")?;
    }
    write!(f, ")")
  }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
  pub statements: Vec<Statement>,
}

impl<T> Visitable<T> for Program {
  fn accept<V>(&self, visitor: &mut V) -> Result<T, V::Error>
  where
    V: Visitor<T>,
  {
    visitor.visit_program(self)
  }
}

impl fmt::Display for Program {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    for stmt in &self.statements {
      write!(f, "{stmt}")?;
    }
    Ok(())
  }
}
