use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

use crate::ast::{
  Array, Block, Boolean, Dangling, Expression, FunctionCall, FunctionLiteral, Identifier, If,
  Index, Infix, InfixOperator, Integer, Let, Prefix, PrefixOperator, Program, Return, Statement,
  Str, Visitor,
};

#[derive(Debug, Clone)]
pub struct Environment {
  store: Rc<RefCell<HashMap<String, Value>>>,
  outer: Option<Box<Environment>>,
}

impl Environment {
  #[must_use]
  pub fn new() -> Self {
    Self {
      store: Rc::new(RefCell::new(HashMap::new())),
      outer: None,
    }
  }

  #[must_use]
  pub fn new_enclosed(outer: Self) -> Self {
    Self {
      store: Rc::new(RefCell::new(HashMap::new())),
      outer: Some(Box::new(outer)),
    }
  }

  #[must_use]
  pub fn get(&self, name: &str) -> Option<Value> {
    self
      .store
      .borrow()
      .get(name)
      .cloned()
      .or_else(|| self.outer.as_ref().and_then(|outer| outer.get(name)))
  }

  pub fn set(&mut self, name: String, value: Value) {
    self.store.borrow_mut().insert(name, value);
  }
}

impl Default for Environment {
  fn default() -> Self {
    Self::new()
  }
}

#[derive(Clone)]
pub struct Builtin {
  name: &'static str,
  // you CAN clone an rc, CANNOT clone a box... so we use rc...
  func: Rc<dyn Fn(Vec<Value>) -> Result<Value, InterpreterError>>,
}

impl fmt::Debug for Builtin {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "Builtin({})", self.name)
  }
}

#[derive(Debug, Clone)]
pub enum Value {
  Integer(i64),
  Boolean(bool),
  String(String),
  Array(Vec<Value>),
  // todo might want to make this a struct, since we are passing in individual
  // components to the apply_function helper -> only need to pass in one thing
  Function {
    parameters: Vec<Identifier>,
    body: Block,
    env: Environment,
  },
  Return(Box<Value>),
  Builtin(Builtin),
  Null,
}

impl Value {
  const fn is_truthy(&self) -> bool {
    match self {
      Self::Boolean(b) => *b,
      Self::Null | Self::Integer(0) => false,
      _ => true,
    }
  }

  const fn type_name(&self) -> &'static str {
    match self {
      Self::Integer(_) => "INTEGER",
      Self::Boolean(_) => "BOOLEAN",
      Self::String(_) => "STRING",
      Self::Array(_) => "ARRAY",
      Self::Function { .. } => "FUNCTION",
      Self::Return(_) => "RETURN",
      Self::Builtin(_) => "BUILTIN",
      Self::Null => "NULL",
    }
  }
}

// todo use checked_ops for integer arithmetic, interpreter can crash on
// over/underflow
mod ops {
  use super::{InfixOperator, InterpreterError, Value};

  pub fn eval_infix(
    operator: InfixOperator,
    left: &Value,
    right: &Value,
  ) -> Result<Value, InterpreterError> {
    match operator {
      InfixOperator::Add => add(left, right, operator),
      InfixOperator::Sub => sub(left, right, operator),
      InfixOperator::Mul => mul(left, right, operator),
      InfixOperator::Div => div(left, right, operator),
      InfixOperator::Lt => lt(left, right, operator),
      InfixOperator::Gt => gt(left, right, operator),
      InfixOperator::Eq => eq(left, right),
      InfixOperator::Neq => neq(left, right),
    }
  }

  #[inline]
  fn add(left: &Value, right: &Value, operator: InfixOperator) -> Result<Value, InterpreterError> {
    match (left, right) {
      (Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l + r)),
      (Value::String(l), Value::String(r)) => Ok(Value::String(l.clone() + r)),
      _ => Err(InterpreterError::TypeMismatch {
        left_type: left.type_name().into(),
        operator: operator.to_string(),
        right_type: right.type_name().into(),
      }),
    }
  }

  #[inline]
  fn sub(left: &Value, right: &Value, operator: InfixOperator) -> Result<Value, InterpreterError> {
    match (left, right) {
      (Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l - r)),
      _ => Err(InterpreterError::TypeMismatch {
        left_type: left.type_name().into(),
        operator: operator.to_string(),
        right_type: right.type_name().into(),
      }),
    }
  }

  #[inline]
  fn mul(left: &Value, right: &Value, operator: InfixOperator) -> Result<Value, InterpreterError> {
    match (left, right) {
      (Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l * r)),
      (Value::String(s), Value::Integer(n)) => {
        if *n < 0 {
          return Err(InterpreterError::UnsupportedOperator {
            operator: "*".into(),
            operand_type: "negative integer applied to string".into(),
          });
        }
        // clippy prefers this for some reason, who am i to fight the formatter
        usize::try_from(*n).map_or_else(
          |_| {
            Err(InterpreterError::UnsupportedOperator {
              operator: "*".into(),
              operand_type: "large integer applied to string".into(),
            })
          },
          |n| Ok(Value::String(s.repeat(n))),
        )
      }
      _ => Err(InterpreterError::TypeMismatch {
        left_type: left.type_name().into(),
        operator: operator.to_string(),
        right_type: right.type_name().into(),
      }),
    }
  }

  #[inline]
  fn div(left: &Value, right: &Value, operator: InfixOperator) -> Result<Value, InterpreterError> {
    match (left, right) {
      (Value::Integer(l), Value::Integer(r)) => {
        if *r == 0 {
          Err(InterpreterError::DivisionByZero)
        } else {
          Ok(Value::Integer(l / r))
        }
      }
      _ => Err(InterpreterError::TypeMismatch {
        left_type: left.type_name().into(),
        operator: operator.to_string(),
        right_type: right.type_name().into(),
      }),
    }
  }

  #[inline]
  fn lt(left: &Value, right: &Value, operator: InfixOperator) -> Result<Value, InterpreterError> {
    match (left, right) {
      (Value::Integer(l), Value::Integer(r)) => Ok(Value::Boolean(l < r)),
      _ => Err(InterpreterError::TypeMismatch {
        left_type: left.type_name().into(),
        operator: operator.to_string(),
        right_type: right.type_name().into(),
      }),
    }
  }

  #[inline]
  fn gt(left: &Value, right: &Value, operator: InfixOperator) -> Result<Value, InterpreterError> {
    match (left, right) {
      (Value::Integer(l), Value::Integer(r)) => Ok(Value::Boolean(l > r)),
      _ => Err(InterpreterError::TypeMismatch {
        left_type: left.type_name().into(),
        operator: operator.to_string(),
        right_type: right.type_name().into(),
      }),
    }
  }

  // eq never fails
  #[inline]
  #[allow(clippy::unnecessary_wraps)]
  fn eq(left: &Value, right: &Value) -> Result<Value, InterpreterError> {
    let result = match (left, right) {
      (Value::Integer(l), Value::Integer(r)) => l == r,
      (Value::Boolean(l), Value::Boolean(r)) => l == r,
      (Value::String(l), Value::String(r)) => l == r,
      (Value::Null, Value::Null) => true,
      _ => false, // diff types not equal
    };
    // `neq` depends on this function always returning the Value::Boolean variant
    Ok(Value::Boolean(result))
  }

  #[inline]
  fn neq(left: &Value, right: &Value) -> Result<Value, InterpreterError> {
    eq(left, right).map(|v| match v {
      Value::Boolean(b) => Value::Boolean(!b),
      // we know that `eq` always returns a Value::Boolean...
      _ => unreachable!(),
    })
  }
}

// builtin functions like `len` and the like
mod builtin {
  use std::collections::HashMap;
  use std::rc::Rc;

  use super::{Builtin, InterpreterError, Value};

  pub fn init() -> HashMap<&'static str, Value> {
    let mut m = HashMap::new();
    insert_builtin_len(&mut m);
    m
  }

  fn insert_builtin_len(builtins: &mut HashMap<&'static str, Value>) {
    builtins.insert(
      "len",
      Value::Builtin(Builtin {
        name: "len",
        func: Rc::new(|args: Vec<Value>| {
          if args.len() != 1 {
            return Err(InterpreterError::WrongArgumentCount {
              expected: 1,
              actual: args.len(),
            });
          }
          match &args[0] {
            Value::String(s) => Ok(Value::Integer(i64::try_from(s.len()).expect("todo"))),
            Value::Array(a) => Ok(Value::Integer(i64::try_from(a.len()).expect("todo"))),
            other => Err(InterpreterError::TypeMismatch {
              left_type: other.type_name().into(),
              operator: "len".into(),
              right_type: String::new(),
            }),
          }
        }),
      }),
    );
  }
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum InterpreterError {
  #[error("identifier not found: {0}")]
  IdentifierNotFound(String),

  #[error("type mismatch: {left_type} {operator} {right_type}")]
  TypeMismatch {
    left_type: String,
    operator: String,
    right_type: String,
  },

  #[error("unsupported operator: {operator} for {operand_type}")]
  UnsupportedOperator {
    operator: String,
    operand_type: String,
  },

  #[error("division by zero")]
  DivisionByZero,

  #[error("not a function: {0}")]
  NotAFunction(String),

  #[error("wrong number of arguments: expected {expected}, got {actual}")]
  WrongArgumentCount { expected: usize, actual: usize },

  #[error("index operator not supported for type {0}")]
  IndexNotSupported(String),
}

pub struct Interpreter {
  environment: Environment,
  builtins: HashMap<&'static str, Value>,
}

impl Interpreter {
  #[must_use]
  pub fn new(environment: Environment) -> Self {
    Self {
      environment,
      builtins: builtin::init(),
    }
  }

  fn apply_index(left: &Value, index: &Value) -> Result<Value, InterpreterError> {
    match (left, index) {
      (Value::Array(elements), Value::Integer(i)) => {
        match usize::try_from(*i) {
          // we guarantee 0 <= uidx < elements.len(), unwrap is safe...
          Ok(i) if i < elements.len() => Ok(elements.get(i).unwrap().clone()),
          // all other cases (out of bounds, conversion error, etc) give null
          _ => Ok(Value::Null),
        }
      }
      _ => Err(InterpreterError::IndexNotSupported(left.type_name().into())),
    }
  }

  // helper to apply a function (in component form) to some argument values...
  fn apply_function(
    &self,
    parameters: &[Identifier],
    body: &Block,
    env: Environment,
    args: &[Value],
  ) -> Result<Value, InterpreterError> {
    if parameters.len() != args.len() {
      return Err(InterpreterError::WrongArgumentCount {
        expected: parameters.len(),
        actual: args.len(),
      });
    }

    let mut new_env = Environment::new_enclosed(env);
    for (param, arg) in parameters.iter().zip(args.iter()) {
      new_env.set(param.value.clone(), arg.clone());
    }

    let mut new_visitor = Self {
      environment: new_env,
      builtins: self.builtins.clone(),
    };
    let result = new_visitor.visit_block(body)?;

    match result {
      Value::Return(val) => Ok(*val),
      _ => Ok(result),
    }
  }
}

impl Visitor<Value> for Interpreter {
  type Error = InterpreterError;

  fn visit_program(&mut self, program: &Program) -> Result<Value, Self::Error> {
    let mut result = Value::Null;
    for stmt in &program.statements {
      result = self.visit_statement(stmt)?;
      if let Value::Return(val) = result {
        return Ok(*val);
      }
    }
    Ok(result)
  }

  fn visit_statement(&mut self, stmt: &Statement) -> Result<Value, Self::Error> {
    match stmt {
      Statement::Let(l) => self.visit_let(l),
      Statement::Return(r) => self.visit_return(r),
      Statement::Expr(e) => self.visit_dangling(e),
      Statement::Block(b) => self.visit_block(b),
    }
  }

  fn visit_let(&mut self, let_stmt: &Let) -> Result<Value, Self::Error> {
    let value = self.visit_expression(&let_stmt.value)?;
    self.environment.set(let_stmt.name.value.clone(), value);
    Ok(Value::Null)
  }

  fn visit_return(&mut self, return_stmt: &Return) -> Result<Value, Self::Error> {
    let value = if let Some(expr) = &return_stmt.value {
      self.visit_expression(expr)?
    } else {
      Value::Null
    };
    Ok(Value::Return(Box::new(value)))
  }

  fn visit_dangling(&mut self, dangling: &Dangling) -> Result<Value, Self::Error> {
    self.visit_expression(&dangling.expr)
  }

  fn visit_block(&mut self, block: &Block) -> Result<Value, Self::Error> {
    let mut result = Value::Null;
    for stmt in &block.statements {
      result = self.visit_statement(stmt)?;
      if matches!(result, Value::Return(_)) {
        return Ok(result);
      }
    }
    Ok(result)
  }

  fn visit_expression(&mut self, expr: &Expression) -> Result<Value, Self::Error> {
    match expr {
      Expression::Identifier(i) => self.visit_identifier(i),
      Expression::Integer(i) => self.visit_integer(i),
      Expression::Boolean(b) => self.visit_boolean(b),
      Expression::String(s) => self.visit_string(s),
      Expression::Array(a) => self.visit_array(a),
      Expression::Index(i) => self.visit_index(i),
      Expression::Prefix(p) => self.visit_prefix(p),
      Expression::Infix(i) => self.visit_infix(i),
      Expression::If(i) => self.visit_if(i),
      Expression::FunctionLiteral(f) => self.visit_function_literal(f),
      Expression::FunctionCall(f) => self.visit_function_call(f),
    }
  }

  fn visit_identifier(&mut self, ident: &Identifier) -> Result<Value, Self::Error> {
    if let Some(val) = self.environment.get(ident.value.as_str()) {
      return Ok(val);
    }
    if let Some(builtin) = self.builtins.get(ident.value.as_str()) {
      return Ok(builtin.clone());
    }
    Err(InterpreterError::IdentifierNotFound(ident.value.clone()))
  }

  fn visit_integer(&mut self, int: &Integer) -> Result<Value, Self::Error> {
    Ok(Value::Integer(int.value))
  }

  fn visit_boolean(&mut self, bool: &Boolean) -> Result<Value, Self::Error> {
    Ok(Value::Boolean(bool.value))
  }

  fn visit_string(&mut self, string: &Str) -> Result<Value, Self::Error> {
    Ok(Value::String(string.value.clone()))
  }

  fn visit_array(&mut self, array: &Array) -> Result<Value, Self::Error> {
    let array: Result<Vec<_>, _> = array
      .elements
      .iter()
      .map(|expr| self.visit_expression(expr))
      .collect();
    let array = array?;
    Ok(Value::Array(array))
  }

  fn visit_index(&mut self, index: &Index) -> Result<Value, Self::Error> {
    let left = self.visit_expression(&index.left)?;
    let index = self.visit_expression(&index.index)?;
    Self::apply_index(&left, &index)
  }

  fn visit_prefix(&mut self, prefix: &Prefix) -> Result<Value, Self::Error> {
    let right = self.visit_expression(&prefix.right)?;
    match prefix.operator {
      PrefixOperator::Not => Ok(Value::Boolean(!right.is_truthy())),
      PrefixOperator::Negative => match right {
        Value::Integer(i) => Ok(Value::Integer(-i)),
        _ => Err(InterpreterError::UnsupportedOperator {
          operator: prefix.operator.to_string(),
          operand_type: right.type_name().to_string(),
        }),
      },
    }
  }

  fn visit_infix(&mut self, infix: &Infix) -> Result<Value, Self::Error> {
    let left = self.visit_expression(&infix.left)?;
    let right = self.visit_expression(&infix.right)?;
    ops::eval_infix(infix.operator, &left, &right)
  }

  fn visit_if(&mut self, if_expr: &If) -> Result<Value, Self::Error> {
    let condition = self.visit_expression(&if_expr.antecedent)?;
    if condition.is_truthy() {
      self.visit_block(&if_expr.consequent)
    } else if let Some(alt) = &if_expr.alternative {
      self.visit_block(alt)
    } else {
      Ok(Value::Null)
    }
  }

  fn visit_function_literal(&mut self, func: &FunctionLiteral) -> Result<Value, Self::Error> {
    Ok(Value::Function {
      parameters: func.parameters.clone(),
      body: func.body.clone(),
      env: self.environment.clone(),
    })
  }

  fn visit_function_call(&mut self, call: &FunctionCall) -> Result<Value, Self::Error> {
    let function = self.visit_expression(&call.function)?;
    let args: Result<Vec<_>, _> = call
      .arguments
      .iter()
      .map(|arg| self.visit_expression(arg))
      .collect();
    let args = args?;

    match function {
      Value::Function {
        parameters,
        body,
        env,
      } => self.apply_function(&parameters, &body, env, &args),
      Value::Builtin(builtin_fn) => (builtin_fn.func)(args),
      _ => Err(InterpreterError::NotAFunction(function.type_name().into())),
    }
  }
}

impl Default for Interpreter {
  fn default() -> Self {
    Self::new(Environment::new())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::lexer::Lexer;
  use crate::parser::Parser;
  use rstest::*;

  fn eval_input(input: &str) -> Result<Value, InterpreterError> {
    let lexer = Lexer::new(input);
    let parser = Parser::new(lexer);
    let program = match parser.parse() {
      Ok(program) => program,
      Err(errors) => panic!("Parser errors: {:?}", errors),
    };
    let mut interpreter = Interpreter::default();
    interpreter.visit_program(&program)
  }

  mod interpreter {
    use super::*;

    mod fixtures {
      use super::*;

      #[fixture]
      pub fn integer_expressions() -> Vec<(&'static str, i64)> {
        vec![
          ("5", 5),
          ("10", 10),
          ("-5", -5),
          ("-10", -10),
          ("5 + 5 + 5 + 5 - 10", 10),
          ("2 * 2 * 2 * 2 * 2", 32),
          ("-50 + 100 + -50", 0),
          ("5 * 2 + 10", 20),
          ("5 + 2 * 10", 25),
          ("20 + 2 * -10", 0),
          ("50 / 2 * 2 + 10", 60),
          ("2 * (5 + 10)", 30),
          ("3 * 3 * 3 + 10", 37),
          ("3 * (3 * 3) + 10", 37),
          ("(5 + 10 * 2 + 15 / 3) * 2 + -10", 50),
        ]
      }

      #[fixture]
      pub fn boolean_expressions() -> Vec<(&'static str, bool)> {
        vec![
          ("true", true),
          ("false", false),
          ("1 < 2", true),
          ("1 > 2", false),
          ("1 < 1", false),
          ("1 > 1", false),
          ("1 == 1", true),
          ("1 != 1", false),
          ("1 == 2", false),
          ("1 != 2", true),
          ("true == true", true),
          ("false == false", true),
          ("true == false", false),
          ("true != false", true),
          ("false != true", true),
          ("(1 < 2) == true", true),
          ("(1 < 2) == false", false),
          ("(1 > 2) == true", false),
          ("(1 > 2) == false", true),
        ]
      }

      #[fixture]
      pub fn bang_expressions() -> Vec<(&'static str, bool)> {
        vec![
          ("!true", false),
          ("!false", true),
          ("!5", false),
          ("!!true", true),
          ("!!false", false),
          ("!!5", true),
        ]
      }

      #[fixture]
      pub fn if_expressions() -> Vec<(&'static str, Option<i64>)> {
        vec![
          ("if (true) { 10 }", Some(10)),
          ("if (false) { 10 }", None),
          ("if (1) { 10 }", Some(10)),
          ("if (1 < 2) { 10 }", Some(10)),
          ("if (1 > 2) { 10 }", None),
          ("if (1 > 2) { 10 } else { 20 }", Some(20)),
          ("if (1 < 2) { 10 } else { 20 }", Some(10)),
        ]
      }

      #[fixture]
      pub fn return_statements() -> Vec<(&'static str, i64)> {
        vec![
          ("return 10;", 10),
          ("return 10; 9;", 10),
          ("return 2 * 5; 9;", 10),
          ("9; return 2 * 5; 9;", 10),
          (
            "if (10 > 1) {
                if (10 > 1) {
                    return 10;
                }
                return 1;
            }",
            10,
          ),
        ]
      }

      #[fixture]
      pub fn let_statements() -> Vec<(&'static str, i64)> {
        vec![
          ("let a = 5; a;", 5),
          ("let a = 5 * 5; a;", 25),
          ("let a = 5; let b = a; b;", 5),
          ("let a = 5; let b = a; let c = a + b + 5; c;", 15),
        ]
      }

      #[fixture]
      pub fn function_applications() -> Vec<(&'static str, i64)> {
        vec![
          ("let identity = fn(x) { x; }; identity(5);", 5),
          ("let identity = fn(x) { return x; }; identity(5);", 5),
          ("let double = fn(x) { x * 2; }; double(5);", 10),
          ("let add = fn(x, y) { x + y; }; add(5, 5);", 10),
          ("let add = fn(x, y) { x + y; }; add(5 + 5, add(5, 5));", 20),
          ("fn(x) { x; }(5)", 5),
        ]
      }

      #[fixture]
      pub fn function_errors() -> Vec<(&'static str, &'static str)> {
        vec![
          ("fn(x) { x }(1, 2)", "wrong number of arguments"),
          ("fn(x, y) { x + y }(1)", "wrong number of arguments"),
          ("1 + fn(x) { x }", "type mismatch"),
        ]
      }

      #[fixture]
      pub fn complex_functions() -> Vec<(&'static str, i64)> {
        vec![
          (
            "let newAdder = fn(x) {
                fn(y) { x + y };
            };
            let addTwo = newAdder(2);
            addTwo(2);",
            4,
          ),
          (
            "let factorial = fn(n) {
                if (n == 0) {
                    1
                } else {
                    n * factorial(n - 1)
                }
            };
            factorial(5)",
            120,
          ),
          (
            "let add = fn(a, b) { a + b };
            let sub = fn(a, b) { a - b };
            let applyFunc = fn(a, b, func) { func(a, b) };
            applyFunc(2, 2, add);",
            4,
          ),
        ]
      }
    }

    #[rstest]
    #[case::integers(fixtures::integer_expressions())]
    #[case::return_stmts(fixtures::return_statements())]
    #[case::let_stmts(fixtures::let_statements())]
    #[case::functions(fixtures::function_applications())]
    #[case::complex(fixtures::complex_functions())]
    fn eval_integer(#[case] tests: Vec<(&str, i64)>) {
      for (input, expected) in tests {
        let result = eval_input(input).unwrap();
        match result {
          Value::Integer(i) => assert_eq!(i, expected),
          _ => panic!("expected integer, got {:?}", result),
        }
      }
    }

    #[rstest]
    #[case::booleans(fixtures::boolean_expressions())]
    #[case::bang(fixtures::bang_expressions())]
    fn eval_boolean(#[case] tests: Vec<(&str, bool)>) {
      for (input, expected) in tests {
        let result = eval_input(input).unwrap();
        match result {
          Value::Boolean(b) => assert_eq!(b, expected),
          _ => panic!("expected boolean, got {:?}", result),
        }
      }
    }

    #[rstest]
    #[case::if_else(fixtures::if_expressions())]
    fn eval_if_else(#[case] tests: Vec<(&str, Option<i64>)>) {
      for (input, expected) in tests {
        let result = eval_input(input).unwrap();
        match (result, expected) {
          (Value::Integer(i), Some(exp)) => assert_eq!(i, exp),
          (Value::Null, None) => {}
          _ => panic!("unexpected result"),
        }
      }
    }

    #[rstest]
    #[case::errors(fixtures::function_errors())]
    fn eval_errors(#[case] tests: Vec<(&str, &str)>) {
      for (input, expected_msg) in tests {
        let result = eval_input(input);
        match result {
          Err(err) => assert!(err.to_string().contains(expected_msg)),
          Ok(val) => panic!("expected error, got {:?}", val),
        }
      }
    }

    #[rstest]
    fn function_object() {
      let input = "fn(x) { x + 2; };";
      let result = eval_input(input).unwrap();
      match result {
        Value::Function {
          parameters, body, ..
        } => {
          assert_eq!(parameters.len(), 1);
          assert_eq!(parameters[0].value, "x");
          assert_eq!(body.statements.len(), 1);
        }
        _ => panic!("expected function, got {:?}", result),
      }
    }
  }
}
