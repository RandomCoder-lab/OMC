// src/ast.rs - Abstract syntax tree definitions

/// Source position. 1-indexed for human-friendly error reports.
/// Lives in ast.rs (rather than parser.rs) so AST nodes can carry
/// positions without depending on parser internals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pos {
    pub line: u32,
    pub col: u32,
}

impl Pos {
    /// Sentinel for synthesized AST nodes that don't trace back to
    /// a real source location (e.g. nodes created by the heal pass).
    pub fn unknown() -> Self {
        Pos { line: 0, col: 0 }
    }
}

impl std::fmt::Display for Pos {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.line == 0 {
            write!(f, "<unknown>")
        } else {
            write!(f, "{}:{}", self.line, self.col)
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Statement {
    Print(Expression),
    Expression(Expression),
    VarDecl {
        name: String,
        value: Expression,
        is_harmonic: bool,
    },
    Parameter {
        name: String,
        value: Expression,
    },
    Assignment {
        name: String,
        value: Expression,
    },
    IndexAssignment {
        name: String,
        index: Expression,
        value: Expression,
    },
    // a[key1][key2] = value — covers dict[key][subkey] and arr[i][field]
    ChainedIndexAssignment {
        name: String,
        first_index: Expression,
        second_index: Expression,
        value: Expression,
    },
    If {
        condition: Expression,
        then_body: Vec<Statement>,
        elif_parts: Vec<(Expression, Vec<Statement>)>,
        else_body: Option<Vec<Statement>>,
    },
    While {
        condition: Expression,
        body: Vec<Statement>,
    },
    For {
        var: String,
        iterable: ForIterable,
        body: Vec<Statement>,
    },
    FunctionDef {
        name: String,
        params: Vec<String>,
        param_types: Vec<Option<String>>,
        body: Vec<Statement>,
        return_type: Option<String>,
        pragmas: Vec<String>,
    },
    Return(Option<Expression>),
    Break,
    Continue,
    Import {
        module: String,
        alias: Option<String>,
        /// Selective imports: `from "path" import name1, name2;`.
        /// When `Some(names)`, only the listed names are imported into
        /// the global namespace (no alias prefix). When `None`, the
        /// whole module imports per `alias` (None = flat merge,
        /// Some = prefix all with `alias.`). Mutually exclusive with
        /// `alias` — parser enforces this.
        selected: Option<Vec<String>>,
    },
    /// `try { ... } catch err { ... }` with optional `finally { ... }`.
    /// If the try block raises an error (via `throw expr`, `error("msg")`,
    /// or any builtin failure), execution jumps to the catch block with
    /// `err_var` bound to a Value::String holding the error message. The
    /// `finally` block, if present, runs unconditionally after both the
    /// try body AND any handler — even when the handler itself raises.
    /// Matches Python's try/except/finally semantics.
    Try {
        body: Vec<Statement>,
        err_var: String,
        handler: Vec<Statement>,
        finally: Option<Vec<Statement>>,
    },
    /// `throw expr` — explicit exception raise. The expression is
    /// evaluated and its display-string becomes the error message
    /// that the surrounding catch (if any) receives in its err_var.
    /// Future work: carry the thrown Value through Err(Value) instead
    /// of stringifying, enabling typed-catch hierarchies.
    Throw(Expression),
    /// `yield expr` — emit one value from a generator function.
    /// MVP semantics (eager list-building): a fn containing any Yield
    /// statement is a generator. Calling it runs the body to completion,
    /// collecting yielded values into an array, and returns that array.
    /// This is NOT lazy — infinite generators would hang. Real
    /// coroutine-based lazy generators are future work; the eager
    /// list-building approach unlocks the syntactic shape today.
    Yield(Expression),
    /// `class Name { field1; field2; fn method1(self, ...) { ... } }`
    /// (optional `extends Parent` clause for inheritance).
    ///
    /// Each ClassDef desugars at register_user_functions time into:
    ///   - A constructor fn `Name(field1, field2, ...)` that builds a
    ///     dict with __class__="Name" plus each field as a key.
    ///   - One top-level fn per method, name-mangled as `Name__method`.
    ///
    /// Method dispatch `obj.method(args)` works because the
    /// call-resolution path checks whether the receiver is a Dict
    /// carrying __class__ and routes to the mangled fn name with
    /// `obj` injected as the first argument (the `self` slot).
    ///
    /// Inheritance: when `parent` is `Some("Parent")`, the
    /// instance's __class__ is still set to the child's name, but
    /// method dispatch falls back to the parent's mangled namespace
    /// (and recursively up the chain) if the child doesn't define
    /// the method. The Interpreter maintains a class_parents table
    /// for the lookup.
    ClassDef {
        name: String,
        parent: Option<String>,
        fields: Vec<String>,
        methods: Vec<Statement>, // each is a FunctionDef
    },
    /// `match expr { pat => stmts, ... }`. First arm whose pattern
    /// accepts the scrutinee runs; remaining arms are skipped.
    /// A wildcard or bare-identifier arm at the end is the default.
    /// If no arm matches, the whole match is a no-op (no error).
    Match {
        scrutinee: Expression,
        arms: Vec<MatchArm>,
    },
}

/// A single arm in a `match` statement. Patterns can:
///  - match literals (number, float, string, bool, null)
///  - match a wildcard (`_`) or bind a variable (any bare ident)
///  - match a range (numeric `1..10` or single-char string `"a".."z"`)
///  - alternate via `|` (`1 | 2 | 3`)
///  - dispatch on type name (`int`, `string`, `dict`, etc.)
///
/// Body is a sequence of statements (block or single `=> stmt;` arm).
#[derive(Clone, Debug, PartialEq)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub body: Vec<Statement>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Pattern {
    /// Matches anything; binds nothing.
    Wildcard,
    /// Matches anything; binds the value to `name` in the arm body.
    Bind(String),
    /// Matches by structural equality with the literal.
    LitInt(i64),
    LitFloat(f64),
    LitString(String),
    LitBool(bool),
    LitNull,
    /// Numeric range, inclusive on both ends. `lo..=hi`. Stored as
    /// inclusive because that's the common case for digit/letter
    /// dispatch (`'0'..='9'`, `'a'..='z'`).
    RangeInt(i64, i64),
    /// Single-char string range, inclusive. Each side must be a
    /// 1-char string at parse time. Matches a 1-char string whose
    /// codepoint falls in [lo, hi]. Useful for the JSON-parser
    /// `is_digit` style dispatch.
    RangeStr(char, char),
    /// Alternation: any of the inner patterns matches.
    Or(Vec<Pattern>),
    /// Match by type tag — same names as the `type_of` builtin.
    /// E.g. `int`, `float`, `string`, `bool`, `array`, `dict`,
    /// `function`, `null`.
    Type(String),
}

#[derive(Clone, Debug, PartialEq)]
pub enum ForIterable {
    Range { start: Expression, end: Expression },
    Expr(Expression),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Expression {
    // Literals
    Number(i64),
    Float(f64),
    String(String),
    Boolean(bool),
    Array(Vec<Expression>),
    /// Dict literal: `{"k1": v1, "k2": v2}`. Keys are string-typed
    /// expressions (must evaluate to strings); values are arbitrary.
    /// Stored as a Vec<(key_expr, value_expr)> so the compiler can
    /// emit them in source order.
    Dict(Vec<(Expression, Expression)>),
    
    // Variables and access
    Variable(String),
    Index {
        name: String,
        index: Box<Expression>,
    },
    ChainedIndex {
        object: Box<Expression>,
        index: Box<Expression>,
    },
    
    // Binary operations
    Add(Box<Expression>, Box<Expression>),
    Sub(Box<Expression>, Box<Expression>),
    Mul(Box<Expression>, Box<Expression>),
    Div(Box<Expression>, Box<Expression>),
    Mod(Box<Expression>, Box<Expression>),
    
    // Comparisons
    Eq(Box<Expression>, Box<Expression>),
    Ne(Box<Expression>, Box<Expression>),
    Lt(Box<Expression>, Box<Expression>),
    Le(Box<Expression>, Box<Expression>),
    Gt(Box<Expression>, Box<Expression>),
    Ge(Box<Expression>, Box<Expression>),
    
    // Logical
    And(Box<Expression>, Box<Expression>),
    Or(Box<Expression>, Box<Expression>),
    Not(Box<Expression>),

    // Bitwise
    BitAnd(Box<Expression>, Box<Expression>),
    BitOr(Box<Expression>, Box<Expression>),
    BitXor(Box<Expression>, Box<Expression>),
    BitNot(Box<Expression>),
    Shl(Box<Expression>, Box<Expression>),
    Shr(Box<Expression>, Box<Expression>),
    
    // Function call. `pos` is the source position of the callee
    // identifier — used for stack-trace line numbers. Synthesized
    // calls (e.g. from the heal pass) use Pos::unknown().
    Call {
        name: String,
        args: Vec<Expression>,
        pos: Pos,
    },

    // Call an expression-valued callee: f(x)(y), make_adder(5)(10).
    // Callee is any expression that evaluates to a Value::Function.
    CallExpr {
        callee: Box<Expression>,
        args: Vec<Expression>,
        pos: Pos,
    },
    
    // Harmonic operations
    Resonance(Box<Expression>),
    Fold(Box<Expression>),

    // H.5: user-declared runtime self-healing intent.
    // `safe <expr>` wraps an expression in self-healing semantics.
    // The interpreter pattern-matches the inner expression at eval
    // time and routes to the appropriate ONN primitive:
    //   safe a / b              → safe_divide(a, b)
    //   safe arr_get(a, idx)    → safe_arr_get(a, idx)
    //   safe arr_set(a, idx, v) → safe_arr_set(a, idx, v)
    // Other shapes fall through to evaluating the inner expression
    // directly (no-op), reserving the slot for future runtime guards.
    Safe(Box<Expression>),

    // Anonymous function expression (closure). Distinguished from
    // Statement::FunctionDef by being usable in expression context —
    // can be passed as an argument, returned from a function, stored
    // in a variable. Capture is by VALUE: when evaluated, the current
    // local scope is snapshot into the resulting Value::Function's
    // `captured` field. Read-only over its environment.
    Lambda {
        params: Vec<String>,
        body: Vec<Statement>,
    },

    // `if cond { stmts } else { stmts }` used as an expression.
    // Value is the last Statement::Expression in whichever branch runs,
    // or Null if the last statement is not an expression-statement.
    // Enables `h x = if cond { a } else { b }` and call args like
    // `fn(if cond { "on" } else { "off" })`.
    IfExpr {
        condition: Box<Expression>,
        then_body: Vec<Statement>,
        else_body: Option<Vec<Statement>>,
    },
}

impl Expression {
    pub fn add(left: Expression, right: Expression) -> Self {
        Expression::Add(Box::new(left), Box::new(right))
    }

    pub fn sub(left: Expression, right: Expression) -> Self {
        Expression::Sub(Box::new(left), Box::new(right))
    }

    pub fn mul(left: Expression, right: Expression) -> Self {
        Expression::Mul(Box::new(left), Box::new(right))
    }

    pub fn div(left: Expression, right: Expression) -> Self {
        Expression::Div(Box::new(left), Box::new(right))
    }

    pub fn and(left: Expression, right: Expression) -> Self {
        Expression::And(Box::new(left), Box::new(right))
    }

    pub fn or(left: Expression, right: Expression) -> Self {
        Expression::Or(Box::new(left), Box::new(right))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ast_construction() {
        let expr = Expression::Add(
            Box::new(Expression::Number(5)),
            Box::new(Expression::Number(3)),
        );
        
        match expr {
            Expression::Add(_, _) => {},
            _ => panic!("Expected Add expression"),
        }
    }
}
