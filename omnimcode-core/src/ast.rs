// src/ast.rs - Abstract syntax tree definitions

#[derive(Clone, Debug, PartialEq)]
pub enum Statement {
    Print(Expression),
    Expression(Expression),
    VarDecl {
        name: String,
        value: Expression,
        is_harmonic: bool,
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
        body: Vec<Statement>,
        return_type: Option<String>,
    },
    Return(Option<Expression>),
    Break,
    Continue,
    Import {
        module: String,
        alias: Option<String>,
    },
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
    
    // Variables and access
    Variable(String),
    Index {
        name: String,
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
    
    // Function call
    Call {
        name: String,
        args: Vec<Expression>,
    },
    
    // Harmonic operations
    Resonance(Box<Expression>),
    Fold(Box<Expression>),
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
