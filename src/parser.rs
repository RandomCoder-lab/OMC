// src/parser.rs - OMNIcode lexer and recursive descent parser

use crate::ast::*;
use std::collections::VecDeque;

#[derive(Clone, Debug, PartialEq)]
pub enum Token {
    // Keywords
    Harmonic,    // 'h'
    If,
    Else,
    While,
    For,
    In,
    Fn,
    Return,
    Break,
    Continue,
    Print,
    Range,
    Import,
    Load,
    As,
    Res,
    Fold,
    
    // Identifiers and literals
    Ident(String),
    Number(i64),
    Float(f64),
    String(String),
    
    // Operators
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Eq,
    EqEq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    Not,
    
    // Delimiters
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Semicolon,
    Comma,
    Arrow,
    Dot,
    
    // Special
    Eof,
}

pub struct Lexer {
    input: Vec<char>,
    pos: usize,
}

impl Lexer {
    pub fn new(input: &str) -> Self {
        Lexer {
            input: input.chars().collect(),
            pos: 0,
        }
    }

    fn current(&self) -> Option<char> {
        if self.pos < self.input.len() {
            Some(self.input[self.pos])
        } else {
            None
        }
    }

    fn peek(&self, offset: usize) -> Option<char> {
        if self.pos + offset < self.input.len() {
            Some(self.input[self.pos + offset])
        } else {
            None
        }
    }

    fn advance(&mut self) -> Option<char> {
        if self.pos < self.input.len() {
            let c = self.input[self.pos];
            self.pos += 1;
            Some(c)
        } else {
            None
        }
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.current() {
            if c.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn skip_comment(&mut self) {
        if self.current() == Some('#') {
            while let Some(c) = self.current() {
                if c == '\n' {
                    break;
                }
                self.advance();
            }
        }
    }

    fn read_string(&mut self, quote: char) -> String {
        let mut result = String::new();
        self.advance(); // Skip opening quote
        while let Some(c) = self.current() {
            if c == quote {
                self.advance(); // Skip closing quote
                break;
            }
            if c == '\\' {
                self.advance();
                match self.current() {
                    Some('n') => result.push('\n'),
                    Some('t') => result.push('\t'),
                    Some('r') => result.push('\r'),
                    Some('\\') => result.push('\\'),
                    Some('"') => result.push('"'),
                    Some('\'') => result.push('\''),
                    Some(c) => result.push(c),
                    None => break,
                }
                self.advance();
            } else {
                result.push(c);
                self.advance();
            }
        }
        result
    }

    fn read_number(&mut self) -> Token {
        let mut num_str = String::new();
        let mut is_float = false;

        while let Some(c) = self.current() {
            if c.is_ascii_digit() {
                num_str.push(c);
                self.advance();
            } else if c == '.' && !is_float && self.peek(1).map_or(false, |ch| ch.is_ascii_digit()) {
                is_float = true;
                num_str.push(c);
                self.advance();
            } else {
                break;
            }
        }

        if is_float {
            Token::Float(num_str.parse().unwrap_or(0.0))
        } else {
            Token::Number(num_str.parse().unwrap_or(0))
        }
    }

    fn read_ident(&mut self) -> String {
        let mut ident = String::new();
        while let Some(c) = self.current() {
            if c.is_alphanumeric() || c == '_' {
                ident.push(c);
                self.advance();
            } else {
                break;
            }
        }
        ident
    }

    pub fn next_token(&mut self) -> Token {
        loop {
            self.skip_whitespace();

            if self.current() == Some('#') {
                self.skip_comment();
                continue;
            }

            match self.current() {
                None => return Token::Eof,
                Some('"') => return Token::String(self.read_string('"')),
                Some('\'') => return Token::String(self.read_string('\'')),
                Some(c) if c.is_ascii_digit() => return self.read_number(),
                Some(c) if c.is_alphabetic() || c == '_' => {
                    let ident = self.read_ident();
                    return match ident.as_str() {
                        "h" => Token::Harmonic,
                        "if" => Token::If,
                        "else" => Token::Else,
                        "while" => Token::While,
                        "for" => Token::For,
                        "in" => Token::In,
                        "fn" => Token::Fn,
                        "return" => Token::Return,
                        "break" => Token::Break,
                        "continue" => Token::Continue,
                        "print" => Token::Print,
                        "range" => Token::Range,
                        "import" => Token::Import,
                        "load" => Token::Load,
                        "as" => Token::As,
                        "res" => Token::Res,
                        "fold" => Token::Fold,
                        "and" => Token::And,
                        "or" => Token::Or,
                        "not" => Token::Not,
                        _ => Token::Ident(ident),
                    };
                }
                Some('+') => {
                    self.advance();
                    return Token::Plus;
                }
                Some('-') => {
                    self.advance();
                    if self.current() == Some('>') {
                        self.advance();
                        return Token::Arrow;
                    }
                    return Token::Minus;
                }
                Some('*') => {
                    self.advance();
                    return Token::Star;
                }
                Some('/') => {
                    self.advance();
                    return Token::Slash;
                }
                Some('%') => {
                    self.advance();
                    return Token::Percent;
                }
                Some('=') => {
                    self.advance();
                    if self.current() == Some('=') {
                        self.advance();
                        return Token::EqEq;
                    }
                    return Token::Eq;
                }
                Some('!') => {
                    self.advance();
                    if self.current() == Some('=') {
                        self.advance();
                        return Token::Ne;
                    }
                    return Token::Not;
                }
                Some('<') => {
                    self.advance();
                    if self.current() == Some('=') {
                        self.advance();
                        return Token::Le;
                    }
                    return Token::Lt;
                }
                Some('>') => {
                    self.advance();
                    if self.current() == Some('=') {
                        self.advance();
                        return Token::Ge;
                    }
                    return Token::Gt;
                }
                Some('(') => {
                    self.advance();
                    return Token::LParen;
                }
                Some(')') => {
                    self.advance();
                    return Token::RParen;
                }
                Some('{') => {
                    self.advance();
                    return Token::LBrace;
                }
                Some('}') => {
                    self.advance();
                    return Token::RBrace;
                }
                Some('[') => {
                    self.advance();
                    return Token::LBracket;
                }
                Some(']') => {
                    self.advance();
                    return Token::RBracket;
                }
                Some(';') => {
                    self.advance();
                    return Token::Semicolon;
                }
                Some(',') => {
                    self.advance();
                    return Token::Comma;
                }
                Some('.') => {
                    self.advance();
                    return Token::Dot;
                }
                Some(_c) => {
                    self.advance();
                    // Skip unknown characters
                }
            }
        }
    }

    pub fn tokenize(&mut self) -> Vec<Token> {
        let mut tokens = Vec::new();
        loop {
            let token = self.next_token();
            if token == Token::Eof {
                tokens.push(token);
                break;
            }
            tokens.push(token);
        }
        tokens
    }
}

pub struct Parser {
    tokens: VecDeque<Token>,
}

impl Parser {
    pub fn new(input: &str) -> Self {
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        Parser {
            tokens: tokens.into_iter().collect(),
        }
    }

    fn current(&self) -> Token {
        self.tokens.front().cloned().unwrap_or(Token::Eof)
    }

    fn advance(&mut self) -> Token {
        self.tokens.pop_front().unwrap_or(Token::Eof)
    }

    fn expect(&mut self, expected: Token) -> Result<(), String> {
        if self.current() == expected {
            self.advance();
            Ok(())
        } else {
            Err(format!("Expected {:?}, got {:?}", expected, self.current()))
        }
    }

    pub fn parse(&mut self) -> Result<Vec<Statement>, String> {
        let mut statements = Vec::new();
        
        while self.current() != Token::Eof {
            statements.push(self.parse_statement()?);
        }
        
        Ok(statements)
    }

    fn parse_statement(&mut self) -> Result<Statement, String> {
        match self.current() {
            Token::Harmonic => {
                self.advance();
                let name = self.parse_ident()?;
                self.expect(Token::Eq)?;
                let value = self.parse_expression()?;
                self.expect(Token::Semicolon)?;
                Ok(Statement::VarDecl {
                    name,
                    value,
                    is_harmonic: true,
                })
            }
            Token::If => self.parse_if_stmt(),
            Token::While => self.parse_while_stmt(),
            Token::For => self.parse_for_stmt(),
            Token::Fn => self.parse_function_def(),
            Token::Return => {
                self.advance();
                if self.current() == Token::Semicolon {
                    self.advance();
                    Ok(Statement::Return(None))
                } else {
                    let expr = self.parse_expression()?;
                    self.expect(Token::Semicolon)?;
                    Ok(Statement::Return(Some(expr)))
                }
            }
            Token::Break => {
                self.advance();
                self.expect(Token::Semicolon)?;
                Ok(Statement::Break)
            }
            Token::Continue => {
                self.advance();
                self.expect(Token::Semicolon)?;
                Ok(Statement::Continue)
            }
            Token::Print => {
                self.advance();
                self.expect(Token::LParen)?;
                let expr = self.parse_expression()?;
                self.expect(Token::RParen)?;
                self.expect(Token::Semicolon)?;
                Ok(Statement::Print(expr))
            }
            Token::Ident(_) => {
                // Could be assignment or expression statement
                let checkpoint = self.tokens.clone();
                let ident = self.parse_ident()?;
                
                match self.current() {
                    Token::Eq => {
                        self.advance();
                        let value = self.parse_expression()?;
                        self.expect(Token::Semicolon)?;
                        Ok(Statement::Assignment {
                            name: ident,
                            value,
                        })
                    }
                    Token::LBracket => {
                        self.advance();
                        let index = self.parse_expression()?;
                        self.expect(Token::RBracket)?;
                        self.expect(Token::Eq)?;
                        let value = self.parse_expression()?;
                        self.expect(Token::Semicolon)?;
                        Ok(Statement::IndexAssignment {
                            name: ident,
                            index,
                            value,
                        })
                    }
                    _ => {
                        // Parse as expression statement
                        self.tokens = checkpoint;
                        let expr = self.parse_expression()?;
                        self.expect(Token::Semicolon)?;
                        Ok(Statement::Expression(expr))
                    }
                }
            }
            _ => {
                let expr = self.parse_expression()?;
                self.expect(Token::Semicolon)?;
                Ok(Statement::Expression(expr))
            }
        }
    }

    fn parse_if_stmt(&mut self) -> Result<Statement, String> {
        self.expect(Token::If)?;
        let condition = self.parse_expression()?;
        self.expect(Token::LBrace)?;
        let then_body = self.parse_block()?;

        let mut elif_parts = Vec::new();
        let mut else_body = None;

        while self.current() == Token::Else {
            self.advance();
            if self.current() == Token::If {
                self.advance();
                let elif_cond = self.parse_expression()?;
                self.expect(Token::LBrace)?;
                let elif_body = self.parse_block()?;
                elif_parts.push((elif_cond, elif_body));
            } else {
                self.expect(Token::LBrace)?;
                else_body = Some(self.parse_block()?);
                break;
            }
        }

        Ok(Statement::If {
            condition,
            then_body,
            elif_parts,
            else_body,
        })
    }

    fn parse_while_stmt(&mut self) -> Result<Statement, String> {
        self.expect(Token::While)?;
        let condition = self.parse_expression()?;
        self.expect(Token::LBrace)?;
        let body = self.parse_block()?;

        Ok(Statement::While { condition, body })
    }

    fn parse_for_stmt(&mut self) -> Result<Statement, String> {
        self.expect(Token::For)?;
        let var = self.parse_ident()?;
        self.expect(Token::In)?;

        let iterable = if self.current() == Token::Range {
            self.advance();
            self.expect(Token::LParen)?;
            let start = self.parse_expression()?;
            self.expect(Token::Comma)?;
            let end = self.parse_expression()?;
            self.expect(Token::RParen)?;
            ForIterable::Range { start, end }
        } else {
            let expr = self.parse_expression()?;
            ForIterable::Expr(expr)
        };

        self.expect(Token::LBrace)?;
        let body = self.parse_block()?;

        Ok(Statement::For { var, iterable, body })
    }

    fn parse_function_def(&mut self) -> Result<Statement, String> {
        self.expect(Token::Fn)?;
        let name = self.parse_ident()?;
        self.expect(Token::LParen)?;

        let mut params = Vec::new();
        while self.current() != Token::RParen {
            params.push(self.parse_ident()?);
            if self.current() == Token::Comma {
                self.advance();
            }
        }
        self.expect(Token::RParen)?;

        let return_type = if self.current() == Token::Arrow {
            self.advance();
            Some(self.parse_ident()?)
        } else {
            None
        };

        self.expect(Token::LBrace)?;
        let body = self.parse_block()?;

        Ok(Statement::FunctionDef {
            name,
            params,
            body,
            return_type,
        })
    }

    fn parse_block(&mut self) -> Result<Vec<Statement>, String> {
        let mut statements = Vec::new();

        while self.current() != Token::RBrace && self.current() != Token::Eof {
            statements.push(self.parse_statement()?);
        }

        self.expect(Token::RBrace)?;
        Ok(statements)
    }

    fn parse_expression(&mut self) -> Result<Expression, String> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<Expression, String> {
        let mut left = self.parse_and()?;

        while self.current() == Token::Or {
            self.advance();
            let right = self.parse_and()?;
            left = Expression::or(left, right);
        }

        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expression, String> {
        let mut left = self.parse_not()?;

        while self.current() == Token::And {
            self.advance();
            let right = self.parse_not()?;
            left = Expression::and(left, right);
        }

        Ok(left)
    }

    fn parse_not(&mut self) -> Result<Expression, String> {
        if self.current() == Token::Not {
            self.advance();
            let expr = self.parse_not()?;
            Ok(Expression::Not(Box::new(expr)))
        } else {
            self.parse_comparison()
        }
    }

    fn parse_comparison(&mut self) -> Result<Expression, String> {
        let mut left = self.parse_additive()?;

        loop {
            let expr = match self.current() {
                Token::EqEq => {
                    self.advance();
                    let right = self.parse_additive()?;
                    Expression::Eq(Box::new(left), Box::new(right))
                }
                Token::Ne => {
                    self.advance();
                    let right = self.parse_additive()?;
                    Expression::Ne(Box::new(left), Box::new(right))
                }
                Token::Lt => {
                    self.advance();
                    let right = self.parse_additive()?;
                    Expression::Lt(Box::new(left), Box::new(right))
                }
                Token::Le => {
                    self.advance();
                    let right = self.parse_additive()?;
                    Expression::Le(Box::new(left), Box::new(right))
                }
                Token::Gt => {
                    self.advance();
                    let right = self.parse_additive()?;
                    Expression::Gt(Box::new(left), Box::new(right))
                }
                Token::Ge => {
                    self.advance();
                    let right = self.parse_additive()?;
                    Expression::Ge(Box::new(left), Box::new(right))
                }
                _ => break,
            };
            left = expr;
        }

        Ok(left)
    }

    fn parse_additive(&mut self) -> Result<Expression, String> {
        let mut left = self.parse_multiplicative()?;

        while matches!(self.current(), Token::Plus | Token::Minus) {
            let expr = match self.current() {
                Token::Plus => {
                    self.advance();
                    let right = self.parse_multiplicative()?;
                    Expression::add(left, right)
                }
                Token::Minus => {
                    self.advance();
                    let right = self.parse_multiplicative()?;
                    Expression::sub(left, right)
                }
                _ => break,
            };
            left = expr;
        }

        Ok(left)
    }

    fn parse_multiplicative(&mut self) -> Result<Expression, String> {
        let mut left = self.parse_primary()?;

        while matches!(self.current(), Token::Star | Token::Slash | Token::Percent) {
            let expr = match self.current() {
                Token::Star => {
                    self.advance();
                    let right = self.parse_primary()?;
                    Expression::mul(left, right)
                }
                Token::Slash => {
                    self.advance();
                    let right = self.parse_primary()?;
                    Expression::div(left, right)
                }
                Token::Percent => {
                    self.advance();
                    let right = self.parse_primary()?;
                    Expression::Mod(Box::new(left), Box::new(right))
                }
                _ => break,
            };
            left = expr;
        }

        Ok(left)
    }

    fn parse_primary(&mut self) -> Result<Expression, String> {
        match self.current() {
            Token::Number(n) => {
                let val = n;
                self.advance();
                Ok(Expression::Number(val))
            }
            Token::Float(f) => {
                let val = f;
                self.advance();
                Ok(Expression::Number(val as i64))
            }
            Token::String(s) => {
                let val = s;
                self.advance();
                Ok(Expression::String(val))
            }
            Token::LBracket => self.parse_array(),
            Token::LParen => {
                self.advance();
                let expr = self.parse_expression()?;
                self.expect(Token::RParen)?;
                Ok(expr)
            }
            Token::Res => {
                self.advance();
                self.expect(Token::LParen)?;
                let arg = self.parse_expression()?;
                self.expect(Token::RParen)?;
                Ok(Expression::Resonance(Box::new(arg)))
            }
            Token::Fold => {
                self.advance();
                self.expect(Token::LParen)?;
                let arg = self.parse_expression()?;
                self.expect(Token::RParen)?;
                Ok(Expression::Fold(Box::new(arg)))
            }
            Token::Ident(_) => self.parse_ident_expr(),
            _ => Err(format!("Unexpected token in expression: {:?}", self.current())),
        }
    }

    fn parse_ident_expr(&mut self) -> Result<Expression, String> {
        let name = self.parse_ident()?;

        match self.current() {
            Token::LParen => {
                self.advance();
                let mut args = Vec::new();
                while self.current() != Token::RParen {
                    args.push(self.parse_expression()?);
                    if self.current() == Token::Comma {
                        self.advance();
                    }
                }
                self.expect(Token::RParen)?;
                Ok(Expression::Call { name, args })
            }
            Token::LBracket => {
                self.advance();
                let index = self.parse_expression()?;
                self.expect(Token::RBracket)?;
                Ok(Expression::Index {
                    name,
                    index: Box::new(index),
                })
            }
            _ => Ok(Expression::Variable(name)),
        }
    }

    fn parse_array(&mut self) -> Result<Expression, String> {
        self.expect(Token::LBracket)?;
        let mut elements = Vec::new();

        while self.current() != Token::RBracket {
            elements.push(self.parse_expression()?);
            if self.current() == Token::Comma {
                self.advance();
            }
        }

        self.expect(Token::RBracket)?;
        Ok(Expression::Array(elements))
    }

    fn parse_ident(&mut self) -> Result<String, String> {
        match self.current() {
            Token::Ident(s) => {
                let val = s;
                self.advance();
                Ok(val)
            }
            _ => Err(format!("Expected identifier, got {:?}", self.current())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lexer_basic() {
        let mut lexer = Lexer::new("h x = 42;");
        assert_eq!(lexer.next_token(), Token::Harmonic);
        assert_eq!(lexer.next_token(), Token::Ident("x".to_string()));
        assert_eq!(lexer.next_token(), Token::Eq);
        assert_eq!(lexer.next_token(), Token::Number(42));
    }

    #[test]
    fn test_parser_simple() {
        let mut parser = Parser::new("print(42);");
        let statements = parser.parse().unwrap();
        assert_eq!(statements.len(), 1);
    }
}
