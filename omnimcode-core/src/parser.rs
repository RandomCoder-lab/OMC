// src/parser.rs - OMNIcode lexer and recursive descent parser

use crate::ast::*;
use std::collections::VecDeque;

/// One segment of an f-string body. `f"x={n+1} done"` lexes as
/// `[Literal("x="), Expr("n+1"), Literal(" done")]`. The parser
/// re-parses each Expr segment via a sub-Parser to produce a real
/// Expression AST and stitches the parts together via `concat_many`.
#[derive(Clone, Debug, PartialEq)]
pub enum FStringPart {
    Literal(String),
    Expr(String),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Token {
    // Keywords
    Harmonic,    // 'h'
    If,
    Else,
    Elif,
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
    From,
    As,
    Res,
    Fold,
    Safe,        // H.5 host-level support: `safe <expr>` prefix
    Try,
    Catch,
    Finally,
    Throw,
    Match,
    Class,
    Extends,
    Yield,
    /// f-string template — alternating literal and expression segments.
    /// Parser turns this into `concat_many(parts...)` at expression
    /// position.
    FString(Vec<FStringPart>),
    /// `..` for inclusive ranges in match patterns: `0..9`, `"a".."z"`.
    /// Lexed when not part of `..=` (which we don't use yet) or `...`.
    DotDot,
    /// `=>` arm separator in match. (Alternation uses the existing
    /// `BitOr` token — `|` in pattern position parses as alternation.)
    FatArrow,

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
    Colon,
    At,
    // Bitwise
    BitAnd,
    BitOr,
    BitXor,
    BitNot,
    Shl,
    Shr,

    // Special
    Eof,
}

// `Pos` lives in crate::ast — re-exported here so existing
// `crate::parser::Pos` references continue to compile.
pub use crate::ast::Pos;

pub struct Lexer {
    input: Vec<char>,
    pos: usize,
    line: u32,
    col: u32,
}

impl Lexer {
    pub fn new(input: &str) -> Self {
        Lexer {
            input: input.chars().collect(),
            pos: 0,
            line: 1,
            col: 1,
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
            if c == '\n' {
                self.line += 1;
                self.col = 1;
            } else {
                self.col += 1;
            }
            Some(c)
        } else {
            None
        }
    }

    /// Position at the start of the next token (i.e. after whitespace/comments
    /// have been skipped). The token-emitting code in `next_token` consumes
    /// the lookahead chars, so we capture this just before that consumption.
    fn snapshot_pos(&self) -> Pos {
        Pos { line: self.line, col: self.col }
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

    fn read_triple_quoted_string(&mut self) -> String {
        // Caller has verified the three opening `"` chars.
        let mut result = String::new();
        self.advance();
        self.advance();
        self.advance();
        loop {
            match self.current() {
                None => break,
                Some('"') if self.peek(1) == Some('"') && self.peek(2) == Some('"') => {
                    self.advance();
                    self.advance();
                    self.advance();
                    break;
                }
                Some(c) => {
                    result.push(c);
                    self.advance();
                }
            }
        }
        result
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

    /// Read an f-string body — `f"x={n}"` syntax. Splits the body into
    /// alternating literal and expression segments at `{...}` markers.
    /// The expression segments are stored as raw source strings; the
    /// parser later re-parses each via a sub-parser into a real
    /// Expression AST. `{{` and `}}` are escape sequences for literal
    /// `{` and `}` (Python-compatible).
    fn read_fstring(&mut self, quote: char) -> Vec<FStringPart> {
        let mut parts: Vec<FStringPart> = Vec::new();
        let mut cur_lit = String::new();
        self.advance(); // skip opening quote
        while let Some(c) = self.current() {
            if c == quote {
                self.advance();
                break;
            }
            if c == '{' {
                // `{{` -> literal `{`
                if self.peek(1) == Some('{') {
                    cur_lit.push('{');
                    self.advance(); self.advance();
                    continue;
                }
                // Flush current literal segment.
                if !cur_lit.is_empty() {
                    parts.push(FStringPart::Literal(std::mem::take(&mut cur_lit)));
                }
                self.advance(); // consume `{`
                let mut depth: i32 = 1;
                let mut expr_src = String::new();
                while let Some(ec) = self.current() {
                    if ec == '{' { depth += 1; expr_src.push(ec); self.advance(); continue; }
                    if ec == '}' {
                        depth -= 1;
                        if depth == 0 { self.advance(); break; }
                        expr_src.push(ec);
                        self.advance();
                        continue;
                    }
                    expr_src.push(ec);
                    self.advance();
                }
                parts.push(FStringPart::Expr(expr_src.trim().to_string()));
                continue;
            }
            if c == '}' {
                // `}}` -> literal `}`
                if self.peek(1) == Some('}') {
                    cur_lit.push('}');
                    self.advance(); self.advance();
                    continue;
                }
                // Bare `}` is an error in Python f-strings, but we
                // accept it as a literal for ergonomics.
                cur_lit.push('}');
                self.advance();
                continue;
            }
            if c == '\\' {
                self.advance();
                match self.current() {
                    Some('n') => cur_lit.push('\n'),
                    Some('t') => cur_lit.push('\t'),
                    Some('r') => cur_lit.push('\r'),
                    Some('\\') => cur_lit.push('\\'),
                    Some('"') => cur_lit.push('"'),
                    Some('\'') => cur_lit.push('\''),
                    Some(c) => cur_lit.push(c),
                    None => break,
                }
                self.advance();
            } else {
                cur_lit.push(c);
                self.advance();
            }
        }
        if !cur_lit.is_empty() {
            parts.push(FStringPart::Literal(cur_lit));
        }
        parts
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

        // Scientific-notation suffix: `e` / `E` optionally followed by
        // `+`/`-` then one or more digits. Only recognized when at
        // least one digit is already accumulated. Forces float type
        // even if the mantissa was integer (1e5 -> Float(100000.0)).
        // Without this, `1e-9` was misparsed as int(1) followed by
        // call(e, -9) — the "Function approx_eq expects 3 arguments,
        // got 4" error surfaced during the optimization-campaign
        // tests for the stats builtins.
        if !num_str.is_empty() {
            if let Some(c) = self.current() {
                if c == 'e' || c == 'E' {
                    let mut lookahead = 1;
                    let mut has_sign = false;
                    if matches!(self.peek(lookahead), Some('+') | Some('-')) {
                        has_sign = true;
                        lookahead += 1;
                    }
                    // Need at least one digit after e/E (and optional sign)
                    // to commit to scientific notation. Otherwise leave
                    // the `e` alone — it's an identifier or keyword.
                    if self.peek(lookahead).map_or(false, |ch| ch.is_ascii_digit()) {
                        is_float = true;
                        num_str.push(c);
                        self.advance();
                        if has_sign {
                            num_str.push(self.current().unwrap());
                            self.advance();
                        }
                        while let Some(c) = self.current() {
                            if c.is_ascii_digit() {
                                num_str.push(c);
                                self.advance();
                            } else {
                                break;
                            }
                        }
                    }
                }
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
            // C-style `// line comment` (used by some canonical .omc files alongside `#`).
            if self.current() == Some('/') && self.peek(1) == Some('/') {
                while let Some(c) = self.current() {
                    if c == '\n' {
                        break;
                    }
                    self.advance();
                }
                continue;
            }
            // C-style `/* block comment */`
            if self.current() == Some('/') && self.peek(1) == Some('*') {
                self.advance();
                self.advance();
                while let Some(c) = self.current() {
                    if c == '*' && self.peek(1) == Some('/') {
                        self.advance();
                        self.advance();
                        break;
                    }
                    self.advance();
                }
                continue;
            }

            match self.current() {
                None => return Token::Eof,
                Some('"') => {
                    // Triple-quoted """multi-line""" docstring detection.
                    if self.peek(1) == Some('"') && self.peek(2) == Some('"') {
                        return Token::String(self.read_triple_quoted_string());
                    }
                    return Token::String(self.read_string('"'));
                }
                Some('\'') => return Token::String(self.read_string('\'')),
                Some(c) if c.is_ascii_digit() => return self.read_number(),
                // f-string prefix: `f"..."` or `f'...'` (also `F"..."`).
                // Triggered ONLY when `f` is directly followed by a
                // quote — a bare `f` identifier still parses normally.
                Some(c) if (c == 'f' || c == 'F')
                    && matches!(self.peek(1), Some('"') | Some('\'')) => {
                    self.advance(); // consume `f`
                    let quote = self.current().unwrap();
                    return Token::FString(self.read_fstring(quote));
                }
                Some(c) if c.is_alphabetic() || c == '_' => {
                    let ident = self.read_ident();
                    return match ident.as_str() {
                        "h" => Token::Harmonic,
                        "if" => Token::If,
                        "else" => Token::Else,
                        "elif" => Token::Elif,
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
                        "from" => Token::From,
                        "load" => Token::Load,
                        "as" => Token::As,
                        "res" => Token::Res,
                        "fold" => Token::Fold,
                        "safe" => Token::Safe,
                        "try" => Token::Try,
                        "catch" => Token::Catch,
                        "finally" => Token::Finally,
                        "throw" => Token::Throw,
                        "class" => Token::Class,
                        "extends" => Token::Extends,
                        "yield" => Token::Yield,
                        "match" => Token::Match,
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
                    if self.current() == Some('>') {
                        // `=>` for match arms.
                        self.advance();
                        return Token::FatArrow;
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
                    if self.current() == Some('<') {
                        self.advance();
                        return Token::Shl;
                    }
                    return Token::Lt;
                }
                Some('>') => {
                    self.advance();
                    if self.current() == Some('=') {
                        self.advance();
                        return Token::Ge;
                    }
                    if self.current() == Some('>') {
                        self.advance();
                        return Token::Shr;
                    }
                    return Token::Gt;
                }
                Some('&') => {
                    self.advance();
                    // `&&` is the C-family logical-AND every LLM reaches for.
                    // Map to the same Token::And as the `and` keyword so
                    // either form works. Single `&` stays as bit-AND.
                    if self.current() == Some('&') {
                        self.advance();
                        return Token::And;
                    }
                    return Token::BitAnd;
                }
                Some('|') => {
                    self.advance();
                    // `||` is the C-family logical-OR every LLM reaches for.
                    // Map to the same Token::Or as the `or` keyword so
                    // either form works. Single `|` stays as bit-OR.
                    if self.current() == Some('|') {
                        self.advance();
                        return Token::Or;
                    }
                    return Token::BitOr;
                }
                Some('^') => {
                    self.advance();
                    return Token::BitXor;
                }
                Some('~') => {
                    self.advance();
                    return Token::BitNot;
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
                    if self.current() == Some('.') {
                        // `..` inclusive range in match patterns. We
                        // treat as inclusive since that's the only
                        // place ranges currently appear.
                        self.advance();
                        return Token::DotDot;
                    }
                    return Token::Dot;
                }
                Some(':') => {
                    self.advance();
                    return Token::Colon;
                }
                Some('@') => {
                    self.advance();
                    return Token::At;
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

    /// Like `tokenize`, but returns each token paired with the source
    /// position where it starts (1-indexed). Used by Parser for error
    /// messages with line:col.
    pub fn tokenize_with_pos(&mut self) -> Vec<(Token, Pos)> {
        let mut tokens = Vec::new();
        loop {
            // Capture position BEFORE skipping whitespace inside next_token.
            // `next_token` skips its own whitespace; we want the position of
            // the first char of the actual token, so we replicate the skip.
            self.skip_whitespace_and_comments_inline();
            let pos = self.snapshot_pos();
            let token = self.next_token();
            if token == Token::Eof {
                tokens.push((token, pos));
                break;
            }
            tokens.push((token, pos));
        }
        tokens
    }

    /// Pre-skip whitespace + comments without consuming the lookahead a
    /// token would start at. Used by `tokenize_with_pos` to grab the right
    /// starting position.
    fn skip_whitespace_and_comments_inline(&mut self) {
        loop {
            self.skip_whitespace();
            if self.current() == Some('#') {
                self.skip_comment();
                continue;
            }
            if self.current() == Some('/') && self.peek(1) == Some('/') {
                while let Some(c) = self.current() {
                    if c == '\n' {
                        break;
                    }
                    self.advance();
                }
                continue;
            }
            if self.current() == Some('/') && self.peek(1) == Some('*') {
                self.advance();
                self.advance();
                while let Some(c) = self.current() {
                    if c == '*' && self.peek(1) == Some('/') {
                        self.advance();
                        self.advance();
                        break;
                    }
                    self.advance();
                }
                continue;
            }
            break;
        }
    }
}

pub struct Parser {
    tokens: VecDeque<(Token, Pos)>,
}

impl Parser {
    pub fn new(input: &str) -> Self {
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize_with_pos();
        Parser {
            tokens: tokens.into_iter().collect(),
        }
    }

    fn current(&self) -> Token {
        self.tokens
            .front()
            .map(|(t, _)| t.clone())
            .unwrap_or(Token::Eof)
    }

    /// Position of the current (lookahead) token. Used to annotate error
    /// messages — "Expected RBrace, got Eof at line 12, col 5".
    fn current_pos(&self) -> Pos {
        self.tokens
            .front()
            .map(|(_, p)| *p)
            .unwrap_or_else(Pos::unknown)
    }

    fn advance(&mut self) -> Token {
        self.tokens
            .pop_front()
            .map(|(t, _)| t)
            .unwrap_or(Token::Eof)
    }

    fn expect(&mut self, expected: Token) -> Result<(), String> {
        if self.current() == expected {
            self.advance();
            Ok(())
        } else {
            Err(format!(
                "at {}: Expected {:?}, got {:?}",
                self.current_pos(),
                expected,
                self.current()
            ))
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
        // Collect any line-prefix pragmas. Two syntaxes accepted:
        //   @pragma[name]     — original verbose form
        //   @name             — short form (matches Rust attributes)
        // Both produce the same AST. The short form is friendlier for
        // user-facing pragmas like @no_heal where the verbose form is
        // boilerplate.
        let mut prefix_pragmas: Vec<String> = Vec::new();
        while self.current() == Token::At {
            self.advance();
            match self.current() {
                Token::Ident(ref s) if s == "pragma" => {
                    self.advance();
                    self.expect(Token::LBracket)?;
                    let name = match self.current() {
                        Token::Ident(s) => { self.advance(); s }
                        other => {
                            return Err(format!(
                                "Expected pragma name in @pragma[...], got {:?}",
                                other
                            ))
                        }
                    };
                    self.expect(Token::RBracket)?;
                    prefix_pragmas.push(name);
                }
                Token::Ident(s) => {
                    // Short form: @name → pragma "name"
                    let name = s.clone();
                    self.advance();
                    prefix_pragmas.push(name);
                }
                other => {
                    return Err(format!(
                        "Expected pragma name after '@' (e.g. @no_heal or @pragma[name]), got {:?}",
                        other
                    ))
                }
            }
        }

        // If we collected pragmas, the next statement must be a fn def — attach them.
        if !prefix_pragmas.is_empty() {
            let stmt = self.parse_statement()?;
            if let Statement::FunctionDef {
                name,
                params,
                param_types,
                body,
                return_type,
                mut pragmas,
            } = stmt
            {
                pragmas.splice(0..0, prefix_pragmas);
                return Ok(Statement::FunctionDef {
                    name,
                    params,
                    param_types,
                    body,
                    return_type,
                    pragmas,
                });
            } else {
                return Err("@pragma[...] must be followed by a function definition".to_string());
            }
        }

        // Docstring statement: bare string at statement position, optional `;`.
        // Canonical Python OMC uses `"""docstring"""` at top of fn body without
        // a trailing semicolon. Treat it as an expression statement.
        if let Token::String(_) = self.current() {
            let expr = self.parse_expression()?;
            if self.current() == Token::Semicolon {
                self.advance();
            }
            return Ok(Statement::Expression(expr));
        }

        match self.current() {
            Token::Harmonic => {
                self.advance();
                // Fixed-size array form: `h[N] name;` => `h name = arr_new(N, 0);`
                if self.current() == Token::LBracket {
                    self.advance();
                    let size_expr = self.parse_expression()?;
                    self.expect(Token::RBracket)?;
                    let name = self.parse_ident()?;
                    self.expect(Token::Semicolon)?;
                    return Ok(Statement::VarDecl {
                        name,
                        value: Expression::Call {
                            name: "arr_new".to_string(),
                            args: vec![size_expr, Expression::Number(0)],
                            pos: Pos::unknown(),
                        },
                        is_harmonic: true,
                    });
                }
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
            Token::Class => self.parse_class_def(),
            Token::Try => self.parse_try_stmt(),
            Token::Throw => {
                // `throw expr;` — evaluate expr, raise its display string
                // as the current frame's error. Caught by surrounding
                // try/catch; uncaught throws propagate to the top-level
                // error handler (which prints + exits the program).
                self.advance(); // consume `throw`
                let expr = self.parse_expression()?;
                self.expect(Token::Semicolon)?;
                Ok(Statement::Throw(expr))
            }
            Token::Yield => {
                // `yield expr;` — emit one value from a generator fn.
                // Eager-list MVP: each yield appends to a collector
                // that the call boundary turns into a Value::Array.
                self.advance();
                let expr = self.parse_expression()?;
                self.expect(Token::Semicolon)?;
                Ok(Statement::Yield(expr))
            }
            Token::Match => self.parse_match_stmt(),
            // `import core;` or `import core as c;` or `load "path";`
            Token::Import | Token::Load => {
                self.advance();
                let module = match self.current() {
                    Token::Ident(s) => {
                        self.advance();
                        s
                    }
                    Token::String(s) => {
                        self.advance();
                        s
                    }
                    other => {
                        return Err(format!(
                            "Expected module name (ident or string) after import/load, got {:?}",
                            other
                        ))
                    }
                };
                let alias = if self.current() == Token::As {
                    self.advance();
                    Some(self.parse_ident()?)
                } else {
                    None
                };
                self.expect(Token::Semicolon)?;
                Ok(Statement::Import { module, alias, selected: None })
            }
            // Selective import: `from "path" import name1, name2;`.
            // Pulls only the listed names into the global namespace,
            // unprefixed. Mutually exclusive with the `as` alias form.
            Token::From => {
                self.advance();
                let module = match self.current() {
                    Token::Ident(s) => { self.advance(); s }
                    Token::String(s) => { self.advance(); s }
                    other => {
                        return Err(format!(
                            "Expected module path (ident or string) after `from`, got {:?}",
                            other
                        ))
                    }
                };
                self.expect(Token::Import)?;
                // Comma-separated identifier list.
                let mut names = Vec::new();
                names.push(self.parse_ident()?);
                while self.current() == Token::Comma {
                    self.advance();
                    names.push(self.parse_ident()?);
                }
                self.expect(Token::Semicolon)?;
                Ok(Statement::Import {
                    module,
                    alias: None,
                    selected: Some(names),
                })
            }
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
                        // Could be `arr[idx] = value;` (IndexAssignment) or
                        // `arr[idx];` / `arr[idx] + 1;` (expression statement).
                        // Distinguish by what follows the `]`. If `=`, it's
                        // an assignment; otherwise rewind and re-parse as
                        // an expression statement so dict / array indexing
                        // works in expression position too.
                        let pre_lbracket = checkpoint.clone();
                        self.advance();
                        let index = self.parse_expression()?;
                        self.expect(Token::RBracket)?;
                        if self.current() == Token::Eq {
                            self.advance();
                            let value = self.parse_expression()?;
                            self.expect(Token::Semicolon)?;
                            Ok(Statement::IndexAssignment {
                                name: ident,
                                index,
                                value,
                            })
                        } else {
                            // Rewind and treat the whole thing as an
                            // expression statement.
                            self.tokens = pre_lbracket;
                            let expr = self.parse_expression()?;
                            self.expect(Token::Semicolon)?;
                            Ok(Statement::Expression(expr))
                        }
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
        // Friendlier hint for the classic `if x = 5 { ... }` typo. After
        // parsing `x` as the condition, the next token will be `=` which
        // is unexpected here. The generic LBrace-expect error says
        // "Expected LBrace, got Eq" — replace with an actionable hint.
        if self.current() == Token::Eq {
            return Err(format!(
                "at {}: `if` condition followed by `=`. Did you mean `==`? \
                 (assignment isn't a value; use `==` for the comparison.)",
                self.current_pos()
            ));
        }
        self.expect(Token::LBrace)?;
        let then_body = self.parse_block()?;

        let mut elif_parts = Vec::new();
        let mut else_body = None;

        // Accept both `else if COND { ... }` (old form, still works) and
        // `elif COND { ... }` (the Python-style sugar). Both produce
        // the same AST — Statement::If with elif_parts populated.
        loop {
            if self.current() == Token::Elif {
                self.advance();
                let elif_cond = self.parse_expression()?;
                self.expect(Token::LBrace)?;
                let elif_body = self.parse_block()?;
                elif_parts.push((elif_cond, elif_body));
            } else if self.current() == Token::Else {
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
            } else {
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

    /// `class Name { field1; field2; fn method1(self, args) { ... } ... }`
    ///
    /// Parser produces Statement::ClassDef. The interpreter's
    /// `register_user_functions` later desugars this into:
    ///   - A constructor fn `Name(field1, field2, ...)` building a Dict
    ///     with __class__="Name" + each positional field.
    ///   - One top-level fn per method, mangled as `Name__method`.
    ///
    /// Method dispatch happens at call time: `obj.method(args)` checks
    /// whether the receiver is a Dict with __class__ field and routes
    /// to the mangled fn name. No new Value variant required — the
    /// instance is just a regular Dict with a marker key.
    fn parse_class_def(&mut self) -> Result<Statement, String> {
        self.expect(Token::Class)?;
        let name = self.parse_ident()?;
        // Optional `extends Parent` clause.
        let parent = if self.current() == Token::Extends {
            self.advance();
            Some(self.parse_ident()?)
        } else {
            None
        };
        self.expect(Token::LBrace)?;
        let mut fields: Vec<String> = Vec::new();
        let mut methods: Vec<Statement> = Vec::new();
        while self.current() != Token::RBrace {
            if self.current() == Token::Fn {
                // Method definition — parse as a regular function.
                let m = self.parse_function_def()?;
                methods.push(m);
            } else {
                // Field declaration: just `field_name;` — implicit
                // positional ordering matches the constructor's
                // parameter list.
                let f = self.parse_ident()?;
                self.expect(Token::Semicolon)?;
                fields.push(f);
            }
        }
        self.expect(Token::RBrace)?;
        Ok(Statement::ClassDef { name, parent, fields, methods })
    }

    /// `try { ... } catch err { ... }` with optional trailing
    /// `finally { ... }`. The caught value is currently a Value::String
    /// holding the error message; future work will carry the thrown
    /// Value through unchanged for typed-catch hierarchies. Single
    /// catch arm only — multi-arm typed matching is later work.
    fn parse_try_stmt(&mut self) -> Result<Statement, String> {
        self.expect(Token::Try)?;
        self.expect(Token::LBrace)?;
        let body = self.parse_block()?;
        self.expect(Token::Catch)?;
        let err_var = self.parse_ident()?;
        self.expect(Token::LBrace)?;
        let handler = self.parse_block()?;
        // Optional `finally { ... }`. Runs unconditionally after both
        // the try body and any handler (including when handler itself
        // raises). Matches Python's try/except/finally semantics.
        let finally = if self.current() == Token::Finally {
            self.expect(Token::Finally)?;
            self.expect(Token::LBrace)?;
            Some(self.parse_block()?)
        } else {
            None
        };
        Ok(Statement::Try { body, err_var, handler, finally })
    }

    /// `match expr { pat => stmt, pat => { stmts }, ... }`
    /// Comma between arms is optional when the body is a brace block.
    fn parse_match_stmt(&mut self) -> Result<Statement, String> {
        self.expect(Token::Match)?;
        let scrutinee = self.parse_expression()?;
        self.expect(Token::LBrace)?;
        let mut arms = Vec::new();
        while self.current() != Token::RBrace {
            let pattern = self.parse_pattern()?;
            self.expect(Token::FatArrow)?;
            // Body is either a block `{ ... }` or a single statement
            // ending in `;` or `,`.
            let body = if self.current() == Token::LBrace {
                self.expect(Token::LBrace)?;
                self.parse_block()?
            } else {
                // Single statement — accept either `expr;` or `expr,`.
                // We parse as one Statement::Expression and require its
                // terminator separately.
                let expr = self.parse_expression()?;
                vec![Statement::Expression(expr)]
            };
            arms.push(crate::ast::MatchArm { pattern, body });
            // Optional comma between arms.
            if self.current() == Token::Comma {
                self.advance();
            }
        }
        self.expect(Token::RBrace)?;
        Ok(Statement::Match { scrutinee, arms })
    }

    /// Parse a single pattern. Alternation (`|`) is handled here;
    /// each alternative is a `parse_pattern_atom`.
    fn parse_pattern(&mut self) -> Result<crate::ast::Pattern, String> {
        let first = self.parse_pattern_atom()?;
        if self.current() != Token::BitOr {
            return Ok(first);
        }
        let mut alts = vec![first];
        while self.current() == Token::BitOr {
            self.advance();
            alts.push(self.parse_pattern_atom()?);
        }
        Ok(crate::ast::Pattern::Or(alts))
    }

    fn parse_pattern_atom(&mut self) -> Result<crate::ast::Pattern, String> {
        use crate::ast::Pattern;
        match self.current() {
            Token::Number(n) => {
                self.advance();
                if self.current() == Token::DotDot {
                    self.advance();
                    let hi = match self.current() {
                        Token::Number(h) => { self.advance(); h }
                        other => return Err(format!(
                            "expected upper bound after `..` in range pattern, got {:?}", other
                        )),
                    };
                    Ok(Pattern::RangeInt(n, hi))
                } else {
                    Ok(Pattern::LitInt(n))
                }
            }
            Token::Float(f) => { self.advance(); Ok(Pattern::LitFloat(f)) }
            Token::String(s) => {
                self.advance();
                if self.current() == Token::DotDot {
                    // `"a".."z"` — both sides must be 1-char strings.
                    let lo_chars: Vec<char> = s.chars().collect();
                    if lo_chars.len() != 1 {
                        return Err(format!(
                            "lower bound of string range must be a 1-char string, got {:?}", s
                        ));
                    }
                    self.advance();
                    let hi = match self.current() {
                        Token::String(h) => { self.advance(); h }
                        other => return Err(format!(
                            "expected string upper bound after `..` in range pattern, got {:?}", other
                        )),
                    };
                    let hi_chars: Vec<char> = hi.chars().collect();
                    if hi_chars.len() != 1 {
                        return Err(format!(
                            "upper bound of string range must be a 1-char string, got {:?}", hi
                        ));
                    }
                    Ok(Pattern::RangeStr(lo_chars[0], hi_chars[0]))
                } else {
                    Ok(Pattern::LitString(s))
                }
            }
            Token::Ident(name) => {
                self.advance();
                // Reserved type-tag names dispatch as Pattern::Type.
                // Anything else is a Bind (binds the value to the
                // identifier in the arm body) — including `_` which
                // we special-case to Wildcard so the body can't refer
                // to it (matches Rust convention).
                Ok(match name.as_str() {
                    "_" => Pattern::Wildcard,
                    "true" => Pattern::LitBool(true),
                    "false" => Pattern::LitBool(false),
                    "null" => Pattern::LitNull,
                    "int" | "float" | "string" | "bool" | "array"
                    | "dict" | "function" | "null_t" | "singularity" => {
                        Pattern::Type(name)
                    }
                    _ => Pattern::Bind(name),
                })
            }
            other => Err(format!("expected pattern, got {:?}", other)),
        }
    }

    fn parse_for_stmt(&mut self) -> Result<Statement, String> {
        self.expect(Token::For)?;
        let var = self.parse_ident()?;
        self.expect(Token::In)?;

        let iterable = if self.current() == Token::Range {
            self.advance();
            self.expect(Token::LParen)?;
            let first = self.parse_expression()?;
            // Canonical OMC supports both range(end) and range(start, end).
            if self.current() == Token::Comma {
                self.advance();
                let end = self.parse_expression()?;
                self.expect(Token::RParen)?;
                ForIterable::Range { start: first, end }
            } else {
                self.expect(Token::RParen)?;
                ForIterable::Range {
                    start: Expression::Number(0),
                    end: first,
                }
            }
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
        let mut param_types: Vec<Option<String>> = Vec::new();
        while self.current() != Token::RParen {
            let pname = self.parse_ident()?;
            // Optional `: type` annotation
            let ptype = if self.current() == Token::Colon {
                self.advance();
                Some(self.parse_ident()?)
            } else {
                None
            };
            params.push(pname);
            param_types.push(ptype);
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

        // Postfix annotations after return type:
        //   `-> int @hbit @register`
        //   `-> int @unroll:16 @avx512`  (parameterized)
        let mut pragmas: Vec<String> = Vec::new();
        while self.current() == Token::At {
            self.advance();
            let mut name = match self.current() {
                Token::Ident(_) => self.parse_ident()?,
                other => {
                    return Err(format!(
                        "Expected pragma name after '@', got {:?}",
                        other
                    ))
                }
            };
            // Optional `:value` parameter on a pragma. Capture as suffix on the name.
            if self.current() == Token::Colon {
                self.advance();
                let val = match self.current() {
                    Token::Number(n) => {
                        self.advance();
                        n.to_string()
                    }
                    Token::Ident(_) => self.parse_ident()?,
                    other => {
                        return Err(format!(
                            "Expected pragma value after ':', got {:?}",
                            other
                        ))
                    }
                };
                name.push(':');
                name.push_str(&val);
            }
            pragmas.push(name);
        }

        self.expect(Token::LBrace)?;
        let body = self.parse_block()?;

        Ok(Statement::FunctionDef {
            name,
            params,
            param_types,
            body,
            return_type,
            pragmas,
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
        // H.5: `safe <expr>` prefix wraps the rest of the expression in
        // self-healing semantics. The interpreter dispatches at eval time
        // based on the inner shape (Div → safe_divide, arr_get → safe_arr_get,
        // etc). Mirrors the OMC-written parser's behaviour in
        // examples/self_healing_h5.omc.
        if self.current() == Token::Safe {
            self.advance();
            let inner = self.parse_or()?;
            return Ok(Expression::Safe(Box::new(inner)));
        }
        // Lambda: `fn(params) { body }` as an expression. The named form
        // `fn name(params) { body }` remains a top-level statement;
        // lambdas distinguish themselves by having no name token between
        // `fn` and `(`.
        if self.current() == Token::Fn {
            // Peek by cloning the tokens — if the second token is LParen,
            // this is a lambda. Otherwise leave it for the statement parser
            // (which will likely error, since `fn name` at expression
            // position isn't valid).
            let lookahead = self.tokens.clone();
            self.advance(); // consume `fn`
            if self.current() == Token::LParen {
                return self.parse_lambda();
            }
            // Restore tokens — not a lambda; fall through. The caller's
            // parse_or will hit Token::Fn and error in parse_primary.
            self.tokens = lookahead;
        }
        self.parse_or()
    }

    /// Parse the parameter list + body of a lambda, after `fn` has been
    /// consumed and the current token is `(`. Mirrors the parameter-list
    /// shape of named function definitions.
    fn parse_lambda(&mut self) -> Result<Expression, String> {
        self.expect(Token::LParen)?;
        let mut params: Vec<String> = Vec::new();
        if self.current() != Token::RParen {
            loop {
                match self.current() {
                    Token::Ident(name) => {
                        self.advance();
                        params.push(name);
                    }
                    other => return Err(format!(
                        "expected parameter name in lambda, got {:?}", other
                    )),
                }
                if self.current() == Token::Comma {
                    self.advance();
                } else {
                    break;
                }
            }
        }
        self.expect(Token::RParen)?;
        // Optional `-> type` annotation, same as named fn defs. Skipped
        // structurally for now (informational only).
        if self.current() == Token::Minus {
            // Could be either `->` arrow or a stray minus; peek ahead.
            let saved = self.tokens.clone();
            self.advance();
            if self.current() == Token::Gt {
                self.advance();
                // Consume the type annotation token (Ident or keyword).
                self.advance();
            } else {
                self.tokens = saved;
            }
        }
        self.expect(Token::LBrace)?;
        let mut body: Vec<Statement> = Vec::new();
        while self.current() != Token::RBrace {
            body.push(self.parse_statement()?);
        }
        self.expect(Token::RBrace)?;
        Ok(Expression::Lambda { params, body })
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
            self.parse_bit_or()
        }
    }

    fn parse_bit_or(&mut self) -> Result<Expression, String> {
        let mut left = self.parse_bit_xor()?;
        while self.current() == Token::BitOr {
            self.advance();
            let right = self.parse_bit_xor()?;
            left = Expression::BitOr(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_bit_xor(&mut self) -> Result<Expression, String> {
        let mut left = self.parse_bit_and()?;
        while self.current() == Token::BitXor {
            self.advance();
            let right = self.parse_bit_and()?;
            left = Expression::BitXor(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_bit_and(&mut self) -> Result<Expression, String> {
        let mut left = self.parse_comparison()?;
        while self.current() == Token::BitAnd {
            self.advance();
            let right = self.parse_comparison()?;
            left = Expression::BitAnd(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_comparison(&mut self) -> Result<Expression, String> {
        let mut left = self.parse_shift()?;

        loop {
            let expr = match self.current() {
                Token::EqEq => {
                    self.advance();
                    let right = self.parse_shift()?;
                    Expression::Eq(Box::new(left), Box::new(right))
                }
                Token::Ne => {
                    self.advance();
                    let right = self.parse_shift()?;
                    Expression::Ne(Box::new(left), Box::new(right))
                }
                Token::Lt => {
                    self.advance();
                    let right = self.parse_shift()?;
                    Expression::Lt(Box::new(left), Box::new(right))
                }
                Token::Le => {
                    self.advance();
                    let right = self.parse_shift()?;
                    Expression::Le(Box::new(left), Box::new(right))
                }
                Token::Gt => {
                    self.advance();
                    let right = self.parse_shift()?;
                    Expression::Gt(Box::new(left), Box::new(right))
                }
                Token::Ge => {
                    self.advance();
                    let right = self.parse_shift()?;
                    Expression::Ge(Box::new(left), Box::new(right))
                }
                _ => break,
            };
            left = expr;
        }

        Ok(left)
    }

    fn parse_shift(&mut self) -> Result<Expression, String> {
        let mut left = self.parse_additive()?;
        loop {
            let expr = match self.current() {
                Token::Shl => {
                    self.advance();
                    let right = self.parse_additive()?;
                    Expression::Shl(Box::new(left), Box::new(right))
                }
                Token::Shr => {
                    self.advance();
                    let right = self.parse_additive()?;
                    Expression::Shr(Box::new(left), Box::new(right))
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
        // Unary bitwise NOT: `~x`
        if self.current() == Token::BitNot {
            self.advance();
            let inner = self.parse_primary()?;
            return Ok(Expression::BitNot(Box::new(inner)));
        }
        // Unary minus: `-x` becomes `0 - x` (cheap, no new AST variant needed)
        if self.current() == Token::Minus {
            self.advance();
            let inner = self.parse_primary()?;
            return Ok(Expression::Sub(
                Box::new(Expression::Number(0)),
                Box::new(inner),
            ));
        }
        match self.current() {
            Token::Number(n) => {
                let val = n;
                self.advance();
                Ok(Expression::Number(val))
            }
            Token::Float(f) => {
                let val = f;
                self.advance();
                Ok(Expression::Float(val))
            }
            Token::String(s) => {
                let val = s;
                self.advance();
                Ok(Expression::String(val))
            }
            Token::FString(parts) => {
                let parts_copy = parts.clone();
                self.advance();
                // Turn the f-string into `concat_many(seg0, seg1, ...)`
                // where literal segments are Expression::String and
                // expression segments are re-parsed via a sub-Parser.
                // concat_many tolerates int/float args by calling
                // to_string internally — so `f"x={n}"` works for any
                // value type without an explicit to_string call.
                let mut args: Vec<Expression> = Vec::new();
                for part in parts_copy {
                    match part {
                        FStringPart::Literal(s) => args.push(Expression::String(s)),
                        FStringPart::Expr(src) => {
                            let mut sub = Parser::new(&src);
                            let expr = sub.parse_expression()
                                .map_err(|e| format!("f-string expr `{}`: {}", src, e))?;
                            args.push(expr);
                        }
                    }
                }
                // Empty f-string `f""` produces "".
                if args.is_empty() { return Ok(Expression::String(String::new())); }
                Ok(Expression::Call {
                    name: "concat_many".to_string(),
                    args,
                    pos: crate::ast::Pos::unknown(),
                })
            }
            Token::LBracket => self.parse_array(),
            Token::LBrace => self.parse_dict(),
            Token::LParen => {
                self.advance();
                let expr = self.parse_expression()?;
                self.expect(Token::RParen)?;
                Ok(expr)
            }
            Token::Res => {
                self.advance();
                self.expect(Token::LParen)?;
                let mut args = Vec::new();
                while self.current() != Token::RParen {
                    args.push(self.parse_expression()?);
                    if self.current() == Token::Comma {
                        self.advance();
                    }
                }
                self.expect(Token::RParen)?;
                if args.len() == 1 {
                    Ok(Expression::Resonance(Box::new(args.into_iter().next().unwrap())))
                } else {
                    Ok(Expression::Call { name: "res".to_string(), args, pos: Pos::unknown() })
                }
            }
            Token::Fold => {
                self.advance();
                self.expect(Token::LParen)?;
                let mut args = Vec::new();
                while self.current() != Token::RParen {
                    args.push(self.parse_expression()?);
                    if self.current() == Token::Comma {
                        self.advance();
                    }
                }
                self.expect(Token::RParen)?;
                if args.len() == 1 {
                    Ok(Expression::Fold(Box::new(args.into_iter().next().unwrap())))
                } else {
                    Ok(Expression::Call { name: "fold".to_string(), args, pos: Pos::unknown() })
                }
            }
            Token::Ident(_) => self.parse_ident_expr(),
            // `range` is a soft keyword: when it appears in a `for x in
            // range(...)` it's special-cased in parse_for_stmt for the
            // optimized ForIterable::Range path; everywhere else it's
            // an ordinary builtin call. Parse it as a Call so it's
            // usable like Python's range outside loops too.
            Token::Range => {
                let pos = self.current_pos();
                self.advance();
                self.expect(Token::LParen)?;
                let mut args = Vec::new();
                while self.current() != Token::RParen {
                    args.push(self.parse_expression()?);
                    if self.current() == Token::Comma { self.advance(); }
                }
                self.expect(Token::RParen)?;
                Ok(Expression::Call { name: "range".to_string(), args, pos })
            }
            other => Err(format!(
                "at {}: Unexpected token in expression: {}",
                self.current_pos(),
                describe_token_in_expr(&other),
            )),
        }
    }

    fn parse_ident_expr(&mut self) -> Result<Expression, String> {
        // Capture position BEFORE consuming the identifier — this is
        // the position attached to any Expression::Call we build for
        // stack-trace line numbers.
        let callee_pos = self.current_pos();
        let mut name = self.parse_ident()?;

        // Handle module-qualified calls: phi.fold, core.fib, phi.res, etc.
        // Lexer emits Token::Dot; we join into a single name like "phi.fold"
        // to keep AST simple. Interpreter dispatches on the dotted name.
        // After a dot, accept keywords like `res`/`fold` as method names too.
        while self.current() == Token::Dot {
            self.advance();
            let part = match self.current() {
                Token::Ident(s) => {
                    self.advance();
                    s
                }
                Token::Res => {
                    self.advance();
                    "res".to_string()
                }
                Token::Fold => {
                    self.advance();
                    "fold".to_string()
                }
                other => {
                    return Err(format!(
                        "Expected method name after '.', got {:?}",
                        other
                    ))
                }
            };
            name.push('.');
            name.push_str(&part);
        }

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
                Ok(Expression::Call { name, args, pos: callee_pos })
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

    /// Parse a dict literal: `{"k1": v1, "k2": v2}` or `{}`.
    /// Reachable only from expression position; statement-level
    /// blocks (after if/while/fn) are matched by their own
    /// LBrace expectations and never enter parse_primary.
    fn parse_dict(&mut self) -> Result<Expression, String> {
        self.expect(Token::LBrace)?;
        let mut pairs = Vec::new();
        while self.current() != Token::RBrace {
            let key = self.parse_expression()?;
            self.expect(Token::Colon)?;
            let val = self.parse_expression()?;
            pairs.push((key, val));
            if self.current() == Token::Comma {
                self.advance();
            }
        }
        self.expect(Token::RBrace)?;
        Ok(Expression::Dict(pairs))
    }

    fn parse_ident(&mut self) -> Result<String, String> {
        match self.current() {
            Token::Ident(s) => {
                let val = s;
                self.advance();
                Ok(val)
            }
            other => {
                // Friendlier error: when the current token is a reserved
                // keyword the user accidentally tried to use as an
                // identifier, name it and suggest a fix. `h h = 1` is
                // the canonical case — the second `h` is the harmonic-
                // type keyword, not an identifier.
                let pos = self.current_pos();
                let hint = reserved_word_hint(&other);
                if let Some(hint) = hint {
                    Err(format!("at {}: {}", pos, hint))
                } else {
                    Err(format!(
                        "at {}: Expected identifier, got {:?}",
                        pos, other
                    ))
                }
            }
        }
    }
}

/// Pretty-print a token that turned up in expression position, with a
/// hint for common mistakes (assignment-vs-equality, semicolon between
/// expressions, etc.). The goal is to spend the parser's already-broken
/// state on something genuinely actionable.
fn describe_token_in_expr(tok: &Token) -> String {
    match tok {
        Token::Eq => "`=` here. Did you mean `==`? `=` is for assignment, \
                      `==` for equality.".to_string(),
        Token::Semicolon => "`;`. An expression can't be empty here — \
                              either remove the leading operator or fill \
                              in the missing value.".to_string(),
        Token::RBrace => "`}`. A block ended where an expression value \
                          was expected — check for a missing `return` or \
                          stray semicolon inside the block.".to_string(),
        Token::RParen => "`)`. Closing paren with no expression — empty \
                          parens are only allowed in fn calls / defs, not \
                          in value position.".to_string(),
        Token::Comma => "`,`. Unexpected comma — likely a stray trailing \
                          delimiter or a missing left-hand-side value.".to_string(),
        Token::Else => "`else` (with no `if`). Floating else — check for \
                         a missing `if` block above.".to_string(),
        Token::Catch | Token::Finally => format!(
            "`{:?}` (with no `try`). Check for a missing `try {{ }}` above.",
            tok
        ),
        other => format!("{:?}", other),
    }
}

/// When the parser expected an identifier but got a keyword, return a
/// human-facing hint that names the keyword and proposes a non-reserved
/// alternative. None for tokens that aren't keyword-like (literals,
/// punctuation) — those fall back to the generic error.
fn reserved_word_hint(tok: &Token) -> Option<String> {
    let (word, suggested) = match tok {
        Token::Harmonic => ("h", "hval"),
        Token::Fn => ("fn", "func"),
        Token::If => ("if", "cond"),
        Token::Else => ("else", "alt"),
        Token::Elif => ("elif", "alt"),
        Token::While => ("while", "loop_cond"),
        Token::For => ("for", "iter"),
        Token::In => ("in", "inside"),
        Token::Return => ("return", "ret"),
        Token::Break => ("break", "stop"),
        Token::Continue => ("continue", "skip"),
        Token::Import => ("import", "imp"),
        Token::From => ("from", "src"),
        Token::Range => ("range", "rng"),
        Token::Fold => ("fold", "folded"),
        Token::Res => ("res", "resval"),
        _ => return None,
    };
    Some(format!(
        "'{}' is a reserved keyword; can't use it as a variable name. \
         Try `{}` (or any non-reserved name).",
        word, suggested
    ))
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
