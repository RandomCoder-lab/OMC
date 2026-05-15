// omnimcode-core/src/formatter.rs — Canonical AST → OMC source emitter.
//
// Mirrors the V.4 pretty-printer from examples/self_healing_h5.omc but
// operates on the host AST (not the nested-array AST used inside the
// OMC-written self-hosting demos).
//
// Output is canonical, not byte-identical to the input. Whitespace,
// comments, and original paren style are dropped. The emitter always
// wraps BIN operations in parens to avoid precedence ambiguity — same
// trade as V.4 ("the round-trip rule is no precedence ambiguity, not
// minimal parens").
//
// Used by `--fmt` in main.rs.

use crate::ast::*;

const INDENT: &str = "    ";

pub fn format_program(stmts: &[Statement]) -> String {
    let mut out = String::new();
    for s in stmts {
        format_stmt(s, 0, &mut out);
    }
    out
}

fn indent_to(level: usize, out: &mut String) {
    for _ in 0..level {
        out.push_str(INDENT);
    }
}

fn format_stmt(stmt: &Statement, level: usize, out: &mut String) {
    indent_to(level, out);
    match stmt {
        Statement::Print(e) => {
            out.push_str("print(");
            format_expr(e, out);
            out.push_str(");\n");
        }
        Statement::Expression(e) => {
            format_expr(e, out);
            out.push_str(";\n");
        }
        Statement::VarDecl { name, value, .. } => {
            out.push_str("h ");
            out.push_str(name);
            out.push_str(" = ");
            format_expr(value, out);
            out.push_str(";\n");
        }
        Statement::Parameter { name, value } => {
            out.push_str("h ");
            out.push_str(name);
            out.push_str(" = ");
            format_expr(value, out);
            out.push_str(";\n");
        }
        Statement::Assignment { name, value } => {
            out.push_str(name);
            out.push_str(" = ");
            format_expr(value, out);
            out.push_str(";\n");
        }
        Statement::IndexAssignment { name, index, value } => {
            out.push_str(name);
            out.push('[');
            format_expr(index, out);
            out.push_str("] = ");
            format_expr(value, out);
            out.push_str(";\n");
        }
        Statement::If { condition, then_body, elif_parts, else_body } => {
            out.push_str("if ");
            format_expr(condition, out);
            out.push_str(" {\n");
            for s in then_body {
                format_stmt(s, level + 1, out);
            }
            indent_to(level, out);
            out.push('}');
            for (econd, ebody) in elif_parts {
                out.push_str(" else if ");
                format_expr(econd, out);
                out.push_str(" {\n");
                for s in ebody {
                    format_stmt(s, level + 1, out);
                }
                indent_to(level, out);
                out.push('}');
            }
            if let Some(body) = else_body {
                out.push_str(" else {\n");
                for s in body {
                    format_stmt(s, level + 1, out);
                }
                indent_to(level, out);
                out.push('}');
            }
            out.push('\n');
        }
        Statement::While { condition, body } => {
            out.push_str("while ");
            format_expr(condition, out);
            out.push_str(" {\n");
            for s in body {
                format_stmt(s, level + 1, out);
            }
            indent_to(level, out);
            out.push_str("}\n");
        }
        Statement::For { var, iterable, body } => {
            out.push_str("for ");
            out.push_str(var);
            out.push_str(" in ");
            match iterable {
                ForIterable::Range { start, end } => {
                    out.push_str("range(");
                    format_expr(start, out);
                    out.push_str(", ");
                    format_expr(end, out);
                    out.push(')');
                }
                ForIterable::Expr(e) => format_expr(e, out),
            }
            out.push_str(" {\n");
            for s in body {
                format_stmt(s, level + 1, out);
            }
            indent_to(level, out);
            out.push_str("}\n");
        }
        Statement::FunctionDef { name, params, body, return_type, .. } => {
            out.push_str("fn ");
            out.push_str(name);
            out.push('(');
            for (i, p) in params.iter().enumerate() {
                if i > 0 { out.push_str(", "); }
                out.push_str(p);
            }
            out.push(')');
            if let Some(rt) = return_type {
                out.push_str(" -> ");
                out.push_str(rt);
            }
            out.push_str(" {\n");
            for s in body {
                format_stmt(s, level + 1, out);
            }
            indent_to(level, out);
            out.push_str("}\n");
        }
        Statement::Return(opt) => {
            out.push_str("return");
            if let Some(e) = opt {
                out.push(' ');
                format_expr(e, out);
            }
            out.push_str(";\n");
        }
        Statement::Break => out.push_str("break;\n"),
        Statement::Continue => out.push_str("continue;\n"),
        Statement::Import { module, alias } => {
            out.push_str("import \"");
            out.push_str(module);
            out.push('"');
            if let Some(a) = alias {
                out.push_str(" as ");
                out.push_str(a);
            }
            out.push_str(";\n");
        }
        Statement::Try { body, err_var, handler } => {
            out.push_str("try {\n");
            for s in body { format_stmt(s, level + 1, out); }
            indent_to(level, out);
            out.push_str("} catch ");
            out.push_str(err_var);
            out.push_str(" {\n");
            for s in handler { format_stmt(s, level + 1, out); }
            indent_to(level, out);
            out.push_str("}\n");
        }
        Statement::Match { scrutinee, arms } => {
            out.push_str("match ");
            format_expr(scrutinee, out);
            out.push_str(" {\n");
            for arm in arms {
                indent_to(level + 1, out);
                format_pattern(&arm.pattern, out);
                out.push_str(" => {\n");
                for s in &arm.body { format_stmt(s, level + 2, out); }
                indent_to(level + 1, out);
                out.push_str("}\n");
            }
            indent_to(level, out);
            out.push_str("}\n");
        }
    }
}

fn format_pattern(pat: &crate::ast::Pattern, out: &mut String) {
    use crate::ast::Pattern;
    match pat {
        Pattern::Wildcard => out.push('_'),
        Pattern::Bind(n) => out.push_str(n),
        Pattern::LitInt(n) => out.push_str(&n.to_string()),
        Pattern::LitFloat(f) => out.push_str(&format!("{}", f)),
        Pattern::LitString(s) => out.push_str(&format!("{:?}", s)),
        Pattern::LitBool(b) => out.push_str(if *b { "true" } else { "false" }),
        Pattern::LitNull => out.push_str("null"),
        Pattern::RangeInt(lo, hi) => out.push_str(&format!("{}..{}", lo, hi)),
        Pattern::RangeStr(lo, hi) => {
            out.push_str(&format!("\"{}\"..\"{}\"", lo, hi));
        }
        Pattern::Or(alts) => {
            for (i, p) in alts.iter().enumerate() {
                if i > 0 { out.push_str(" | "); }
                format_pattern(p, out);
            }
        }
        Pattern::Type(name) => out.push_str(name),
    }
}

fn format_expr(expr: &Expression, out: &mut String) {
    match expr {
        Expression::Number(n) => out.push_str(&n.to_string()),
        Expression::Float(f) => {
            // Keep the decimal point so re-parse doesn't collapse to int.
            let s = format!("{}", f);
            if s.contains('.') || s.contains('e') || s.contains('E') {
                out.push_str(&s);
            } else {
                out.push_str(&s);
                out.push_str(".0");
            }
        }
        Expression::String(s) => {
            out.push('"');
            for c in s.chars() {
                match c {
                    '\\' => out.push_str("\\\\"),
                    '"' => out.push_str("\\\""),
                    '\n' => out.push_str("\\n"),
                    '\t' => out.push_str("\\t"),
                    '\r' => out.push_str("\\r"),
                    _ => out.push(c),
                }
            }
            out.push('"');
        }
        Expression::Boolean(b) => out.push_str(if *b { "true" } else { "false" }),
        Expression::Variable(name) => out.push_str(name),
        Expression::Index { name, index } => {
            out.push_str(name);
            out.push('[');
            format_expr(index, out);
            out.push(']');
        }
        Expression::Array(items) => {
            out.push('[');
            for (i, e) in items.iter().enumerate() {
                if i > 0 { out.push_str(", "); }
                format_expr(e, out);
            }
            out.push(']');
        }
        Expression::Dict(pairs) => {
            out.push('{');
            for (i, (k, v)) in pairs.iter().enumerate() {
                if i > 0 { out.push_str(", "); }
                format_expr(k, out);
                out.push_str(": ");
                format_expr(v, out);
            }
            out.push('}');
        }
        Expression::Add(l, r) => format_binop(l, "+", r, out),
        Expression::Sub(l, r) => format_binop(l, "-", r, out),
        Expression::Mul(l, r) => format_binop(l, "*", r, out),
        Expression::Div(l, r) => format_binop(l, "/", r, out),
        Expression::Mod(l, r) => format_binop(l, "%", r, out),
        Expression::Eq(l, r) => format_binop(l, "==", r, out),
        Expression::Ne(l, r) => format_binop(l, "!=", r, out),
        Expression::Lt(l, r) => format_binop(l, "<", r, out),
        Expression::Le(l, r) => format_binop(l, "<=", r, out),
        Expression::Gt(l, r) => format_binop(l, ">", r, out),
        Expression::Ge(l, r) => format_binop(l, ">=", r, out),
        Expression::And(l, r) => format_binop(l, "and", r, out),
        Expression::Or(l, r) => format_binop(l, "or", r, out),
        Expression::Not(e) => {
            out.push_str("not ");
            format_expr(e, out);
        }
        Expression::BitAnd(l, r) => format_binop(l, "&", r, out),
        Expression::BitOr(l, r) => format_binop(l, "|", r, out),
        Expression::BitXor(l, r) => format_binop(l, "^", r, out),
        Expression::BitNot(e) => {
            out.push('~');
            format_expr(e, out);
        }
        Expression::Shl(l, r) => format_binop(l, "<<", r, out),
        Expression::Shr(l, r) => format_binop(l, ">>", r, out),
        Expression::Call { name, args, .. } => {
            out.push_str(name);
            out.push('(');
            for (i, a) in args.iter().enumerate() {
                if i > 0 { out.push_str(", "); }
                format_expr(a, out);
            }
            out.push(')');
        }
        Expression::Resonance(e) => { out.push_str("res("); format_expr(e, out); out.push(')'); }
        Expression::Fold(e) => { out.push_str("fold("); format_expr(e, out); out.push(')'); }
        Expression::Safe(inner) => {
            out.push_str("safe ");
            format_expr(inner, out);
        }
        Expression::Lambda { params, body } => {
            out.push_str("fn(");
            for (i, p) in params.iter().enumerate() {
                if i > 0 { out.push_str(", "); }
                out.push_str(p);
            }
            out.push_str(") {\n");
            for s in body {
                format_stmt(s, 1, out);
            }
            out.push('}');
        }
    }
}

fn format_binop(l: &Expression, op: &str, r: &Expression, out: &mut String) {
    out.push('(');
    format_expr(l, out);
    out.push(' ');
    out.push_str(op);
    out.push(' ');
    format_expr(r, out);
    out.push(')');
}
