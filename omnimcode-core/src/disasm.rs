// omnimcode-core/src/disasm.rs — pretty-print a CompiledFunction as a
// human-readable bytecode listing. Triggered via OMC_DISASM=1 from
// main.rs, or callable directly for testing.

use crate::bytecode::*;

/// Render a single Op into its readable mnemonic form. For jumps the
/// caller patches the resolved target after the fact (we don't know our
/// own offset in this fn).
fn op_mnemonic(op: &Op, ip: usize, constants: &[Const]) -> String {
    match op {
        Op::Nop => "NOP".to_string(),
        Op::LoadConst(idx) => {
            let preview = constants
                .get(*idx)
                .map(|c| format!(" ; {}", short_const(c)))
                .unwrap_or_default();
            format!("LOAD_CONST   {}{}", idx, preview)
        }
        Op::Pop => "POP".to_string(),
        Op::LoadVar(name) => format!("LOAD_VAR     {}", name),
        Op::StoreVar(name) => format!("STORE_VAR    {}", name),
        Op::LoadParam(slot) => format!("LOAD_PARAM   {}", slot),

        Op::Add => "ADD".to_string(),
        Op::Sub => "SUB".to_string(),
        Op::Mul => "MUL".to_string(),
        Op::Div => "DIV".to_string(),
        Op::Mod => "MOD".to_string(),
        Op::Neg => "NEG".to_string(),

        Op::AddInt => "ADD_INT".to_string(),
        Op::SubInt => "SUB_INT".to_string(),
        Op::MulInt => "MUL_INT".to_string(),
        Op::AddFloat => "ADD_FLOAT".to_string(),
        Op::SubFloat => "SUB_FLOAT".to_string(),
        Op::MulFloat => "MUL_FLOAT".to_string(),
        Op::DivFloat => "DIV_FLOAT".to_string(),

        Op::Eq => "EQ".to_string(),
        Op::Ne => "NE".to_string(),
        Op::Lt => "LT".to_string(),
        Op::Le => "LE".to_string(),
        Op::Gt => "GT".to_string(),
        Op::Ge => "GE".to_string(),
        Op::EqFloat => "EQ_FLOAT".to_string(),
        Op::NeFloat => "NE_FLOAT".to_string(),
        Op::LtFloat => "LT_FLOAT".to_string(),
        Op::LeFloat => "LE_FLOAT".to_string(),
        Op::GtFloat => "GT_FLOAT".to_string(),
        Op::GeFloat => "GE_FLOAT".to_string(),

        Op::And => "AND".to_string(),
        Op::Or => "OR".to_string(),
        Op::Not => "NOT".to_string(),

        Op::BitAnd => "BIT_AND".to_string(),
        Op::BitOr => "BIT_OR".to_string(),
        Op::BitXor => "BIT_XOR".to_string(),
        Op::BitNot => "BIT_NOT".to_string(),
        Op::Shl => "SHL".to_string(),
        Op::Shr => "SHR".to_string(),

        Op::Jump(off) => format!("JUMP         {:+}     ; -> {}", off, jump_target(ip, *off)),
        Op::JumpIfFalse(off) => format!(
            "JUMP_IF_FALSE {:+}    ; -> {}",
            off,
            jump_target(ip, *off)
        ),
        Op::JumpIfTrue(off) => format!(
            "JUMP_IF_TRUE  {:+}    ; -> {}",
            off,
            jump_target(ip, *off)
        ),

        Op::Call(name, argc) => format!("CALL         {}/{}", name, argc),
        Op::Return => "RETURN".to_string(),
        Op::ReturnNull => "RETURN_NULL".to_string(),

        Op::NewArray(n) => format!("NEW_ARRAY    {}", n),
        Op::NewDict(n) => format!("NEW_DICT     {}", n),
        Op::DictSetNamed(name) => format!("DICT_SET_NAMED  {}", name),
        Op::DictDelNamed(name) => format!("DICT_DEL_NAMED  {}", name),
        Op::ExecStmt(_) => "EXEC_STMT       <ast>".to_string(),
        Op::ArrayIndex => "ARRAY_INDEX".to_string(),
        Op::ArrayIndexAssign(name) => format!("ARRAY_INDEX_ASSIGN {}", name),
        Op::ArrPushNamed(name) => format!("ARR_PUSH_NAMED  {}", name),
        Op::ArrSetNamed(name) => format!("ARR_SET_NAMED   {}", name),
        Op::SafeArrSetNamed(name) => format!("SAFE_ARR_SET_NAMED {}", name),
        Op::Lambda(name) => format!("LAMBDA          {}", name),
        Op::AssignVar(name) => format!("ASSIGN_VAR      {}", name),

        Op::Resonance => "RESONANCE".to_string(),
        Op::Fold1 => "FOLD".to_string(),
        Op::IsFibonacci => "IS_FIB".to_string(),
        Op::Fibonacci => "FIB".to_string(),
        Op::ArrayLen => "ARR_LEN".to_string(),
        Op::HimScore => "HIM".to_string(),

        Op::Print => "PRINT".to_string(),
    }
}

fn jump_target(from_ip: usize, offset: i32) -> i64 {
    (from_ip as i64) + 1 + (offset as i64)
}

fn short_const(c: &Const) -> String {
    match c {
        Const::Int(n) => n.to_string(),
        Const::Float(f) => format!("{:.6}", f),
        Const::Str(s) => {
            if s.len() > 30 {
                format!("\"{}...\"", &s[..30])
            } else {
                format!("\"{}\"", s)
            }
        }
        Const::Bool(b) => b.to_string(),
        Const::Null => "null".to_string(),
    }
}

pub fn disassemble_function(func: &CompiledFunction) -> String {
    let mut out = String::new();
    // Header: name, param/return type signature, op + const counts.
    let sig_params: Vec<String> = func
        .params
        .iter()
        .zip(func.param_types.iter())
        .map(|(name, ty)| match ty {
            Some(t) => format!("{}: {}", name, t),
            None => name.clone(),
        })
        .collect();
    let ret = func
        .return_type
        .as_deref()
        .map(|t| format!(" -> {}", t))
        .unwrap_or_default();
    out.push_str(&format!(
        "fn {}({}){}    [{} ops, {} consts]\n",
        func.name,
        sig_params.join(", "),
        ret,
        func.ops.len(),
        func.constants.len(),
    ));
    out.push_str(&"-".repeat(72));
    out.push('\n');

    // Constants pool (only show if non-trivial).
    if !func.constants.is_empty() {
        out.push_str("  constants:\n");
        for (i, c) in func.constants.iter().enumerate() {
            out.push_str(&format!("    [{}] {}\n", i, short_const(c)));
        }
        out.push('\n');
    }

    // Ops with offsets.
    for (i, op) in func.ops.iter().enumerate() {
        out.push_str(&format!("  {:04}: {}\n", i, op_mnemonic(op, i, &func.constants)));
    }
    out
}

pub fn disassemble_module(module: &Module) -> String {
    let mut out = String::new();
    out.push_str("=== OMNIcode Bytecode Disassembly ===\n\n");
    out.push_str(&disassemble_function(&module.main));
    out.push('\n');
    // Sort function names for stable output.
    let mut fn_names: Vec<&String> = module.functions.keys().collect();
    fn_names.sort();
    for name in fn_names {
        out.push_str(&disassemble_function(&module.functions[name]));
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::compile_program;
    use crate::parser::Parser;

    fn compile(src: &str) -> Module {
        let mut parser = Parser::new(src);
        let stmts = parser.parse().unwrap();
        compile_program(&stmts).unwrap()
    }

    #[test]
    fn disassembly_renders_a_simple_program() {
        let module = compile("h x = 89; print(x);");
        let s = disassemble_module(&module);
        assert!(s.contains("LOAD_CONST"));
        assert!(s.contains("STORE_VAR"));
        assert!(s.contains("PRINT"));
    }

    #[test]
    fn disassembly_shows_typed_opcodes() {
        // With both operands int, the compiler emits ADD_INT not ADD.
        let module = compile("fn add(x: int, y: int) -> int { return x + y; }");
        let s = disassemble_module(&module);
        assert!(s.contains("ADD_INT"), "expected ADD_INT in: {}", s);
    }

    #[test]
    fn disassembly_resolves_jumps() {
        let module = compile("h i = 0; while i < 5 { i = i + 1; }");
        let s = disassemble_module(&module);
        assert!(s.contains("JUMP"), "expected JUMP in: {}", s);
    }
}
