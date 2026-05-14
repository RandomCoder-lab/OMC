// omnimcode-core/src/bytecode.rs — Bytecode IR for the OMNIcode VM
//
// Coexists with the tree-walk interpreter. The VM is selected at the
// CLI / lib level (e.g., env var OMC_VM=1) — when not selected, the
// tree-walk path remains the default. This keeps the language semantics
// in one place (the interpreter) and uses the VM purely as an
// alternative dispatch.

use crate::value::Value;

/// Constant pool entry. Strings, floats, and ints all use it so opcodes
/// only need a small index payload instead of inline literals.
#[derive(Clone, Debug)]
pub enum Const {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    Null,
}

impl Const {
    pub fn to_value(&self) -> Value {
        match self {
            Const::Int(n) => Value::HInt(crate::value::HInt::new(*n)),
            Const::Float(f) => Value::HFloat(*f),
            Const::Str(s) => Value::String(s.clone()),
            Const::Bool(b) => Value::Bool(*b),
            Const::Null => Value::Null,
        }
    }
}

/// Bytecode opcodes. Designed to be cheap to dispatch — no allocation
/// for the common int-arithmetic and load/store paths.
#[derive(Clone, Debug)]
pub enum Op {
    // Stack manipulation
    LoadConst(usize),      // push constants[idx]
    Pop,

    // Variables
    LoadVar(String),       // push value of variable
    StoreVar(String),      // pop and store
    LoadParam(usize),      // push param at slot N (call frame)

    // Arithmetic / comparison (operate on top two of stack)
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Neg,

    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,

    And,
    Or,
    Not,

    // Control flow
    Jump(i32),             // relative jump
    JumpIfFalse(i32),
    JumpIfTrue(i32),

    // Calls
    /// Call a function by name with N args already on the stack.
    /// Result pushed; works for both built-ins and user-defined.
    Call(String, usize),

    /// Return from current function. Pops one value as the return.
    Return,
    /// Return Null (no expression).
    ReturnNull,

    // Arrays
    NewArray(usize),       // pop N items into a new array, push
    ArrayIndex,            // pop index, pop array, push array[index]
    ArrayIndexAssign(String), // pop value, pop index, assign array_var[idx] = value

    // Special harmonic operations (short-circuit to built-in semantics
    // without the call overhead — these are the hot ones).
    Resonance,             // pop x, push res(x) as HFloat
    Fold1,                 // pop x, push fold(x) as HInt (Fibonacci snap)

    // Print (statement form)
    Print,                 // pop and println

    // No-op (filled by patcher when fixing up jump offsets, etc.)
    Nop,
}

/// A compiled function body.
#[derive(Clone, Debug)]
pub struct CompiledFunction {
    pub name: String,
    pub params: Vec<String>,
    pub ops: Vec<Op>,
    pub constants: Vec<Const>,
}

/// A compiled module / program.
#[derive(Clone, Debug)]
pub struct Module {
    pub main: CompiledFunction,
    pub functions: std::collections::HashMap<String, CompiledFunction>,
}

impl Default for Module {
    fn default() -> Self {
        Module {
            main: CompiledFunction {
                name: "__main__".to_string(),
                params: Vec::new(),
                ops: Vec::new(),
                constants: Vec::new(),
            },
            functions: std::collections::HashMap::new(),
        }
    }
}
