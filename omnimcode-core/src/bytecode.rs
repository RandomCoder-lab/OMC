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

    // Typed fast-path arithmetic: skip the runtime is_float() check when the
    // compiler proves both operands are int-typed. Emitted by Phase M's HIR.
    AddInt,
    SubInt,
    MulInt,
    // Typed fast-path arithmetic for floats (both operands provably float).
    AddFloat,
    SubFloat,
    MulFloat,

    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,

    And,
    Or,
    Not,

    // Bitwise (operate on integer values; floats are truncated to i64)
    BitAnd,
    BitOr,
    BitXor,
    BitNot,
    Shl,
    Shr,

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
    /// Mutating array push: pop one value off the stack and append it
    /// to the named array variable in the current scope. Emitted by the
    /// compiler when it sees `arr_push(tokens, expr)` with a literal
    /// variable as the first argument. Bypasses vm_call_builtin's
    /// synthetic-arg shim, which would otherwise lose the mutation.
    ArrPushNamed(String),
    /// Mutating array store: pop value, pop index, store at named array's
    /// index. Same rationale as ArrPushNamed.
    ArrSetNamed(String),
    /// H.5.2: self-healing mutating array store. Pop value, pop raw_idx,
    /// fold raw_idx onto the nearest Fibonacci attractor, Euclidean-mod by
    /// arr_len, then store at the healed index. Out-of-bounds writes
    /// become attractor-landing in-bounds writes. Same name-on-opcode trick
    /// as ArrSetNamed — required so the mutation propagates back through
    /// the VM scope instead of getting lost in vm_call_builtin's shim.
    SafeArrSetNamed(String),

    // Special harmonic operations (short-circuit to built-in semantics
    // without the call overhead — these are the hot ones).
    Resonance,             // pop x, push res(x) as HFloat
    Fold1,                 // pop x, push fold(x) as HInt (Fibonacci snap)
    IsFibonacci,           // pop x, push 1/0 (HInt) if x is Fibonacci
    Fibonacci,             // pop n, push fibonacci(n) as HInt
    ArrayLen,              // pop array, push HInt(len)
    HimScore,              // pop x, push HInt's HIM score as HFloat

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
    /// Optional type annotation per parameter ("int" / "float" / "string" / "bool" / etc.)
    /// Phase M: used by the compiler to specialize arithmetic on known-int args.
    pub param_types: Vec<Option<String>>,
    /// Optional return-type annotation. Used by the type-inference helper
    /// when a call's return type is statically known.
    pub return_type: Option<String>,
    pub ops: Vec<Op>,
    pub constants: Vec<Const>,
    /// Phase Q inline call cache: one Cell per op. The VM populates the
    /// matching slot on the first execution of an `Op::Call` with the
    /// resolved kind (user-defined vs built-in), letting subsequent passes
    /// skip the HashMap probe. 0 = uncached, 1 = user, 2 = built-in.
    ///
    /// Stored as `Cell<u8>` so it can be mutated through an immutable
    /// borrow (typical for monomorphic ICs). Cell<u8> is Copy + Clone so
    /// the surrounding struct stays cleanly cloneable.
    pub call_cache: Vec<std::cell::Cell<u8>>,
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
                param_types: Vec::new(),
                return_type: None,
                ops: Vec::new(),
                constants: Vec::new(),
                call_cache: Vec::new(),
            },
            functions: std::collections::HashMap::new(),
        }
    }
}
