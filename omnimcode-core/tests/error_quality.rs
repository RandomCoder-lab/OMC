//! Error-message and ergonomic-runtime tests for OMC.
//!
//! These lock in the "no trouble using OMC" pass: parser hints,
//! runtime did-you-mean, negative array indexing, helpful bounds
//! errors. Each test phrases an error message we *want* and asserts
//! the message contains the helpful tokens.

use omnimcode_core::interpreter::Interpreter;
use omnimcode_core::parser::Parser;

fn run(src: &str) -> Result<(), String> {
    // Append a top-level main() invocation so fn main() { ... } bodies
    // actually run during the test. Interpreter::execute only processes
    // top-level statements; a bare FunctionDef just registers, doesn't
    // call.
    let wrapped = format!("{}\nmain();\n", src);
    let mut p = Parser::new(&wrapped);
    let stmts = p.parse()?;
    let mut i = Interpreter::new();
    i.execute(stmts).map(|_| ())
}

fn parse_err(src: &str) -> String {
    let mut p = Parser::new(src);
    p.parse().unwrap_err()
}

// ---------- Reserved word as identifier ------------------------------------

#[test]
fn reserved_h_as_var_name_gives_actionable_error() {
    let err = parse_err("fn main() { h h = 1; }");
    assert!(err.contains("'h' is a reserved keyword"),
            "should name the reserved word, got: {}", err);
    assert!(err.contains("Try `hval`"),
            "should suggest an alternative, got: {}", err);
}

#[test]
fn reserved_fn_as_var_name_is_friendlier() {
    let err = parse_err("fn main() { h fn = 1; }");
    assert!(err.contains("'fn' is a reserved keyword"),
            "should name fn, got: {}", err);
}

// ---------- Assignment vs equality ------------------------------------------

#[test]
fn equals_in_expression_position_suggests_eq_eq() {
    // `if x = 5` is a classic typo — `=` should suggest `==`.
    let err = parse_err("fn main() { h x = 3; if x = 5 { return 1; } return 0; }");
    assert!(err.contains("Did you mean `==`?"),
            "should hint at ==, got: {}", err);
}

// ---------- Negative array indexing -----------------------------------------

#[test]
fn negative_index_via_arr_get_returns_last() {
    let src = "fn main() {
        h xs = [10, 20, 30, 40];
        h last = arr_get(xs, 0 - 1);
        if last != 40 { error(\"expected 40 got \" + to_string(last)); }
    }";
    run(src).unwrap();
}

#[test]
fn negative_index_via_subscript_returns_last() {
    let src = "fn main() {
        h xs = [10, 20, 30, 40];
        h last = xs[0 - 1];
        if last != 40 { error(\"expected 40 got \" + to_string(last)); }
    }";
    run(src).unwrap();
}

#[test]
fn out_of_bounds_error_includes_array_name_and_length() {
    let src = "fn main() {
        h xs = [1, 2, 3];
        h v = xs[99];
    }";
    let err = run(src).unwrap_err();
    assert!(err.contains("xs[99]") && err.contains("length 3"),
            "should name the array and report length, got: {}", err);
    // And hint at safe_arr_get for wrap-around access.
    assert!(err.contains("safe_arr_get"),
            "should hint at safe_arr_get, got: {}", err);
}

// ---------- Undefined variable did-you-mean --------------------------------

#[test]
fn undefined_variable_suggests_close_name() {
    let src = "fn main() {
        h hello = 42;
        return hellp;
    }";
    let err = run(src).unwrap_err();
    assert!(err.contains("Undefined variable") && err.contains("hellp"),
            "names the bad ident, got: {}", err);
    assert!(err.contains("did you mean") && err.contains("hello"),
            "suggests the close name, got: {}", err);
}

// ---------- Python-idiom builtins (smoke) ----------------------------------

#[test]
fn range_with_step_runs() {
    let src = "fn main() {
        h r = range(0, 10, 2);
        if arr_len(r) != 5 { error(\"len wrong\"); }
        if arr_get(r, 4) != 8 { error(\"value wrong\"); }
    }";
    run(src).unwrap();
}

#[test]
fn len_dispatches_on_dict() {
    let src = "fn main() {
        h d = dict_new();
        dict_set(d, \"a\", 1);
        dict_set(d, \"b\", 2);
        if len(d) != 2 { error(\"dict len\"); }
    }";
    run(src).unwrap();
}

#[test]
fn to_hex_and_from_hex_round_trip() {
    let src = "fn main() {
        if from_hex(to_hex(255)) != 255 { error(\"round trip 255\"); }
        if to_hex(16) != \"0x10\" { error(\"format 16\"); }
    }";
    run(src).unwrap();
}

// ---------- Wrong-container hints ------------------------------------------

#[test]
fn arr_get_called_on_dict_suggests_dict_get() {
    let src = "fn main() {
        h d = dict_new();
        dict_set(d, \"k\", 1);
        h v = arr_get(d, 0);
    }";
    let err = run(src).unwrap_err();
    assert!(err.contains("arr_get"), "names builtin: {}", err);
    assert!(err.contains("dict_get"), "suggests dict_get: {}", err);
    assert!(err.contains("got dict"), "reports received type: {}", err);
}

#[test]
fn dict_get_called_on_array_suggests_arr_get() {
    let src = "fn main() {
        h xs = [1, 2, 3];
        h v = dict_get(xs, \"k\");
    }";
    let err = run(src).unwrap_err();
    assert!(err.contains("dict_get"), "names builtin: {}", err);
    assert!(err.contains("arr_get"), "suggests arr_get: {}", err);
    assert!(err.contains("got array"), "reports received type: {}", err);
}

// ---------- Calling non-function -------------------------------------------

#[test]
fn calling_an_int_as_function_is_friendlier() {
    // `call(value, args)` routes through the first-class callable path.
    let src = "fn main() {
        h x = 42;
        call(x, []);
    }";
    let err = run(src).unwrap_err();
    assert!(err.contains("Cannot call") && err.contains("int"),
            "should name the type, got: {}", err);
}

#[test]
fn getenv_with_default_returns_default_when_unset() {
    let src = "fn main() {
        h v = getenv(\"OMC_TEST_DEFINITELY_NOT_SET_XYZZY\", \"backup\");
        if v != \"backup\" { error(\"fallback\"); }
    }";
    run(src).unwrap();
}
