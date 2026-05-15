// omnimcode-core/src/python_embed.rs
//
// Embeds CPython into OMC. Only compiled when the `python-embed`
// feature is on. Exposes a small `py_*` builtin family that lets OMC
// programs reach the entire Python ecosystem — numpy, pandas, requests,
// any pip-installable library.
//
// Architecture: PyObjects can't be stored in OMC's Value enum (no
// pointer types in the language), so we keep a process-level registry
// that maps integer handles → PyObject. OMC code holds the handle as
// a Value::HInt; py_call / py_get look up the PyObject. The registry
// uses a thread_local RefCell — pyo3 already requires single-threaded
// access via Python::with_gil, so no extra synchronisation needed.
//
// Conversion rules (Python → OMC, automatic):
//   int          → Value::HInt
//   float        → Value::HFloat
//   str          → Value::String
//   bool         → Value::Bool
//   None         → Value::Null
//   list, tuple  → Value::Array (recursive)
//   dict (str-k) → Value::Dict (recursive)
//   numpy ndarray (any-D)        → Value::Array (via .tolist())
//   anything else                → opaque handle (Value::HInt registry id)

use crate::interpreter::{with_active_interp, Interpreter};
use crate::value::{HArray, HInt, Value};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};
use std::cell::RefCell;
use std::collections::HashMap;

/// Handle IDs start at this offset so they never collide with
/// regular OMC integers used as data. Without this, `[1, 2, 3]`
/// would be mistaken for handle 1 and unwrapped to a PyObject —
/// breaking every numeric array passed back into Python.
///
/// 10^15 leaves plenty of headroom for real numeric data
/// (Python ints up to ~9 * 10^18 still round-trip fine via
/// extract::<i64> first; only the value itself would alias).
const HANDLE_BASE: i64 = 1_000_000_000_000_000;

thread_local! {
    /// Process-local registry of PyObjects held by OMC code via
    /// integer handles. Cleared with `py_clear_registry()`.
    static PY_REGISTRY: RefCell<HashMap<i64, PyObject>> = RefCell::new(HashMap::new());
    static NEXT_HANDLE: RefCell<i64> = const { RefCell::new(HANDLE_BASE) };
}

fn alloc_handle() -> i64 {
    NEXT_HANDLE.with(|n| {
        let mut x = n.borrow_mut();
        let id = *x;
        *x += 1;
        id
    })
}

/// Cheap test before doing the registry lookup. Avoids paying a
/// HashMap probe on every numeric value going Python-ward.
#[inline]
fn looks_like_handle(n: i64) -> bool {
    n >= HANDLE_BASE
}

fn store_handle(obj: PyObject) -> i64 {
    let id = alloc_handle();
    PY_REGISTRY.with(|r| r.borrow_mut().insert(id, obj));
    id
}

/// Caller must hold the GIL (we use the `py` token to clone_ref).
fn fetch_handle(py: Python<'_>, id: i64) -> Option<PyObject> {
    PY_REGISTRY.with(|r| r.borrow().get(&id).map(|o| o.clone_ref(py)))
}

fn is_handle(id: i64) -> bool {
    PY_REGISTRY.with(|r| r.borrow().contains_key(&id))
}

/// OMC Value → Python object (pyo3 0.21 API: `.to_object(py)` and
/// `.into_py(py)` are the canonical conversions).
fn omc_to_py(py: Python<'_>, v: &Value) -> PyResult<PyObject> {
    match v {
        Value::HInt(h) => {
            // Disambiguate: only large IDs (above HANDLE_BASE) are
            // handle candidates. This keeps regular numeric data
            // round-tripping correctly — `[1, 2, 3]` stays as a
            // list of ints even though handle id 1 may exist.
            if looks_like_handle(h.value) && is_handle(h.value) {
                if let Some(obj) = fetch_handle(py, h.value) {
                    return Ok(obj);
                }
            }
            Ok(h.value.into_py(py))
        }
        Value::HFloat(f) => Ok(f.into_py(py)),
        Value::String(s) => Ok(s.into_py(py)),
        Value::Bool(b) => Ok(b.into_py(py)),
        Value::Null => Ok(py.None()),
        Value::Array(arr) => {
            let items = arr.items.borrow();
            let list = PyList::empty_bound(py);
            for item in items.iter() {
                list.append(omc_to_py(py, item)?)?;
            }
            Ok(list.into_py(py))
        }
        Value::Dict(d) => {
            let dict = PyDict::new_bound(py);
            for (k, val) in d.borrow().iter() {
                dict.set_item(k, omc_to_py(py, val)?)?;
            }
            Ok(dict.into_py(py))
        }
        Value::Function { .. } => Err(pyo3::exceptions::PyTypeError::new_err(
            "cannot convert OMC Function to Python (no callback bridge yet)",
        )),
        Value::Singularity { numerator, denominator, context } => Ok(format!(
            "Singularity({}/{}, ctx={})",
            numerator, denominator, context
        )
        .into_py(py)),
        Value::Circuit(_) => Err(pyo3::exceptions::PyTypeError::new_err(
            "cannot convert OMC Circuit to Python",
        )),
    }
}

/// Python → OMC. Anything not directly representable becomes an
/// opaque handle the user can pass back via py_call / py_get.
fn py_to_omc(py: Python<'_>, obj: &Bound<PyAny>) -> Value {
    // bool BEFORE int (bool subclasses int in Python).
    if let Ok(b) = obj.extract::<bool>() {
        return Value::Bool(b);
    }
    if obj.is_none() {
        return Value::Null;
    }
    if let Ok(n) = obj.extract::<i64>() {
        return Value::HInt(HInt::new(n));
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Value::HFloat(f);
    }
    // Strict string check: only convert if obj is actually a PyString.
    // extract::<String> would call str() on anything (DataFrames, etc.)
    // and silently strip the entire object's repr — disastrous for
    // pandas/numpy interop where users want to keep the handle.
    if let Ok(s) = obj.downcast::<PyString>() {
        return Value::String(s.to_string());
    }
    if let Ok(list) = obj.downcast::<PyList>() {
        let items: Vec<Value> = list.iter().map(|item| py_to_omc(py, &item)).collect();
        return Value::Array(HArray::from_vec(items));
    }
    if let Ok(tup) = obj.downcast::<PyTuple>() {
        let items: Vec<Value> = tup.iter().map(|item| py_to_omc(py, &item)).collect();
        return Value::Array(HArray::from_vec(items));
    }
    if let Ok(d) = obj.downcast::<PyDict>() {
        let mut map = std::collections::BTreeMap::new();
        for (k, v) in d.iter() {
            let key = k.str().map(|s| s.to_string()).unwrap_or_else(|_| "?".to_string());
            map.insert(key, py_to_omc(py, &v));
        }
        return Value::dict_from(map);
    }
    // numpy.ndarray (any rank) — convert via .tolist() and recurse.
    if let Ok(tolist) = obj.getattr("tolist") {
        if let Ok(listed) = tolist.call0() {
            return py_to_omc(py, &listed);
        }
    }
    // Anything else: opaque handle.
    let id = store_handle(obj.clone().unbind());
    Value::HInt(HInt::new(id))
}

/// OMC array of args → owned PyTuple ready for .call1 / .call_method1.
/// Auto-wraps scalars: py_call(h, "f", x) is shorthand for [x].
fn arr_to_py_tuple<'py>(py: Python<'py>, arr_arg: &Value) -> PyResult<Bound<'py, PyTuple>> {
    let items: Vec<PyObject> = match arr_arg {
        Value::Array(arr) => {
            let inner = arr.items.borrow();
            let mut out = Vec::with_capacity(inner.len());
            for v in inner.iter() {
                out.push(omc_to_py(py, v)?);
            }
            out
        }
        other => vec![omc_to_py(py, other)?],
    };
    Ok(PyTuple::new_bound(py, items))
}

/// Register the py_* builtin family on `interp`. After this:
///
///   py_import("numpy")            → handle
///   py_call(handle, "method", a)  → Value
///   py_get(handle, "attr")        → handle / scalar Value
///   py_call_fn(handle, args)      → Value         (call handle as fn)
///   py_eval("expr")               → Value         (run a Python expression)
///   py_exec("code")               → null          (run Python statements)
///   py_repr(handle)               → string
///   py_clear_registry()           → null
///
/// Args are converted automatically; numpy arrays come back as
/// nested OMC arrays. Anything not directly representable becomes
/// an opaque handle that round-trips correctly.
pub fn register_python_builtins(interp: &mut Interpreter) {
    interp.register_builtin("py_import", |args| {
        if args.is_empty() {
            return Err("py_import requires (module_name)".to_string());
        }
        let name = args[0].to_display_string();
        Python::with_gil(|py| {
            let module = py
                .import_bound(name.as_str())
                .map_err(|e| format!("py_import({}): {}", name, e))?;
            Ok(Value::HInt(HInt::new(store_handle(module.into_py(py)))))
        })
    });

    interp.register_builtin("py_call", |args| {
        if args.len() < 2 {
            return Err("py_call requires (handle, method_name, args?)".to_string());
        }
        let handle = args[0].to_int();
        let method = args[1].to_display_string();
        let call_args = args.get(2).cloned().unwrap_or(Value::Array(HArray::new()));
        Python::with_gil(|py| {
            let obj = fetch_handle(py, handle)
                .ok_or_else(|| format!("py_call: invalid handle {}", handle))?;
            let bound = obj.bind(py);
            let tuple = arr_to_py_tuple(py, &call_args)
                .map_err(|e| format!("py_call: arg conversion failed: {}", e))?;
            let result = bound
                .call_method1(method.as_str(), tuple)
                .map_err(|e| format!("py_call({}): {}", method, e))?;
            Ok(py_to_omc(py, &result))
        })
    });

    interp.register_builtin("py_get", |args| {
        if args.len() < 2 {
            return Err("py_get requires (handle, attr_name)".to_string());
        }
        let handle = args[0].to_int();
        let attr = args[1].to_display_string();
        Python::with_gil(|py| {
            let obj = fetch_handle(py, handle)
                .ok_or_else(|| format!("py_get: invalid handle {}", handle))?;
            let bound = obj.bind(py);
            let result = bound
                .getattr(attr.as_str())
                .map_err(|e| format!("py_get({}): {}", attr, e))?;
            Ok(py_to_omc(py, &result))
        })
    });

    interp.register_builtin("py_call_fn", |args| {
        if args.is_empty() {
            return Err("py_call_fn requires (handle, args?)".to_string());
        }
        let handle = args[0].to_int();
        let call_args = args.get(1).cloned().unwrap_or(Value::Array(HArray::new()));
        Python::with_gil(|py| {
            let obj = fetch_handle(py, handle)
                .ok_or_else(|| format!("py_call_fn: invalid handle {}", handle))?;
            let bound = obj.bind(py);
            let tuple = arr_to_py_tuple(py, &call_args)
                .map_err(|e| format!("py_call_fn: arg conversion failed: {}", e))?;
            let result = bound
                .call1(tuple)
                .map_err(|e| format!("py_call_fn: {}", e))?;
            Ok(py_to_omc(py, &result))
        })
    });

    // ---- py_call_kw / py_call_fn_kw -----------------------------------
    // Same as py_call / py_call_fn but accept an OMC dict as a final
    // kwargs argument. Required for Python APIs like sklearn that
    // distinguish positional arrays from named scalars
    // (`train_test_split(X, y, test_size=0.3)`).
    // ---- py_call_raw: like py_call but ALWAYS returns a handle ------
    // Skip the py_to_omc auto-conversion. Useful when chaining ops
    // on objects that would otherwise auto-collapse (pandas Series
    // → OMC array, dict subclasses → OMC dict). The user explicitly
    // wants to keep the Python object alive for further py_call.
    interp.register_builtin("py_call_raw", |args| {
        if args.len() < 2 {
            return Err("py_call_raw requires (handle, method, args?)".to_string());
        }
        let handle = args[0].to_int();
        let method = args[1].to_display_string();
        let call_args = args.get(2).cloned().unwrap_or(Value::Array(HArray::new()));
        Python::with_gil(|py| {
            let obj = fetch_handle(py, handle)
                .ok_or_else(|| format!("py_call_raw: invalid handle {}", handle))?;
            let bound = obj.bind(py);
            let tuple = arr_to_py_tuple(py, &call_args)
                .map_err(|e| format!("py_call_raw: arg conversion failed: {}", e))?;
            let result = bound
                .call_method1(method.as_str(), tuple)
                .map_err(|e| format!("py_call_raw({}): {}", method, e))?;
            // Force handle — no py_to_omc.
            Ok(Value::HInt(HInt::new(store_handle(result.into_py(py)))))
        })
    });

    interp.register_builtin("py_call_kw", |args| {
        if args.len() < 4 {
            return Err("py_call_kw requires (handle, method, args, kwargs)".to_string());
        }
        let handle = args[0].to_int();
        let method = args[1].to_display_string();
        let pos_args = args[2].clone();
        let kwargs_v = args[3].clone();
        Python::with_gil(|py| {
            let obj = fetch_handle(py, handle)
                .ok_or_else(|| format!("py_call_kw: invalid handle {}", handle))?;
            let bound = obj.bind(py);
            let tuple = arr_to_py_tuple(py, &pos_args)
                .map_err(|e| format!("py_call_kw: pos arg conversion: {}", e))?;
            let kwargs = match &kwargs_v {
                Value::Dict(d) => {
                    let py_d = PyDict::new_bound(py);
                    for (k, v) in d.borrow().iter() {
                        py_d.set_item(k, omc_to_py(py, v).map_err(|e|
                            format!("py_call_kw: kwarg {}: {}", k, e))?)
                            .map_err(|e| format!("py_call_kw: set kwarg {}: {}", k, e))?;
                    }
                    Some(py_d)
                }
                Value::Null => None,
                _ => return Err("py_call_kw: kwargs must be a dict or null".to_string()),
            };
            let result = bound
                .call_method(method.as_str(), tuple, kwargs.as_ref())
                .map_err(|e| format!("py_call_kw({}): {}", method, e))?;
            Ok(py_to_omc(py, &result))
        })
    });

    interp.register_builtin("py_call_fn_kw", |args| {
        if args.len() < 3 {
            return Err("py_call_fn_kw requires (handle, args, kwargs)".to_string());
        }
        let handle = args[0].to_int();
        let pos_args = args[1].clone();
        let kwargs_v = args[2].clone();
        Python::with_gil(|py| {
            let obj = fetch_handle(py, handle)
                .ok_or_else(|| format!("py_call_fn_kw: invalid handle {}", handle))?;
            let bound = obj.bind(py);
            let tuple = arr_to_py_tuple(py, &pos_args)
                .map_err(|e| format!("py_call_fn_kw: pos arg conversion: {}", e))?;
            let kwargs = match &kwargs_v {
                Value::Dict(d) => {
                    let py_d = PyDict::new_bound(py);
                    for (k, v) in d.borrow().iter() {
                        py_d.set_item(k, omc_to_py(py, v).map_err(|e|
                            format!("py_call_fn_kw: kwarg {}: {}", k, e))?)
                            .map_err(|e| format!("py_call_fn_kw: set kwarg {}: {}", k, e))?;
                    }
                    Some(py_d)
                }
                Value::Null => None,
                _ => return Err("py_call_fn_kw: kwargs must be a dict or null".to_string()),
            };
            let result = bound
                .call(tuple, kwargs.as_ref())
                .map_err(|e| format!("py_call_fn_kw: {}", e))?;
            Ok(py_to_omc(py, &result))
        })
    });

    interp.register_builtin("py_eval", |args| {
        if args.is_empty() {
            return Err("py_eval requires (code_string)".to_string());
        }
        let code = args[0].to_display_string();
        Python::with_gil(|py| {
            let cstr = std::ffi::CString::new(code.as_str())
                .map_err(|e| format!("py_eval: {}", e))?;
            let result = py
                .eval_bound(cstr.to_str().unwrap(), None, None)
                .map_err(|e| format!("py_eval: {}", e))?;
            Ok(py_to_omc(py, &result))
        })
    });

    interp.register_builtin("py_exec", |args| {
        if args.is_empty() {
            return Err("py_exec requires (code_string)".to_string());
        }
        let code = args[0].to_display_string();
        Python::with_gil(|py| {
            let cstr = std::ffi::CString::new(code.as_str())
                .map_err(|e| format!("py_exec: {}", e))?;
            py.run_bound(cstr.to_str().unwrap(), None, None)
                .map_err(|e| format!("py_exec: {}", e))?;
            Ok(Value::Null)
        })
    });

    interp.register_builtin("py_repr", |args| {
        if args.is_empty() {
            return Err("py_repr requires (handle)".to_string());
        }
        let handle = args[0].to_int();
        Python::with_gil(|py| {
            let obj = fetch_handle(py, handle)
                .ok_or_else(|| format!("py_repr: invalid handle {}", handle))?;
            let bound = obj.bind(py);
            let r = bound.repr().map_err(|e| format!("py_repr: {}", e))?;
            Ok(Value::String(r.to_string()))
        })
    });

    interp.register_builtin("py_clear_registry", |_args| {
        PY_REGISTRY.with(|r| r.borrow_mut().clear());
        Ok(Value::Null)
    });

    // ---- py_fetch_text(url) -> string -------------------------------
    // Convenience: HTTP GET via embedded Python `requests`. Returns
    // body as string on 2xx, errors on anything else. Used internally
    // by `omc --install` so we don't need a separate Rust HTTP crate.
    interp.register_builtin("py_fetch_text", |args| {
        if args.is_empty() {
            return Err("py_fetch_text requires (url)".to_string());
        }
        let url = args[0].to_display_string();
        let body = fetch_url(&url)
            .map_err(|e| format!("py_fetch_text({}): {}", url, e))?;
        Ok(Value::String(body))
    });

    // ---- py_callback("omc_fn_name") -> handle (Python callable) -------
    // Returns a Python callable that, when invoked from Python with
    // positional args, calls back into OMC's `omc_fn_name` with the
    // converted args and returns the converted result. Enables the
    // df.apply(omc_fn) style.
    //
    // Lifecycle: the Python callable is valid only while the OMC
    // interpreter that created it is still on the call stack — i.e.
    // for the duration of the OMC program. Calling a stale callback
    // after the interp is destroyed is an error (the thread_local
    // pointer is null).
    interp.register_builtin("py_callback", |args| {
        if args.is_empty() {
            return Err("py_callback requires (omc_fn_name)".to_string());
        }
        let fn_name = args[0].to_display_string();
        Python::with_gil(|py| {
            let cb = OmcCallback { fn_name };
            let py_obj = Py::new(py, cb)
                .map_err(|e| format!("py_callback: pyclass alloc failed: {}", e))?;
            let id = store_handle(py_obj.into_any());
            Ok(Value::HInt(HInt::new(id)))
        })
    });
}

/// PyClass that wraps an OMC function name and exposes it as a
/// Python callable. When Python invokes `cb(*args)`, the __call__
/// method converts each arg to an OMC Value, dispatches to the
/// OMC function via the active interpreter, and converts the
/// result back to a PyObject.
#[pyclass]
struct OmcCallback {
    fn_name: String,
}

#[pymethods]
impl OmcCallback {
    /// Python __call__ entry point. PyO3 maps to `cb(*args)` from
    /// Python code. We collect the args via *PyTuple, convert each
    /// to a Value, run the OMC fn, return the converted result.
    #[pyo3(signature = (*args))]
    fn __call__(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
    ) -> PyResult<PyObject> {
        // Convert each Python positional arg to an OMC Value.
        let mut omc_args: Vec<Value> = Vec::with_capacity(args.len());
        for item in args.iter() {
            omc_args.push(py_to_omc(py, &item));
        }
        // Dispatch into the live interp.
        let fn_name = self.fn_name.clone();
        let result = with_active_interp(|interp| {
            interp.call_function_with_values(&fn_name, &omc_args)
        });
        let v = match result {
            None => return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "OmcCallback('{}'): no active OMC interpreter — \
                 callback invoked outside the OMC call that created it",
                fn_name
            ))),
            Some(Err(e)) => return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "OmcCallback('{}'): {}",
                fn_name, e
            ))),
            Some(Ok(v)) => v,
        };
        // omc_to_py returns Bound<'py, PyAny> — propagate.
        omc_to_py(py, &v)
    }

    fn __repr__(&self) -> String {
        format!("<OmcCallback '{}'>", self.fn_name)
    }
}

// ===========================================================================
// Package manager helpers — used by `omc --install` from main.rs.
//
// We do the HTTP fetch and TOML parse via embedded Python (`requests`
// + `tomllib`) rather than pulling in Rust HTTP/TOML crates. The
// dependency model is already "Python is always on" — leaning on it
// for tooling avoids dep bloat and proves the integration works for
// our own infrastructure.
// ===========================================================================

/// HTTP GET via embedded Python's `requests`. Returns the response
/// body on 2xx; Err on connection failure, non-2xx, or a missing
/// `requests` install.
pub fn fetch_url(url: &str) -> Result<String, String> {
    Python::with_gil(|py| {
        let requests = py
            .import_bound("requests")
            .map_err(|e| format!("requests not installed: {}", e))?;
        let response = requests
            .call_method1("get", (url,))
            .map_err(|e| format!("GET failed: {}", e))?;
        let status: u16 = response
            .getattr("status_code")
            .and_then(|s| s.extract())
            .map_err(|e| format!("status_code: {}", e))?;
        if !(200..300).contains(&status) {
            return Err(format!("HTTP {}", status));
        }
        let body: String = response
            .getattr("text")
            .and_then(|t| t.extract())
            .map_err(|e| format!("read body: {}", e))?;
        Ok(body)
    })
}

/// Fetch `url` and write to `omc_modules/<name>.omc`. Returns the
/// final on-disk path on success. If `expected_sha256` is Some,
/// verify the body matches before writing — guards against MITM
/// and registry corruption. Hash mismatch is a hard error.
pub fn install_url_via_python(
    name: &str,
    url: &str,
    expected_sha256: Option<&str>,
) -> Result<String, String> {
    let body = fetch_url(url)?;
    if let Some(want) = expected_sha256 {
        let got = sha256_hex(body.as_bytes());
        if got != want {
            return Err(format!(
                "hash mismatch — expected {}, got {} (URL may have been tampered with or registry is stale)",
                want, got
            ));
        }
    }
    let path = format!("omc_modules/{}.omc", name);
    std::fs::write(&path, body).map_err(|e| format!("write {}: {}", path, e))?;
    Ok(path)
}

/// Hex-encoded sha256 of `bytes`. Computed via embedded Python
/// `hashlib` to avoid pulling in a Rust crypto crate. The hashlib
/// path is cold (called once per install), so the overhead is fine.
pub fn sha256_hex(bytes: &[u8]) -> String {
    Python::with_gil(|py| -> PyResult<String> {
        let hashlib = py.import_bound("hashlib")?;
        let h = hashlib.call_method1("sha256", (bytes,))?;
        let hex = h.call_method0("hexdigest")?;
        hex.extract::<String>()
    })
    .unwrap_or_else(|_| "<sha256 failed>".to_string())
}

/// Resolve a registry entry by short name. Fetches the registry
/// index.json (cached for the duration of the process via thread_local),
/// looks up `name`, returns (url, sha256). None if name is not in
/// the registry — caller can fall back to "treat as URL" or error.
///
/// Registry URL defaults to the canonical sovereignlattice/omnimcode
/// repo; override with OMC_REGISTRY env var.
pub fn registry_lookup(name: &str) -> Result<(String, String), String> {
    let registry_url = std::env::var("OMC_REGISTRY").unwrap_or_else(|_| {
        "https://raw.githubusercontent.com/sovereignlattice/omnimcode/main/registry/index.json"
            .to_string()
    });
    let body = fetch_url(&registry_url)
        .map_err(|e| format!("registry fetch {}: {}", registry_url, e))?;
    Python::with_gil(|py| {
        let json = py.import_bound("json").map_err(|e| format!("json: {}", e))?;
        let parsed = json
            .call_method1("loads", (body,))
            .map_err(|e| format!("registry parse: {}", e))?;
        let dict = parsed
            .downcast::<PyDict>()
            .map_err(|e| format!("registry root not a JSON object: {}", e))?;
        let pkgs = match dict.get_item("packages") {
            Ok(Some(p)) => p,
            _ => return Err("registry has no 'packages' key".to_string()),
        };
        let pkgs_d = pkgs
            .downcast::<PyDict>()
            .map_err(|e| format!("packages not an object: {}", e))?;
        let entry = match pkgs_d.get_item(name) {
            Ok(Some(e)) => e,
            _ => return Err(format!("'{}' not in registry", name)),
        };
        let entry_d = entry
            .downcast::<PyDict>()
            .map_err(|e| format!("entry not an object: {}", e))?;
        let url: String = entry_d
            .get_item("url")
            .map_err(|e| format!("url: {}", e))?
            .ok_or_else(|| "entry missing url".to_string())?
            .extract()
            .map_err(|e| format!("url extract: {}", e))?;
        let sha: String = entry_d
            .get_item("sha256")
            .map_err(|e| format!("sha256: {}", e))?
            .ok_or_else(|| "entry missing sha256".to_string())?
            .extract()
            .map_err(|e| format!("sha extract: {}", e))?;
        Ok((url, sha))
    })
}

/// Parse `omc.toml`'s `[dependencies]` table via Python's `tomllib`
/// (stdlib in 3.11+). Returns a list of (name, url) pairs preserving
/// source order.
pub fn parse_omc_toml_via_python(text: &str) -> Result<Vec<(String, String)>, String> {
    Python::with_gil(|py| {
        let tomllib = py
            .import_bound("tomllib")
            .map_err(|e| format!("tomllib not available (need Python 3.11+): {}", e))?;
        // tomllib.loads(text) — needs bytes in some versions, str in others.
        // Use loads with str, fall back to bytes.
        let parsed = match tomllib.call_method1("loads", (text,)) {
            Ok(v) => v,
            Err(_) => tomllib
                .call_method1("loads", (text.as_bytes(),))
                .map_err(|e| format!("tomllib.loads: {}", e))?,
        };
        let dict = parsed
            .downcast::<PyDict>()
            .map_err(|e| format!("toml root must be a table: {}", e))?;
        let deps_obj = match dict.get_item("dependencies") {
            Ok(Some(o)) => o,
            _ => return Ok(Vec::new()),
        };
        let deps = deps_obj
            .downcast::<PyDict>()
            .map_err(|e| format!("[dependencies] must be a table: {}", e))?;
        let mut out: Vec<(String, String)> = Vec::with_capacity(deps.len());
        for (k, v) in deps.iter() {
            let name: String = k.extract().map_err(|e| format!("dep name: {}", e))?;
            // Accept either a string URL or a table with `url = "..."`.
            let url: String = if let Ok(s) = v.extract::<String>() {
                s
            } else if let Ok(t) = v.downcast::<PyDict>() {
                match t.get_item("url") {
                    Ok(Some(u)) => u
                        .extract::<String>()
                        .map_err(|e| format!("dep {} url: {}", name, e))?,
                    _ => return Err(format!("dep {} table missing `url`", name)),
                }
            } else {
                return Err(format!(
                    "dep {} must be a string URL or table with `url`",
                    name
                ));
            };
            out.push((name, url));
        }
        Ok(out)
    })
}
