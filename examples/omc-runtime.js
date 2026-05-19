/**
 * omc-runtime.js — OMC JavaScript runtime
 *
 * Provides the type system and standard built-ins needed by code generated
 * by the OMCJ transpiler.  Import it as an ES module:
 *
 *   import * as omc from './omc-runtime.js';
 *
 * or (Node CommonJS bundle):
 *
 *   const omc = require('./omc-runtime.js');
 *
 * All public symbols are also available as named exports so tree-shakers
 * can drop unused built-ins.
 */

// ── Harmonic constants ────────────────────────────────────────────────────────
const PHI  = (1 + Math.sqrt(5)) / 2;   // golden ratio
const PI   = Math.PI;

// ── HInt ─────────────────────────────────────────────────────────────────────
/**
 * Harmonic integer — OMC's native integer type.
 * Wraps a JS Number (safe for |value| < 2^53) and carries two metadata
 * fields that the substrate uses for resonance-aware routing:
 *   • resonance  — |cos(value · π / φ)|, in [0, 1]
 *   • him        — harmonic index magnitude, |value| / φ²
 */
export class HInt {
    constructor(v) {
        this.value = (typeof v === 'bigint') ? Number(v) : Math.trunc(Number(v));
        const angle = (this.value * PI) / PHI;
        this.resonance = Math.abs(Math.cos(angle));
        this.him       = Math.abs(this.value) / (PHI * PHI);
    }

    // Arithmetic — returns HInt
    add(b)  { return new HInt(this.value + coerce(b)); }
    sub(b)  { return new HInt(this.value - coerce(b)); }
    mul(b)  { return new HInt(this.value * coerce(b)); }
    div(b)  { const d = coerce(b); return new HInt(d !== 0 ? Math.trunc(this.value / d) : 0); }
    mod(b)  { const d = coerce(b); return new HInt(d !== 0 ? this.value % d : 0); }
    pow(b)  { return new HInt(Math.trunc(this.value ** coerce(b))); }
    neg()   { return new HInt(-this.value); }

    // Bitwise
    band(b) { return new HInt(this.value & coerce(b)); }
    bor(b)  { return new HInt(this.value | coerce(b)); }
    bxor(b) { return new HInt(this.value ^ coerce(b)); }
    shl(b)  { return new HInt(this.value << coerce(b)); }
    shr(b)  { return new HInt(this.value >> coerce(b)); }

    // Comparisons — return plain boolean
    eq(b)   { return this.value === coerce(b); }
    ne(b)   { return this.value !== coerce(b); }
    lt(b)   { return this.value <   coerce(b); }
    le(b)   { return this.value <=  coerce(b); }
    gt(b)   { return this.value >   coerce(b); }
    ge(b)   { return this.value >=  coerce(b); }

    // Conversion
    toHFloat()   { return new HFloat(this.value); }
    display()    { return String(this.value); }
    toString()   { return `HInt { value: ${this.value}, resonance: ${this.resonance.toFixed(3)}, him: ${this.him.toFixed(3)} }`; }
    valueOf()    { return this.value; }   // lets `+` coerce to number in JS
}

// ── HFloat ────────────────────────────────────────────────────────────────────
/**
 * Harmonic float — OMC's native floating-point type.
 */
export class HFloat {
    constructor(v) {
        this.value = Number(v);
        const angle = (this.value * PI) / PHI;
        this.resonance = Math.abs(Math.cos(angle));
    }

    add(b)  { return new HFloat(this.value + hfCoerce(b)); }
    sub(b)  { return new HFloat(this.value - hfCoerce(b)); }
    mul(b)  { return new HFloat(this.value * hfCoerce(b)); }
    div(b)  { return new HFloat(this.value / hfCoerce(b)); }
    pow(b)  { return new HFloat(this.value ** hfCoerce(b)); }
    neg()   { return new HFloat(-this.value); }

    eq(b)   { return this.value === hfCoerce(b); }
    ne(b)   { return this.value !== hfCoerce(b); }
    lt(b)   { return this.value <   hfCoerce(b); }
    le(b)   { return this.value <=  hfCoerce(b); }
    gt(b)   { return this.value >   hfCoerce(b); }
    ge(b)   { return this.value >=  hfCoerce(b); }

    toHInt()   { return new HInt(Math.trunc(this.value)); }
    display()  { return this.value.toString().includes('.') ? this.value.toString() : this.value.toFixed(1); }
    toString() { return `HFloat { value: ${this.value}, resonance: ${this.resonance.toFixed(3)} }`; }
    valueOf()  { return this.value; }
}

// ── Internal coercion helpers ─────────────────────────────────────────────────
function coerce(v) {
    if (v instanceof HInt)   return v.value;
    if (v instanceof HFloat) return Math.trunc(v.value);
    return Math.trunc(Number(v));
}
function hfCoerce(v) {
    if (v instanceof HInt)   return v.value;
    if (v instanceof HFloat) return v.value;
    return Number(v);
}

/** Lift a raw JS value to the appropriate OMC type. */
export function lift(v) {
    if (v instanceof HInt || v instanceof HFloat) return v;
    if (typeof v === 'number') {
        return Number.isInteger(v) ? new HInt(v) : new HFloat(v);
    }
    return v;  // strings, bools, arrays, null — pass through
}

/** Arithmetic dispatch: picks HFloat if either operand is float. */
export function omcAdd(a, b) {
    if (a instanceof HFloat || b instanceof HFloat) return new HFloat(hfCoerce(a) + hfCoerce(b));
    if (a instanceof HInt)   return a.add(b);
    return lift(Number(a) + Number(b));
}
export function omcSub(a, b) {
    if (a instanceof HFloat || b instanceof HFloat) return new HFloat(hfCoerce(a) - hfCoerce(b));
    if (a instanceof HInt)   return a.sub(b);
    return lift(Number(a) - Number(b));
}
export function omcMul(a, b) {
    if (a instanceof HFloat || b instanceof HFloat) return new HFloat(hfCoerce(a) * hfCoerce(b));
    if (a instanceof HInt)   return a.mul(b);
    return lift(Number(a) * Number(b));
}
export function omcDiv(a, b) {
    const bv = hfCoerce(b);
    if (a instanceof HFloat) return new HFloat(a.value / bv);
    if (b instanceof HFloat) return new HFloat(hfCoerce(a) / bv);
    if (a instanceof HInt)   return a.div(b);
    return lift(Math.trunc(Number(a) / Number(b)));
}
export function omcMod(a, b) {
    if (a instanceof HInt)   return a.mod(b);
    return lift(Number(a) % Number(b));
}
export function omcPow(a, b) {
    if (a instanceof HFloat || b instanceof HFloat) return new HFloat(hfCoerce(a) ** hfCoerce(b));
    if (a instanceof HInt)   return a.pow(b);
    return lift(Number(a) ** Number(b));
}
export function omcNeg(a) {
    if (a instanceof HInt)   return a.neg();
    if (a instanceof HFloat) return a.neg();
    return lift(-Number(a));
}

export function omcEq(a, b)  { return omcCmp(a, b) === 0; }
export function omcNe(a, b)  { return omcCmp(a, b) !== 0; }
export function omcLt(a, b)  { return omcCmp(a, b) <  0; }
export function omcLe(a, b)  { return omcCmp(a, b) <= 0; }
export function omcGt(a, b)  { return omcCmp(a, b) >  0; }
export function omcGe(a, b)  { return omcCmp(a, b) >= 0; }
function omcCmp(a, b) {
    const av = hfCoerce(a), bv = hfCoerce(b);
    return av < bv ? -1 : av > bv ? 1 : 0;
}

/** Display a value (same rules as OMC to_display_string). */
export function display(v) {
    if (v === null || v === undefined) return "null";
    if (v instanceof HInt)   return v.display();
    if (v instanceof HFloat) return v.display();
    if (typeof v === 'boolean') return v ? "true" : "false";
    if (Array.isArray(v)) return '[' + v.map(display).join(', ') + ']';
    return String(v);
}

// ── Standard I/O ──────────────────────────────────────────────────────────────
const _stdout = (typeof process !== 'undefined' && process.stdout)
    ? (s) => process.stdout.write(s)
    : (s) => console.log(s);  // browser fallback (adds newline)

export function println(...args) { _stdout(args.map(display).join(' ') + '\n'); return null; }
export function print(...args)   { _stdout(args.map(display).join(' ')); return null; }
export function eprint(v)        { console.error(display(v)); return null; }

// ── Type checking ─────────────────────────────────────────────────────────────
export function is_int(v)    { return v instanceof HInt; }
export function is_float(v)  { return v instanceof HFloat; }
export function is_string(v) { return typeof v === 'string'; }
export function is_bool(v)   { return typeof v === 'boolean'; }
export function is_array(v)  { return Array.isArray(v); }
export function is_null(v)   { return v === null || v === undefined; }

// ── Type conversion ───────────────────────────────────────────────────────────
export function to_int(v)    { return new HInt(coerce(v)); }
export function to_float(v)  { return new HFloat(hfCoerce(v)); }
export function to_string(v) { return display(v); }
export function to_bool(v)   {
    if (v instanceof HInt)   return v.value !== 0;
    if (v instanceof HFloat) return v.value !== 0;
    return Boolean(v);
}

// ── Array built-ins ───────────────────────────────────────────────────────────
export function len(v) {
    if (Array.isArray(v)) return new HInt(v.length);
    if (typeof v === 'string') return new HInt(v.length);
    return new HInt(0);
}
export function arr_push(arr, item) { const a = [...arr, item]; return a; }
export function arr_pop(arr) { if (!arr.length) return null; return [arr.slice(0,-1), arr[arr.length-1]]; }
export function arr_map(arr, fn) { return arr.map(fn); }
export function arr_filter(arr, fn) { return arr.filter(fn); }
export function arr_reduce(arr, fn, init) { return arr.reduce(fn, init); }
export function arr_sum(arr) { return lift(arr.reduce((a,b) => hfCoerce(a)+hfCoerce(b), 0)); }
export function arr_min(arr) { return lift(arr.reduce((a,b) => hfCoerce(a)<hfCoerce(b)?a:b)); }
export function arr_max(arr) { return lift(arr.reduce((a,b) => hfCoerce(a)>hfCoerce(b)?a:b)); }
export function arr_sort(arr) { return [...arr].sort((a,b) => omcCmp(a,b)); }
export function arr_reverse(arr) { return [...arr].reverse(); }
export function arr_contains(arr, v) { return arr.some(x => omcEq(x, v)); }
export function arr_join(arr, sep) { return arr.map(display).join(sep ?? ','); }
export function arr_zip(a, b) {
    const n = Math.min(a.length, b.length);
    const r = [];
    for (let i = 0; i < n; i++) r.push([a[i], b[i]]);
    return r;
}
export function arr_enumerate(arr) { return arr.map((v, i) => [new HInt(i), v]); }
export function arr_flatten(arr) { return arr.flat(1); }
export function arr_unique(arr) {
    const seen = new Set();
    return arr.filter(v => { const k = display(v); if (seen.has(k)) return false; seen.add(k); return true; });
}
export function arr_slice(arr, start, end) {
    return arr.slice(coerce(start), end != null ? coerce(end) : undefined);
}
export function arr_first(arr) { return arr.length ? arr[0] : null; }
export function arr_last(arr) { return arr.length ? arr[arr.length-1] : null; }
export function arr_head(arr) { return arr.slice(0, -1); }
export function arr_tail(arr) { return arr.slice(1); }
export function arr_count(arr, fn) { return new HInt(arr.filter(fn).length); }
export function arr_any(arr, fn) { return arr.some(fn); }
export function arr_all(arr, fn) { return arr.every(fn); }
export function arr_flat_map(arr, fn) { return arr.flatMap(fn); }

// ── Math ──────────────────────────────────────────────────────────────────────
export function abs(v)    { return lift(Math.abs(hfCoerce(v))); }
export function sqrt(v)   { return new HFloat(Math.sqrt(hfCoerce(v))); }
export function floor(v)  { return new HInt(Math.floor(hfCoerce(v))); }
export function ceil(v)   { return new HInt(Math.ceil(hfCoerce(v))); }
export function round(v)  { return new HInt(Math.round(hfCoerce(v))); }
export function sin(v)    { return new HFloat(Math.sin(hfCoerce(v))); }
export function cos(v)    { return new HFloat(Math.cos(hfCoerce(v))); }
export function tan(v)    { return new HFloat(Math.tan(hfCoerce(v))); }
export function exp(v)    { return new HFloat(Math.exp(hfCoerce(v))); }
export function log(v)    { return new HFloat(Math.log(hfCoerce(v))); }
export function log2(v)   { return new HFloat(Math.log2(hfCoerce(v))); }
export function log10(v)  { return new HFloat(Math.log10(hfCoerce(v))); }
export function pow(a, b) { return omcPow(a, b); }
export function max(a, b) { return omcGe(a,b) ? a : b; }
export function min(a, b) { return omcLe(a,b) ? a : b; }
export function clamp(v, lo, hi) { return omcLt(v,lo) ? lo : omcGt(v,hi) ? hi : v; }
export function sign(v)   { const n = hfCoerce(v); return new HInt(n > 0 ? 1 : n < 0 ? -1 : 0); }
export function gcd(a, b) {
    let x = Math.abs(coerce(a)), y = Math.abs(coerce(b));
    while (y) { [x, y] = [y, x % y]; }
    return new HInt(x);
}
export function lcm(a, b) { const g = coerce(gcd(a,b)); return new HInt(Math.trunc(Math.abs(coerce(a)*coerce(b))/g)); }
export const PI_VAL  = new HFloat(PI);
export const PHI_VAL = new HFloat(PHI);

// ── String built-ins ──────────────────────────────────────────────────────────
export function str_concat(a, b) { return display(a) + display(b); }
export function str_len(s) { return new HInt(String(s).length); }
export function str_upper(s) { return String(s).toUpperCase(); }
export function str_lower(s) { return String(s).toLowerCase(); }
export function str_trim(s)  { return String(s).trim(); }
export function str_split(s, sep) { return String(s).split(String(sep)); }
export function str_contains(s, sub) { return String(s).includes(String(sub)); }
export function str_starts_with(s, p) { return String(s).startsWith(String(p)); }
export function str_ends_with(s, p)   { return String(s).endsWith(String(p)); }
export function str_replace(s, from, to) { return String(s).split(String(from)).join(String(to)); }
export function str_index_of(s, sub) { return new HInt(String(s).indexOf(String(sub))); }
export function str_slice(s, start, end) {
    return String(s).slice(coerce(start), end != null ? coerce(end) : undefined);
}
export function str_repeat(s, n) { return String(s).repeat(coerce(n)); }
export function str_chars(s) { return [...String(s)]; }
export function str_lines(s) { return String(s).split('\n'); }
export function str_pad_start(s, n, ch) { return String(s).padStart(coerce(n), String(ch ?? ' ')); }
export function str_pad_end(s, n, ch)   { return String(s).padEnd(coerce(n), String(ch ?? ' ')); }
export function format_float(v, prec) {
    const f = hfCoerce(v);
    return prec != null ? f.toFixed(coerce(prec)) : (Number.isInteger(f) ? f.toFixed(1) : String(f));
}

// ── Dict / Object ─────────────────────────────────────────────────────────────
export function dict_get(d, k) { return d[display(k)] ?? null; }
export function dict_set(d, k, v) { return {...d, [display(k)]: v}; }
export function dict_has(d, k) { return Object.prototype.hasOwnProperty.call(d, display(k)); }
export function dict_keys(d) { return Object.keys(d); }
export function dict_values(d) { return Object.values(d); }
export function dict_items(d) { return Object.entries(d).map(([k,v]) => [k,v]); }
export function dict_delete(d, k) { const r = {...d}; delete r[display(k)]; return r; }

// ── Harmonic / Substrate ops (pure JS approximations) ─────────────────────────
const FIB = [1,1,2,3,5,8,13,21,34,55,89,144,233,377,610,987,1597,2584,4181,6765];
export function phi_floor(n) { return new HInt(Math.round(hfCoerce(n) / PHI)); }
export function phi_ceil(n)  { return new HInt(Math.ceil(hfCoerce(n) * PHI - 0.5)); }
export function is_fibonacci(n) {
    const v = Math.abs(coerce(n));
    return FIB.includes(v) || (function isPerfectSquare(x) {
        const s = Math.round(Math.sqrt(x)); return s*s===x;
    })(5*v*v+4) || (function isPerfectSquare(x) {
        const s = Math.round(Math.sqrt(x)); return s*s===x;
    })(5*v*v-4);
}
export function fnv1a_64(s) {
    // 64-bit FNV-1a approximation (BigInt for correctness, returns HInt)
    let hash = 14695981039346656037n;
    const FNV_PRIME = 1099511628211n;
    for (const ch of String(s)) {
        hash ^= BigInt(ch.charCodeAt(0));
        hash = BigInt.asUintN(64, hash * FNV_PRIME);
    }
    return new HInt(Number(BigInt.asIntN(64, hash)));
}
export function harmonic(n) {
    const v = hfCoerce(n);
    return new HFloat(Math.abs(Math.cos(v * PI / PHI)));
}
export function phi_pi_fib(n) {
    const v = Math.abs(coerce(n));
    // Nearest Fibonacci attractor
    let lo = 1, hi = 1;
    for (const f of FIB) { if (f <= v) lo = f; else { hi = f; break; } }
    const dist = Math.min(v - lo, hi - v);
    return new HInt(dist);
}

// ── Ranges ────────────────────────────────────────────────────────────────────
export function range(start, end, step) {
    const s = coerce(start), e = coerce(end), st = step != null ? coerce(step) : 1;
    const r = [];
    if (st > 0) for (let i = s; i < e; i += st) r.push(new HInt(i));
    else         for (let i = s; i > e; i += st) r.push(new HInt(i));
    return r;
}
export function range_inclusive(start, end, step) {
    const s = coerce(start), e = coerce(end), st = step != null ? coerce(step) : 1;
    const r = [];
    if (st > 0) for (let i = s; i <= e; i += st) r.push(new HInt(i));
    else         for (let i = s; i >= e; i += st) r.push(new HInt(i));
    return r;
}

// ── I/O helpers (Node.js) ─────────────────────────────────────────────────────
export function read_file(path) {
    try {
        const fs = (typeof require !== 'undefined') ? require('fs') : null;
        return fs ? fs.readFileSync(String(path), 'utf8') : null;
    } catch { return null; }
}
export function write_file(path, content) {
    try {
        const fs = (typeof require !== 'undefined') ? require('fs') : null;
        if (fs) { fs.writeFileSync(String(path), String(content)); return true; }
        return false;
    } catch { return false; }
}
export function file_exists(path) {
    try {
        const fs = (typeof require !== 'undefined') ? require('fs') : null;
        return fs ? fs.existsSync(String(path)) : false;
    } catch { return false; }
}

// ── Error handling ────────────────────────────────────────────────────────────
export class OmcError extends Error {
    constructor(msg) { super(String(msg)); this.name = 'OmcError'; this.omcMessage = msg; }
}
export function omcThrow(v) { throw new OmcError(v); }
export function omcAssert(cond, msg) { if (!to_bool(cond)) throw new OmcError(msg ?? 'assertion failed'); }

// ── Null helpers ──────────────────────────────────────────────────────────────
export const NULL = null;
export function is_none(v) { return v === null || v === undefined; }
export function unwrap(v)  { if (is_none(v)) throw new OmcError('unwrap on null'); return v; }
export function unwrap_or(v, def) { return is_none(v) ? def : v; }

// ── Random ────────────────────────────────────────────────────────────────────
export function rand_float() { return new HFloat(Math.random()); }
export function rand_int(lo, hi) {
    const l = coerce(lo), h = coerce(hi);
    return new HInt(l + Math.floor(Math.random() * (h - l + 1)));
}

// ── Default export: all builtins as a flat object ─────────────────────────────
export default {
    HInt, HFloat, lift, display,
    omcAdd, omcSub, omcMul, omcDiv, omcMod, omcPow, omcNeg,
    omcEq, omcNe, omcLt, omcLe, omcGt, omcGe,
    println, print, eprint,
    is_int, is_float, is_string, is_bool, is_array, is_null,
    to_int, to_float, to_string, to_bool,
    len, arr_push, arr_pop, arr_map, arr_filter, arr_reduce,
    arr_sum, arr_min, arr_max, arr_sort, arr_reverse, arr_contains,
    arr_join, arr_zip, arr_enumerate, arr_flatten, arr_unique,
    arr_slice, arr_first, arr_last, arr_head, arr_tail,
    arr_count, arr_any, arr_all, arr_flat_map,
    abs, sqrt, floor, ceil, round, sin, cos, tan, exp, log, log2, log10,
    pow, max, min, clamp, sign, gcd, lcm, PI: PI_VAL, PHI: PHI_VAL,
    str_concat, str_len, str_upper, str_lower, str_trim, str_split,
    str_contains, str_starts_with, str_ends_with, str_replace,
    str_index_of, str_slice, str_repeat, str_chars, str_lines,
    str_pad_start, str_pad_end, format_float,
    dict_get, dict_set, dict_has, dict_keys, dict_values, dict_items, dict_delete,
    phi_floor, phi_ceil, is_fibonacci, fnv1a_64, harmonic, phi_pi_fib,
    range, range_inclusive,
    read_file, write_file, file_exists,
    OmcError, omcThrow, omcAssert, NULL, is_none, unwrap, unwrap_or,
    rand_float, rand_int,
};
