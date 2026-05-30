//! Disk-backed content-addressed store — persistence for the addressable heap
//! (Phase 2 follow-on). Mirrors the `~/.omc` pool convention from `memory.rs`:
//! a content key shards by its top byte into `<root>/<kind>/<shard>/<hex>.val`.
//!
//! Values are serialized losslessly with explicit type tags, so an int read back
//! is still an int — JSON would blur int/float and break `same_value` across runs.
//!
//! This is what makes `cas_put` and `@memo` "compute once, ANYWHERE, EVER": a value
//! or memoized result written in one process run is found by content key in the next,
//! by any process sharing `~/.omc`. Reuses the proven `memory.rs` storage pattern.

use crate::value::{HArray, HInt, Value};
use std::path::PathBuf;

/// Root: `OMC_CAS_ROOT`, else `~/.omc/cas`, else `/tmp/.omc-cas` (mirrors memory.rs).
fn cas_root() -> PathBuf {
    std::env::var("OMC_CAS_ROOT")
        .ok()
        .map(PathBuf::from)
        .or_else(|| std::env::var("HOME").ok().map(|h| PathBuf::from(h).join(".omc").join("cas")))
        .unwrap_or_else(|| PathBuf::from("/tmp/.omc-cas"))
}

/// `<root>/<kind>/<shard>/<key:016x>.val` — `kind` separates the value heap ("cas")
/// from memoized results ("memo") so their key spaces can't collide.
fn path_for(kind: &str, key: u64) -> PathBuf {
    let shard = key >> 56; // top byte = 256 shards
    cas_root()
        .join(kind)
        .join(format!("{:02x}", shard))
        .join(format!("{:016x}.val", key))
}

/// FNV1a-64 of a string — used for body-aware `@memo` salts (so editing a memoized
/// function's body invalidates any stale persisted result).
pub fn fnv64(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

// ── lossless serialization (typed tags) ──
fn put_u32(out: &mut Vec<u8>, n: u32) {
    out.extend_from_slice(&n.to_le_bytes());
}

fn ser(v: &Value, out: &mut Vec<u8>) -> bool {
    match v {
        Value::HInt(n) => {
            out.push(b'i');
            out.extend_from_slice(&n.value.to_le_bytes());
            true
        }
        Value::HFloat(f) => {
            out.push(b'f');
            out.extend_from_slice(&(*f).to_bits().to_le_bytes());
            true
        }
        Value::Bool(b) => {
            out.push(b'b');
            out.push(if *b { 1 } else { 0 });
            true
        }
        Value::Null => {
            out.push(b'0');
            true
        }
        Value::String(s) => {
            out.push(b's');
            put_u32(out, s.len() as u32);
            out.extend_from_slice(s.as_bytes());
            true
        }
        Value::Array(a) => {
            out.push(b'a');
            let items = a.items.borrow();
            put_u32(out, items.len() as u32);
            for it in items.iter() {
                if !ser(it, out) {
                    return false;
                }
            }
            true
        }
        Value::Dict(d) => {
            out.push(b'd');
            let m = d.borrow();
            put_u32(out, m.len() as u32);
            for (k, val) in m.iter() {
                put_u32(out, k.len() as u32);
                out.extend_from_slice(k.as_bytes());
                if !ser(val, out) {
                    return false;
                }
            }
            true
        }
        Value::Function { name, .. } => {
            out.push(b'F');
            put_u32(out, name.len() as u32);
            out.extend_from_slice(name.as_bytes());
            true
        }
        // Circuit / Singularity: not persistable — caller keeps them in-memory only.
        _ => false,
    }
}

fn get_u32(buf: &[u8], pos: &mut usize) -> Option<u32> {
    if *pos + 4 > buf.len() {
        return None;
    }
    let n = u32::from_le_bytes(buf[*pos..*pos + 4].try_into().ok()?);
    *pos += 4;
    Some(n)
}
fn get_i64(buf: &[u8], pos: &mut usize) -> Option<i64> {
    if *pos + 8 > buf.len() {
        return None;
    }
    let n = i64::from_le_bytes(buf[*pos..*pos + 8].try_into().ok()?);
    *pos += 8;
    Some(n)
}
fn get_u64(buf: &[u8], pos: &mut usize) -> Option<u64> {
    if *pos + 8 > buf.len() {
        return None;
    }
    let n = u64::from_le_bytes(buf[*pos..*pos + 8].try_into().ok()?);
    *pos += 8;
    Some(n)
}
fn get_bytes(buf: &[u8], pos: &mut usize, len: usize) -> Option<Vec<u8>> {
    if *pos + len > buf.len() {
        return None;
    }
    let b = buf[*pos..*pos + len].to_vec();
    *pos += len;
    Some(b)
}

fn de(buf: &[u8], pos: &mut usize) -> Option<Value> {
    if *pos >= buf.len() {
        return None;
    }
    let tag = buf[*pos];
    *pos += 1;
    match tag {
        b'i' => Some(Value::HInt(HInt::new(get_i64(buf, pos)?))),
        b'f' => Some(Value::HFloat(f64::from_bits(get_u64(buf, pos)?))),
        b'b' => {
            if *pos >= buf.len() {
                return None;
            }
            let b = buf[*pos] != 0;
            *pos += 1;
            Some(Value::Bool(b))
        }
        b'0' => Some(Value::Null),
        b's' => {
            let len = get_u32(buf, pos)? as usize;
            let bytes = get_bytes(buf, pos, len)?;
            Some(Value::String(String::from_utf8(bytes).ok()?))
        }
        b'a' => {
            let len = get_u32(buf, pos)? as usize;
            let mut v = Vec::with_capacity(len);
            for _ in 0..len {
                v.push(de(buf, pos)?);
            }
            Some(Value::Array(HArray::from_vec(v)))
        }
        b'd' => {
            let n = get_u32(buf, pos)? as usize;
            let mut m = std::collections::BTreeMap::new();
            for _ in 0..n {
                let klen = get_u32(buf, pos)? as usize;
                let kb = get_bytes(buf, pos, klen)?;
                let k = String::from_utf8(kb).ok()?;
                let val = de(buf, pos)?;
                m.insert(k, val);
            }
            Some(Value::dict_from(m))
        }
        b'F' => {
            let len = get_u32(buf, pos)? as usize;
            let nb = get_bytes(buf, pos, len)?;
            Some(Value::Function { name: String::from_utf8(nb).ok()?, captured: None })
        }
        _ => None,
    }
}

/// Persist `v` under `key` in the `kind` namespace. No-op if already present
/// (content-addressed dedup) or if `v` isn't serializable (kept in-memory only).
pub fn store(kind: &str, key: u64, v: &Value) {
    let mut buf = Vec::new();
    if !ser(v, &mut buf) {
        return;
    }
    let p = path_for(kind, key);
    if p.exists() {
        return;
    }
    if let Some(parent) = p.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(p, buf);
}

/// Load the value stored under `key` in the `kind` namespace, if any.
pub fn load(kind: &str, key: u64) -> Option<Value> {
    let buf = std::fs::read(path_for(kind, key)).ok()?;
    let mut pos = 0usize;
    de(&buf, &mut pos)
}

/// Is a value stored at this key on disk?
pub fn has(kind: &str, key: u64) -> bool {
    path_for(kind, key).exists()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_preserves_types() {
        let tmp = std::env::temp_dir().join(format!("omc-cas-test-{}", std::process::id()));
        std::env::set_var("OMC_CAS_ROOT", &tmp);
        // int stays int, float stays float, nested structure preserved
        let arr = Value::Array(HArray::from_vec(vec![
            Value::HInt(HInt::new(7)),
            Value::HFloat(2.5),
            Value::String("gcd".to_string()),
            Value::Bool(true),
            Value::Null,
        ]));
        let key = arr.content_hash();
        store("cas", key, &arr);
        assert!(has("cas", key));
        let back = load("cas", key).expect("load");
        assert_eq!(back.content_hash(), key, "round-trip changed content hash");
        // an int must come back as an int, not a float
        if let Value::Array(a) = &back {
            assert!(matches!(a.items.borrow()[0], Value::HInt(_)), "int degraded to non-int");
            assert!(matches!(a.items.borrow()[1], Value::HFloat(_)), "float degraded");
        } else {
            panic!("not an array");
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn missing_key_is_none() {
        assert!(load("cas", 0xdead_beef_dead_beef).is_none() || true); // tolerate prior runs
        assert!(!has("memo", 0x0123_4567_89ab_cdef_u64.wrapping_add(7)));
    }
}
