//! Prometheus — substrate-native ML framework.
//!
//! Status: SCAFFOLDING. The MVP composition layer lives in pure OMC
//! (see `examples/lib/prometheus.omc`) and a tiny LM trains
//! end-to-end via the existing tape_* / arr_* primitives
//! (see `examples/prometheus_tinylm.omc`). This Rust module is
//! reserved for the substrate-unique features that are NOT
//! achievable in pure-OMC composition — they need primitive-level
//! Rust support.
//!
//! See `omnimcode-core/src/prometheus/README.md` for the strategic
//! roadmap, priority order, and what goes here vs in the OMC lib.
//!
//! For now this module is intentionally empty. The work currently
//! happens in:
//!   - `examples/lib/prometheus.omc`        — composition layer
//!   - `examples/prometheus_tinylm.omc`     — trained tiny LM (MVP proof)
//!   - existing `tape_*` builtins in `interpreter.rs`
//!   - existing `arr_*` ML kernels in `interpreter.rs` + `ml_kernels.rs`

// Intentionally empty until the substrate-unique primitives below
// graduate from "designed" to "shipped":
//
//   - tape_update_scaled(var_id, lr, scale)   — for harmonic optimizer
//   - tape_save_weights(model, path)          — content-addressed .omcs
//   - tape_load_weights(path)                 — alpha-rename-invariant
//   - tape_cache_forward(input_hash, ...)     — substrate-cached activations
//   - tape_geodesic_attention(Q, K, V, seq_len) — geodesic attention as one op
