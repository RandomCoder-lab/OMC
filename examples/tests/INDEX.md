# Test suite index

**65 test files, 997 `fn test_*` functions, all green under `omnimcode-standalone --test FILE`.**

This is a map of what's covered, organized by surface area — not an
exhaustive doc. Run any file with `--test FILE` to see the actual
assertions.

## Quick categories

| Surface area | Files | ~tests |
|---|---|--:|
| Substrate primitives & arrays | substrate_primitives, substrate_extras, substrate_more, substrate_array | 108 |
| Substrate codec, messaging, canonical | codec, compressed_messaging, substrate_messaging, canonical, canonical_extras, find_similar, tokenizer, tokenizer_extras | 96 |
| Code intelligence, introspection, LLM workflow | code_intel, code_intel_extras, code_intel_more, introspection, introspection_extras, introspection_helpers, llm_workflow, workflow_extras, error_catalog, session_memory | 152 |
| Core language | core_features, classes (×3), exceptions (×2), typed_exceptions, generators (×3), lazy_generators, fstrings (×2), regex (×2), json (×2) | 195 |
| ML / autograd / numerics | autograd (×2), reverse_autograd (×2), broadcasting (×2), matmul, linalg_extras, ml_kernels (×2), math_extras | 137 |
| Stdlib / utility | arr_extras, str_extras, dict_extras, stdlib (×2) | 104 |
| Harmonic libraries | harmonic_libs | 18 |
| ONN / geodesic | onn, geodesic | 24 |
| Self-healing pass | heal_pass | 16 |
| Parametric / mega-coverage | parametric (×4), mega_parametric | 59 |
| Misc / kitchen-sink | new_builtins | 70 |

Numbers are `grep -c '^fn test_'` per file. Some parametric tests
exercise many sub-assertions inside one `test_*` body, so the
*assertion* count is much higher than 997.

## File-by-file

### Substrate codec, messaging, canonical (the LLM-channel layer)
| File | Tests | Purpose |
|---|--:|---|
| `test_codec.omc` | 7 | `omc_codec_encode/decode_lookup` — alpha-rename invariant library recovery + inline error-hint UX check |
| `test_compressed_messaging.omc` | 6 | `omc_msg_sign_compressed/recover` — substrate-signed wire payloads carrying codec output, JSON round-trip |
| `test_substrate_messaging.omc` | 10 | The base substrate-signed messaging protocol (pre-codec) — `omc_msg_sign / verify / serialize` |
| `test_canonical.omc` | 15 | AST canonicalization — the LLM-reach-for semantic-equivalence layer |
| `test_canonical_extras.omc` | 11 | More canonical / structural-equivalence cases |
| `test_find_similar.omc` | 8 | Substrate-distance code retrieval — content-addressed code search |
| `test_tokenizer.omc` | 15 | Substrate-token adapter — LLM compression / semantic-distance layer |
| `test_tokenizer_extras.omc` | 20 | Additional tokenizer + canonical + code_intel coverage |

### Substrate primitives & arrays
| File | Tests | Purpose |
|---|--:|---|
| `test_substrate_primitives.omc` | 57 | The O(log_phi_pi_fibonacci N) primitive family — `substrate_search`, `substrate_lower_bound`, Zeckendorf, etc. |
| `test_substrate_extras.omc` | 25 | Additional substrate-primitive coverage |
| `test_substrate_more.omc` | 11 | More substrate-primitive coverage |
| `test_substrate_array.omc` | 15 | Substrate-typed array library — MVP |

### Code intel, introspection, LLM workflow
| File | Tests | Purpose |
|---|--:|---|
| `test_code_intel.omc` | 20 | The LLM-iteration primitives layered on top of canonical form |
| `test_code_intel_extras.omc` | 14 | Diff + metrics |
| `test_code_intel_more.omc` | 10 | Yet more |
| `test_introspection.omc` | 13 | Built-in introspection — what LLMs need to know what OMC can do |
| `test_introspection_extras.omc` | 20 | Additional introspection / discoverability coverage |
| `test_introspection_helpers.omc` | 14 | Helper builtins |
| `test_llm_workflow.omc` | 14 | End-to-end LLM workflow primitives |
| `test_workflow_extras.omc` | 22 | More workflow / introspection tests |
| `test_error_catalog.omc` | 7 | `omc_explain_error` pattern-matches runtime errors |
| `test_session_memory.omc` | 18 | Session memory + token introspection + substrate scoring builtins |

### Core language
| File | Tests | Purpose |
|---|--:|---|
| `test_core_features.omc` | 13 | Control flow, recursion, lambdas |
| `test_classes.omc` / `_extras` / `_more` | 11/11/11 | Minimum-viable class system + inheritance |
| `test_exceptions.omc` / `_extras` | 8/11 | throw, try/catch, finally |
| `test_typed_exceptions.omc` | 9 | Typed exception hierarchies (Track 1) |
| `test_generators.omc` / `_extras` / `_more` | 8/14/10 | Eager-list generator MVP |
| `test_lazy_generators.omc` | 12 | Streaming yield via callback |
| `test_fstrings.omc` / `_extras` | 10/10 | `f"..."` interpolation |
| `test_regex.omc` / `_extras` | 10/10 | `re_match`, `re_find`, `re_find_all`, `re_replace`, `re_split` |
| `test_json.omc` / `_extras` | 17/14 | `json_parse` / `json_stringify` |

### ML / autograd / numerics
| File | Tests | Purpose |
|---|--:|---|
| `test_autograd.omc` / `_extras` | 17/22 | Substrate-aware forward-mode autograd via dual numbers (Track 2) |
| `test_reverse_autograd.omc` / `_extras` | 12/10 | Reverse-mode autograd — the real ML training engine |
| `test_broadcasting.omc` / `_extras` | 9/10 | 2D-aware broadcasting on `arr_add/sub/mul` |
| `test_matmul.omc` | 9 | 2D arrays + matrix multiplication |
| `test_linalg_extras.omc` | 11 | Linalg / matmul / transpose coverage |
| `test_ml_kernels.omc` / `_extras` | 16/13 | Native-Rust ML primitives keeping inner loops out of OMC |
| `test_math_extras.omc` | 36 | Math builtin coverage |

### Stdlib / utility
| File | Tests | Purpose |
|---|--:|---|
| `test_arr_extras.omc` | 38 | `arr_*` builtin coverage |
| `test_str_extras.omc` | 24 | `str_*` builtin coverage |
| `test_dict_extras.omc` | 16 | `dict_*` builtin coverage |
| `test_stdlib.omc` / `_extras` | 12/14 | Hashing, base64, datetime |

### Harmonic libraries / ONN / heal
| File | Tests | Purpose |
|---|--:|---|
| `test_harmonic_libs.omc` | 18 | `harmonic_anomaly`, `harmonic_clustering`, `harmonic_recommend` |
| `test_onn.omc` | 14 | ONN / self-instantiation / context-compression |
| `test_geodesic.omc` | 10 | ChildFold / geodesic-expand (Sovereign Lattice port) |
| `test_heal_pass.omc` | 16 | Self-healing compiler heal classes + per-class pragmas |

### Parametric / mega-coverage
| File | Tests | Purpose |
|---|--:|---|
| `test_parametric.omc` | 13 | Table-driven, several inputs/properties per test |
| `test_parametric_2.omc` | 8 | More table-driven coverage |
| `test_parametric_3.omc` | 12 | Many sub-assertions per test |
| `test_parametric_4.omc` | 13 | Yet more table-driven assertions |
| `test_mega_parametric.omc` | 13 | Max-coverage table-driven tests (~900 atomic sub-assertions) |

### Misc / kitchen-sink
| File | Tests | Purpose |
|---|--:|---|
| `test_new_builtins.omc` | 70 | Coverage for the steady stream of new builtins — should be triaged into topical files over time |

## Known gaps & overlap

- **`test_new_builtins.omc` (70 tests)** is a kitchen-sink that's
  grown across many sessions. Worth splitting into topical files —
  but only when it's the actual blocker for a change.
- **`test_substrate_messaging.omc` vs `test_compressed_messaging.omc`**:
  no overlap. The former covers raw substrate-signed messaging
  (pre-codec); the latter covers the codec-augmented variant. Both
  stay.
- **`test_classes*.omc` (3 files)** and **`test_parametric*.omc`
  (4 files)**: these accumulated by-session, not by-topic. Worth
  a one-shot consolidation pass when convenient.
- **No top-level test runner** that exercises every file in
  sequence. `omnimcode-standalone --test FILE` is per-file; a
  `scripts/run_all_tests.sh` (or a Cargo test that shells out)
  would prevent regression-by-omission.

## Run anything

```bash
omnimcode-standalone --test examples/tests/test_codec.omc
omnimcode-standalone --test examples/tests/test_substrate_primitives.omc
# ...
```
