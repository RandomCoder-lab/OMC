# omc-grep — canonical-hash code archaeology

> The new primitive: find duplicate fns under whitespace, comment,
> parameter-rename, **and (with `--body-only`) under entirely
> different fn names**. Nothing else does the last one.

## What it does

Walks a directory of `.omc` files, extracts every top-level fn,
canonicalizes each one (whitespace stripped, comments removed,
parameter binding normalized), and hashes the canonical form.

Reports:

- **EXACT clusters** — groups of 2+ fns with identical canonical
  hash. These are true duplicates regardless of whitespace, comment
  edits, or parameter renaming.
- **NEAR clusters** (with `--near N`) — fn pairs sharing the same
  Fibonacci attractor whose canonical hashes differ by at most `N`.
  Use this to surface near-duplicates that diverged slightly.
- **Body-only mode** (with `--body-only`) — drops the `fn NAME(...)`
  signature from the hash. This finds fns with identical bodies
  under DIFFERENT NAMES — the form of duplication that name-based
  tools and text grep can never catch.

## What it found on OMC's own examples tree

```
omc-grep examples/
→ 151 files, 2388 fns, 1631 unique → 757 dupes (31.7% redundant)

omc-grep --body-only examples/
→ 151 files, 2388 fns, 1600 unique → 788 dupes (33.0% redundant)
```

The body-only mode caught 31 additional alpha-equivalent clusters
that the name-sensitive pass missed, including:

| Cluster | Members | Distinct names |
|---|--:|---|
| `is_digit` family | 19 | `is_digit`, `is_digit_b`, `is_digit_t` |
| `is_alpha` family | 16 | `is_alpha`, `is_alpha_b` |
| `is_space` family | 16 | `is_space`, `is_space_b` |
| `tok_kind` / `tkind` | 15 | classic rename-during-refactor leftover |
| `tok_value` / `tval` | 15 | same |
| `arr_concat` / `arr_concat_b` | 14 | same |
| 3-bucket family | 5 | `_bucket_discrete`, `endpoint_bucket`, `status_bucket` |
| counter family | 5 | `count_anom_hits`, `count_caught`, `count_hits` |

The 3-bucket family is the case that proves the value: three
domain-specific names (`_bucket_discrete`, `endpoint_bucket`,
`status_bucket`) wrapping the *same code*. No text-grep, ast-grep,
or tree-sitter query can find this because there's no shared token
between the names — only the canonical body matches.

## How the substrate makes this fast

The fnv1a → nearest-Fibonacci-attractor lookup gives every fn an
O(1) substrate address (`attractor_bucket`). Pre-bucketing all fns
by their attractor means near-duplicate detection probes only
within the same bucket, not the full corpus. Combined with the
`log_phi_pi_fibonacci(N)` substrate-search primitive available
inside OMC programs, the same architecture scales to multi-million-
fn corpora.

## Usage

```bash
omc-grep [OPTIONS] DIR

Options:
  --body-only      hash the fn body only (drop name + signature);
                   finds alpha-equivalent fns under DIFFERENT NAMES
  --near N         also report fn pairs within substrate distance N
                   (sharing same Fibonacci attractor) [default: 0]
  --min-cluster K  only report exact clusters with K+ members [default: 2]
  -h, --help       this help
```

Skips: `target/`, `node_modules/`, `.git/`, `__pycache__/`,
`omc_modules/`.

## Building

```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo build --release --bin omc-grep
./target/release/omc-grep DIR
```

No JIT or Python dependencies — pure tree-walk over the canonical
form. ~30s build, <1s scan over 150 files.

## What it doesn't do (yet)

- **Non-OMC languages.** Phase 2 will add Python via the stdlib `ast`
  module (no tree-sitter dependency). After that: JS/TS via the
  tree-sitter bindings.
- **Refactor-suggest mode.** Currently reports clusters; doesn't
  propose which one is the canonical-form, doesn't generate
  rename/import-rewrite diffs. Easy to add but requires a
  per-cluster "winner" heuristic (oldest file? most-used name?
  shortest? linted highest?).
- **Cross-repo dedupe.** Walks one tree. Multi-tree mode (`omc-grep
  A B C/`) would need a per-root prefix for the file column.

These are all worth doing but each is a separable extension on
top of the working core.
