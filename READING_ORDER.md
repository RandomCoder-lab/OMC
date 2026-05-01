# READING ORDER - OMNIcode v1.1 Documentation

**Start here** if you're new to the project.

---

## First Time Users

### 1️⃣ **README.md** (5 min read)
   - What is OMNIcode?
   - Feature overview
   - Quick start guide

### 2️⃣ **COMPLETION_SUMMARY.md** (10 min read)
   - Tier 1 implementation status
   - What was delivered
   - Performance metrics
   - Next steps

### 3️⃣ **Build and Run** (5 min)
   ```
   cd /home/thearchitect/OMC
   cargo build --release
   ./standalone.omc examples/hello_world.omc
   ```

---

## Developers

### For Understanding the Architecture

1. **DEVELOPER.md** (30 min read) - Detailed architecture guide
   - Module breakdown
   - Circuit DSL grammar
   - Compiler pipeline
   - Testing strategy

2. **src/circuits.rs** (20 min) - Gate implementations
   - Gate enum definition
   - Hard & soft evaluation
   - Validation logic

3. **src/evolution.rs** (15 min) - Genetic operators
   - Mutation, crossover, selection
   - GA loop
   - Fitness calculation

### For Performance Tuning

1. **BENCHMARKS.md** (15 min) - Detailed metrics
   - Before/after performance
   - Bottleneck analysis
   - Optimization opportunities

---

## Project Managers / Stakeholders

### For Understanding Scope & Progress

1. **COMPLETION_SUMMARY.md** (10 min)
   - Status: Tier 1 Complete ✅
   - Deliverables checklist

2. **IMPROVEMENT_PLAN.md** (15 min)
   - 5-tier roadmap
   - Time estimates

3. **BENCHMARKS.md** (5 min)
   - Performance gains

---

## File-by-File Guide

### Documentation (Read First)
| File | Size | Purpose | Time |
|------|------|---------|------|
| README.md | 10 KB | Overview | 5 min |
| COMPLETION_SUMMARY.md | 15 KB | Status | 10 min |
| DEVELOPER.md | 24 KB | Architecture | 30 min |
| IMPROVEMENT_PLAN.md | 20 KB | Roadmap | 15 min |
| BENCHMARKS.md | 8 KB | Metrics | 10 min |
| BUILD.md | 10 KB | Building | 5 min |

### Source Code (Read Next)
| File | Lines | Purpose | Time |
|------|-------|---------|------|
| src/circuits.rs | 540 | Gate logic | 20 min |
| src/evolution.rs | 360 | GA operators | 15 min |
| src/interpreter.rs | 520 | Execution | 20 min |

---

## Quick Reference

### Commands

```
cd /home/thearchitect/OMC
cargo build --release      # Build
cargo test --release       # Test
./standalone.omc examples/hello_world.omc  # Run
```

### Key Files by Objective

| Goal | Read This |
|------|-----------|
| Understand project | README.md |
| Learn architecture | DEVELOPER.md |
| Optimize performance | BENCHMARKS.md |
| Add feature | DEVELOPER.md + src/circuits.rs |
| Debug issue | DEVELOPER.md |
| See roadmap | IMPROVEMENT_PLAN.md |

---

## Recommended Reading Paths

### Path 1: Complete Overview (2 hours)
1. README.md
2. COMPLETION_SUMMARY.md
3. BENCHMARKS.md
4. DEVELOPER.md
5. IMPROVEMENT_PLAN.md
6. Build and run examples
7. Study src/circuits.rs
8. Study src/evolution.rs

### Path 2: Developer Setup (1 hour)
1. BUILD.md
2. Build project
3. Run tests
4. DEVELOPER.md - "Module Breakdown"
5. Study src/circuits.rs
6. Run examples

### Path 3: Quick Start (15 minutes)
1. README.md
2. BUILD.md
3. Build and run hello_world.omc

---

## Key Concepts (Glossary)

| Term | Explanation |
|------|-------------|
| Circuit | DAG of logic gates producing output |
| Gate | Basic logic operation (xAND, xOR, etc.) |
| Hard eval | Boolean evaluation (true/false) |
| Soft eval | Probabilistic evaluation (0.0-1.0) |
| Mutation | Random gate modifications |
| Crossover | Breeding operation combining parents |
| Fitness | Score measuring circuit correctness |
| GA | Genetic Algorithm - evolution framework |
| DAG | Directed Acyclic Graph structure |

---

**Start with README.md, then follow your role/goal path.** ✨
