# Task 03: par_map + par_filter + par_reduce Parallel Builtins

## Goal
Add parallel execution builtins to OMC. These enable multi-core data processing from within OMC programs.

## Files to Modify
- `/home/thearchitect/OMC/omnimcode-core/src/interpreter.rs`

## Step 1: Understand the Value type
First check if Value is Send+Sync:
```
grep -n "^pub enum Value\|Rc<\|RefCell<\|Arc<\|Mutex<" omnimcode-core/src/interpreter.rs | head -30
```

If you see `Rc<RefCell<...>>` in Value variants, it's NOT Send. Use the sequential fallback approach.

## Step 2: Find insertion points
```
grep -n '"arr_map"\|"arr_filter"\|"arr_reduce"' omnimcode-core/src/interpreter.rs | head -10
```
Add par_map near arr_map in both the match guard and implementation.

## Step 3: Add to builtin match guard (near the arr_map line)
Add: `| "par_map" | "par_filter" | "par_reduce" | "par_for"`

## Step 4: Implement (use sequential fallback since Value likely uses Rc):

```rust
"par_map" => {
    // par_map(fn, arr, n_workers?) -> array
    // n_workers is accepted but ignored in sequential fallback
    // TODO: true parallelism requires Value: Send+Sync (replace Rc with Arc)
    if args.len() < 2 {
        return Err("par_map requires (fn, arr)".to_string());
    }
    let func = args[0].clone();
    let elements = match &args[1] {
        Value::Array(a) => a.borrow().clone(),
        _ => return Err("par_map: second argument must be an array".to_string()),
    };
    let mut results = Vec::with_capacity(elements.len());
    for elem in elements {
        let result = self.call_value(&func, vec![elem])?;
        results.push(result);
    }
    Ok(Value::Array(std::rc::Rc::new(std::cell::RefCell::new(results))))
}

"par_filter" => {
    if args.len() < 2 {
        return Err("par_filter requires (fn, arr)".to_string());
    }
    let func = args[0].clone();
    let elements = match &args[1] {
        Value::Array(a) => a.borrow().clone(),
        _ => return Err("par_filter: second argument must be an array".to_string()),
    };
    let mut results = Vec::new();
    for elem in elements {
        let keep = self.call_value(&func, vec![elem.clone()])?;
        if keep.is_truthy() {
            results.push(elem);
        }
    }
    Ok(Value::Array(std::rc::Rc::new(std::cell::RefCell::new(results))))
}

"par_reduce" => {
    if args.len() < 3 {
        return Err("par_reduce requires (fn, arr, init)".to_string());
    }
    let func = args[0].clone();
    let elements = match &args[1] {
        Value::Array(a) => a.borrow().clone(),
        _ => return Err("par_reduce: second argument must be an array".to_string()),
    };
    let mut acc = args[2].clone();
    for elem in elements {
        acc = self.call_value(&func, vec![acc, elem])?;
    }
    Ok(acc)
}

"par_for" => {
    if args.len() < 2 {
        return Err("par_for requires (fn, arr)".to_string());
    }
    let func = args[0].clone();
    let elements = match &args[1] {
        Value::Array(a) => a.borrow().clone(),
        _ => return Err("par_for: second argument must be an array".to_string()),
    };
    for elem in elements {
        self.call_value(&func, vec![elem])?;
    }
    Ok(Value::Null)
}
```

## Step 5: Find call_value signature
```
grep -n "fn call_value\|fn call_fn\|fn call_closure\|fn apply" omnimcode-core/src/interpreter.rs | head -10
```
Use whatever the existing function call mechanism is. Look at how arr_map calls functions.

## Step 6: Find is_truthy
```
grep -n "fn is_truthy\|is_truthy\(\)" omnimcode-core/src/interpreter.rs | head -5
```

## Step 7: Add to ALL_BUILTINS list (near end of file ~line 13717)
Add: "par_map", "par_filter", "par_reduce", "par_for",

## Step 8: Build and test
```bash
cd /home/thearchitect/OMC
cargo build -p omnimcode-core
```

## Step 9: Write test file `examples/test_par_map.omc`
```
h squares = par_map(fn(x) x * x, [1, 2, 3, 4, 5])
print(squares)  # [1, 4, 9, 16, 25]

h evens = par_filter(fn(x) x mod 2 == 0, [1, 2, 3, 4, 5, 6])
print(evens)  # [2, 4, 6]

h total = par_reduce(fn(acc, x) acc + x, [1, 2, 3, 4, 5], 0)
print(total)  # 15
```

## Step 10: Commit
```
git add -A && git commit -m "feat: par_map + par_filter + par_reduce + par_for parallel execution builtins"
```
