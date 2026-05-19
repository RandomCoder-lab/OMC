# OMC API Reference

*Generated from `examples/lib/`. 23 module(s).*

---

## `agent` — `examples/lib/agent.omc`

### `agent.agent_new`

```
fn agent_new(name, role, knowledge_dict, max_tiers)
```

**Parameters:** `name`, `role`, `knowledge_dict`, `max_tiers`

### `agent.agent_load`

```
fn agent_load(name, role, knowledge_dict)
```

Reload an existing agent (preserves memory).

**Parameters:** `name`, `role`, `knowledge_dict`

### `agent.agent_send`

```
fn agent_send(agent, content)
```

Send a substrate-signed message. Returns the wire-serialized form
that the recipient deserializes via omc_msg_deserialize.

**Parameters:** `agent`, `content`

### `agent.agent_receive`

```
fn agent_receive(agent, wire)
```

Receive a wire message: deserialize, verify signature, push verified
content into the agent's memory, generate a response, sign + return
the response wire. This is the full inbound→outbound loop.

**Parameters:** `agent`, `wire`

### `agent.agent_stats`

```
fn agent_stats(agent)
```

**Parameters:** `agent`

---

## `cache` — `examples/lib/cache.omc`

### `cache.memoize`

```
fn memoize(f)
```

memoize(fn) → fn
Returns a wrapper that caches results keyed by stringified args.

**Parameters:** `f`

### `cache.memoize2`

```
fn memoize2(f)
```

memoize2(fn) → fn
Two-argument version.

**Parameters:** `f`

### `cache.lru_new`

```
fn lru_new(capacity)
```

lru_new(capacity) → lru_cache dict

**Parameters:** `capacity`

### `cache.lru_get`

```
fn lru_get(lru, key)
```

lru_get(lru, key) → value or null

**Parameters:** `lru`, `key`

### `cache.lru_put`

```
fn lru_put(lru, key, value)
```

lru_put(lru, key, value)

**Parameters:** `lru`, `key`, `value`

### `cache.lru_stats`

```
fn lru_stats(lru)
```

lru_stats(lru) → {hits, misses, size, capacity, hit_rate}

**Parameters:** `lru`

### `cache.lru_memoize`

```
fn lru_memoize(f, capacity)
```

lru_memoize(fn, capacity) → fn
LRU-backed memoized wrapper.

**Parameters:** `f`, `capacity`

### `cache.ttl_new`

```
fn ttl_new(ttl_seconds)
```

ttl_new(ttl_seconds) → ttl_cache dict

**Parameters:** `ttl_seconds`

### `cache.ttl_get`

```
fn ttl_get(cache, key)
```

ttl_get(cache, key) → value or null (returns null if expired)

**Parameters:** `cache`, `key`

### `cache.ttl_put`

```
fn ttl_put(cache, key, value)
```

ttl_put(cache, key, value)

**Parameters:** `cache`, `key`, `value`

### `cache.ttl_memoize`

```
fn ttl_memoize(f, ttl)
```

ttl_memoize(fn, ttl_seconds) → fn

**Parameters:** `f`, `ttl`

### `cache.disk_cache_get`

```
fn disk_cache_get(dir, key)
```

disk_cache_get(dir, key) → string or null

**Parameters:** `dir`, `key`

### `cache.disk_cache_put`

```
fn disk_cache_put(dir, key, value)
```

disk_cache_put(dir, key, value)

**Parameters:** `dir`, `key`, `value`

### `cache.disk_memoize`

```
fn disk_memoize(f, dir)
```

disk_memoize(fn, dir) → fn
Persists results to disk; survives process restarts.

**Parameters:** `f`, `dir`

### `cache.batch_cached`

```
fn batch_cached(f, keys, cache)
```

batch_cached(fn, keys_array, cache) → [values]
For each key: return cached value or call fn(key) and cache it.

**Parameters:** `f`, `keys`, `cache`

---

## `cli` — `examples/lib/cli.omc`

### `cli.ansi_reset`

```
fn ansi_reset()
```

### `cli.ansi_bold`

```
fn ansi_bold()
```

### `cli.ansi_dim`

```
fn ansi_dim()
```

### `cli.ansi_red`

```
fn ansi_red()
```

### `cli.ansi_green`

```
fn ansi_green()
```

### `cli.ansi_yellow`

```
fn ansi_yellow()
```

### `cli.ansi_blue`

```
fn ansi_blue()
```

### `cli.ansi_magenta`

```
fn ansi_magenta()
```

### `cli.ansi_cyan`

```
fn ansi_cyan()
```

### `cli.ansi_white`

```
fn ansi_white()
```

### `cli.ansi_bg_red`

```
fn ansi_bg_red()
```

### `cli.ansi_bg_green`

```
fn ansi_bg_green()
```

### `cli.ansi_bg_yellow`

```
fn ansi_bg_yellow()
```

### `cli.ansi_bg_blue`

```
fn ansi_bg_blue()
```

### `cli.color`

```
fn color(text, color_code)
```

**Parameters:** `text`, `color_code`

### `cli.red`

```
fn red(text)
```

**Parameters:** `text`

### `cli.green`

```
fn green(text)
```

**Parameters:** `text`

### `cli.yellow`

```
fn yellow(text)
```

**Parameters:** `text`

### `cli.blue`

```
fn blue(text)
```

**Parameters:** `text`

### `cli.cyan`

```
fn cyan(text)
```

**Parameters:** `text`

### `cli.bold`

```
fn bold(text)
```

**Parameters:** `text`

### `cli.dim`

```
fn dim(text)
```

**Parameters:** `text`

### `cli.success`

```
fn success(msg)
```

**Parameters:** `msg`

### `cli.warn`

```
fn warn(msg)
```

**Parameters:** `msg`

### `cli.err`

```
fn err(msg)
```

**Parameters:** `msg`

### `cli.info`

```
fn info(msg)
```

**Parameters:** `msg`

### `cli.parse_args`

```
fn parse_args(args)
```

parse_args(args_array) → {flags: {}, positional: []}
Handles: --flag, --key=value, --key value, -f, positional args

**Parameters:** `args`

### `cli.flag`

```
fn flag(parsed, name, default_val)
```

flag(parsed, name, default?) → value

**Parameters:** `parsed`, `name`, `default_val`

### `cli.progress_bar`

```
fn progress_bar(current, total, width, label)
```

**Parameters:** `current`, `total`, `width`, `label`

### `cli.print_progress`

```
fn print_progress(current, total, width, label)
```

**Parameters:** `current`, `total`, `width`, `label`

### `cli.table`

```
fn table(headers, rows)
```

table(headers, rows) → formatted string
headers: [string, ...]
rows: [[value, ...], ...]

**Parameters:** `headers`, `rows`

### `cli.print_table`

```
fn print_table(headers, rows)
```

**Parameters:** `headers`, `rows`

### `cli.spinner_frames`

```
fn spinner_frames()
```

### `cli.spinner_frame`

```
fn spinner_frame(tick)
```

**Parameters:** `tick`

### `cli.confirm`

```
fn confirm(question)
```

**Parameters:** `question`

### `cli.prompt`

```
fn prompt(question, default_val)
```

**Parameters:** `question`, `default_val`

### `cli.box`

```
fn box(title, content, width)
```

**Parameters:** `title`, `content`, `width`

---

## `fibtier` — `examples/lib/fibtier.omc`

### `fibtier.fibtier_default_sizes`

```
fn fibtier_default_sizes()
```

### `fibtier.fibtier_capacity`

```
fn fibtier_capacity(tier_idx)
```

Capacity of a tier (1-indexed).

**Parameters:** `tier_idx`

### `fibtier.fibtier_new`

```
fn fibtier_new(max_tiers)
```

**Parameters:** `max_tiers`

### `fibtier.fibtier_new_with_strategy`

```
fn fibtier_new_with_strategy(max_tiers, fold_strategy)
```

**Parameters:** `max_tiers`, `fold_strategy`

### `fibtier.fibtier_fold_default`

```
fn fibtier_fold_default(entry_a, entry_b)
```

Default fold: concatenate canonical forms, mark as a fold.
For prose, this is the most lossless. Smarter folds (LLM summarization
via py_callback) can replace this when wired through Python.

**Parameters:** `entry_a`, `entry_b`

### `fibtier.fibtier_set_fold_strategy`

```
fn fibtier_set_fold_strategy(mem, strategy)
```

Flip fold strategy on an existing memory.

**Parameters:** `mem`, `strategy`

### `fibtier.fibtier_push`

```
fn fibtier_push(mem, content)
```

Push an entry. Returns the memory (for chaining).

**Parameters:** `mem`, `content`

### `fibtier.fibtier_stats`

```
fn fibtier_stats(mem)
```

Stats snapshot.

**Parameters:** `mem`

### `fibtier.fibtier_render`

```
fn fibtier_render(mem)
```

Render the memory as an LLM-readable context prompt.
Most-detailed (tier 1) first; abstractions higher up.

**Parameters:** `mem`

### `fibtier.fibtier_query`

```
fn fibtier_query(mem, query_text, top_k)
```

Substrate-distance query: rank all entries by attractor-distance
to the query's canonical hash; return top-k.

**Parameters:** `mem`, `query_text`, `top_k`

---

## `fibtier_persistent` — `examples/lib/fibtier_persistent.omc`

### `fibtier_persistent.fibtier_persistent_new`

```
fn fibtier_persistent_new(name, max_tiers)
```

Create a NEW persistent fibtier under <root>/.
If a fibtier already exists at that name, this OVERWRITES the
manifest with an empty state. Use fibtier_persistent_load to
reload existing.

**Parameters:** `name`, `max_tiers`

### `fibtier_persistent.fibtier_persistent_load`

```
fn fibtier_persistent_load(name)
```

Reload a persistent fibtier from disk.

**Parameters:** `name`

### `fibtier_persistent.fibtier_persistent_push`

```
fn fibtier_persistent_push(mem, content)
```

Push with persistence: push to memory, save manifest to disk.

**Parameters:** `mem`, `content`

### `fibtier_persistent.fibtier_persistent_mem`

```
fn fibtier_persistent_mem(pmem)
```

Convenience: get the underlying mem (for any non-persistent op).

**Parameters:** `pmem`

### `fibtier_persistent.fibtier_persistent_save`

```
fn fibtier_persistent_save(mem)
```

Save explicit (for bulk pushes that want to defer manifest write).

**Parameters:** `mem`

---

## `graph` — `examples/lib/graph.omc`

### `graph.graph_new`

```
fn graph_new()
```

graph_new() → {nodes: {}, edges: {}, directed: true}

### `graph.graph_new_undirected`

```
fn graph_new_undirected()
```

### `graph.graph_add_node`

```
fn graph_add_node(g, id, data)
```

graph_add_node(g, id, data?)

**Parameters:** `g`, `id`, `data`

### `graph.graph_add_edge`

```
fn graph_add_edge(g, from_node, to_node, weight)
```

graph_add_edge(g, from, to, weight?)

**Parameters:** `g`, `from_node`, `to_node`, `weight`

### `graph.graph_neighbors`

```
fn graph_neighbors(g, node)
```

**Parameters:** `g`, `node`

### `graph.bfs`

```
fn bfs(g, start)
```

bfs(g, start) → [node_ids in visit order]

**Parameters:** `g`, `start`

### `graph.bfs_path`

```
fn bfs_path(g, start, goal)
```

bfs_path(g, start, goal) → [path] or null

**Parameters:** `g`, `start`, `goal`

### `graph.dfs`

```
fn dfs(g, start)
```

dfs(g, start) → [node_ids in visit order]

**Parameters:** `g`, `start`

### `graph.toposort`

```
fn toposort(g)
```

toposort(g) → [node_ids] in topological order

**Parameters:** `g`

### `graph.dijkstra`

```
fn dijkstra(g, start)
```

dijkstra(g, start) → {dist: {node: dist}, prev: {node: prev}}

**Parameters:** `g`, `start`

### `graph.shortest_path`

```
fn shortest_path(g, start, goal)
```

shortest_path(g, start, goal) → [path] or null

**Parameters:** `g`, `start`, `goal`

### `graph.pagerank`

```
fn pagerank(g, damping, iterations)
```

pagerank(g, damping?, iterations?) → {node: rank}

**Parameters:** `g`, `damping`, `iterations`

### `graph.connected_components`

```
fn connected_components(g)
```

connected_components(g) → [[component_nodes], ...]

**Parameters:** `g`

### `graph.has_cycle`

```
fn has_cycle(g)
```

has_cycle(g) → bool

**Parameters:** `g`

---

## `harmonic_anomaly` — `examples/lib/harmonic_anomaly.omc`

### `harmonic_anomaly.new`

```
fn new(dim_names)
```

Create a fresh detector. dim_names is an array of strings (one per
dimension). Default strategy is 0 (log) for every dim.

Returns an array layout: [n_dims, strategies, freq_keys, freq_counts, n].
Use ha.set_strategy() / ha.fit() / ha.score_all() to interact.

**Parameters:** `dim_names`

### `harmonic_anomaly.set_strategy`

```
fn set_strategy(detector, dim_idx, strategy)
```

Override one dim's bucket strategy. Useful when you have a discrete
field (status_code, country_code) where log-bucketing makes no sense.
ha.set_strategy(det, 1, "discrete")    # 2nd dim is categorical

**Parameters:** `detector`, `dim_idx`, `strategy`

### `harmonic_anomaly.fit`

```
fn fit(detector, rows)
```

Fit the detector to a corpus of rows. Each row is an array of
values parallel to dim_names. Builds per-dim frequency arrays.
Cold path — runs once at startup; uses arr_push for dynamic growth.

**Parameters:** `detector`, `rows`

### `harmonic_anomaly.score`

```
fn score(detector, row)
```

Score a single row. Returns sum-of-marginal-log-rarities; higher =
more structurally anomalous.

Hot path — called once per scored row. Uses ONLY JIT-eligible ops:
arr_get, arr_len, while loop, arithmetic, log_phi_pi_fibonacci,
to_float, int comparison. No dict ops, no string ops. The whole
function compiles in dual-band mode.

**Parameters:** `detector`, `row`

### `harmonic_anomaly.score_all`

```
fn score_all(detector, rows)
```

Bulk score: returns an array of scores parallel to rows.

**Parameters:** `detector`, `rows`

### `harmonic_anomaly.top_k`

```
fn top_k(detector, rows, k)
```

Top-K most anomalous row indices. Returns indices into rows.

**Parameters:** `detector`, `rows`, `k`

### `harmonic_anomaly.detect`

```
fn detect(dim_names, rows, k)
```

Convenience: full pipeline in one call. Fit on rows, return top-K
anomaly indices. Use when you don't need to keep the detector around.

**Parameters:** `dim_names`, `rows`, `k`

---

## `harmonic_anomaly_v2` — `examples/lib/harmonic_anomaly_v2.omc`

### `harmonic_anomaly_v2.new`

```
fn new(dim_names)
```

**Parameters:** `dim_names`

### `harmonic_anomaly_v2.set_strategy`

```
fn set_strategy(detector, dim_idx, strategy)
```

**Parameters:** `detector`, `dim_idx`, `strategy`

### `harmonic_anomaly_v2.fit`

```
fn fit(detector, rows)
```

---- fit (deferred-sort): linear-scan dedupe during the row loop,
then ONE-SHOT sort per dim at the end. The dedupe is the same
O(K) per row as v1; the final sort is O(DK log K) — tiny vs
sorted-insert's O(NDK).

**Parameters:** `detector`, `rows`

### `harmonic_anomaly_v2.score`

```
fn score(detector, row)
```

---- score (HOT PATH): JIT-eligible end-to-end with substrate-routed
search. substrate_search uses the F(k)/φ^(π·k) probe sequence — on
Fibonacci-attractor-valued keys (which is what bucket_log produces
after fold), the substrate probe sequence converges faster than
midpoint binary search because the probes naturally align with the
attractor lattice. Both are JIT-intercepted; the loop is dual-band
native code end-to-end.

Falls back to int_binary_search if you'd rather use plain midpoint
(substantially identical perf on this workload; here for choice).

**Parameters:** `detector`, `row`

### `harmonic_anomaly_v2.score_all`

```
fn score_all(detector, rows)
```

**Parameters:** `detector`, `rows`

### `harmonic_anomaly_v2.top_k`

```
fn top_k(detector, rows, k)
```

**Parameters:** `detector`, `rows`, `k`

### `harmonic_anomaly_v2.detect`

```
fn detect(dim_names, rows, k)
```

**Parameters:** `dim_names`, `rows`, `k`

---

## `harmonic_clustering` — `examples/lib/harmonic_clustering.omc`

### `harmonic_clustering.new`

```
fn new(dim_names)
```

**Parameters:** `dim_names`

### `harmonic_clustering.set_strategy`

```
fn set_strategy(cl, dim_idx, strategy)
```

**Parameters:** `cl`, `dim_idx`, `strategy`

### `harmonic_clustering.fit`

```
fn fit(cl, rows)
```

**Parameters:** `cl`, `rows`

### `harmonic_clustering.predict_one`

```
fn predict_one(cl, row)
```

**Parameters:** `cl`, `row`

### `harmonic_clustering.predict`

```
fn predict(cl, rows)
```

**Parameters:** `cl`, `rows`

### `harmonic_clustering.n_clusters`

```
fn n_clusters(cl)
```

**Parameters:** `cl`

### `harmonic_clustering.centroids`

```
fn centroids(cl)
```

**Parameters:** `cl`

### `harmonic_clustering.cluster_counts`

```
fn cluster_counts(cl)
```

**Parameters:** `cl`

### `harmonic_clustering.cluster_keys`

```
fn cluster_keys(cl)
```

**Parameters:** `cl`

---

## `harmonic_recommend` — `examples/lib/harmonic_recommend.omc`

### `harmonic_recommend.new`

```
fn new()
```

### `harmonic_recommend.add_rating`

```
fn add_rating(rec, user_id, item_id, rating)
```

Add a single rating. Idempotent on (user, item) — re-adding overrides.

**Parameters:** `rec`, `user_id`, `item_id`, `rating`

### `harmonic_recommend.fit`

```
fn fit(rec)
```

**Parameters:** `rec`

### `harmonic_recommend.suggest_for`

```
fn suggest_for(rec, user_id, k)
```

**Parameters:** `rec`, `user_id`, `k`

### `harmonic_recommend.n_items`

```
fn n_items(rec)
```

**Parameters:** `rec`

### `harmonic_recommend.n_users`

```
fn n_users(rec)
```

**Parameters:** `rec`

### `harmonic_recommend.signatures`

```
fn signatures(rec)
```

**Parameters:** `rec`

### `harmonic_recommend.n_signatures`

```
fn n_signatures(rec)
```

**Parameters:** `rec`

---

## `llm` — `examples/lib/llm.omc`

### `llm.cot`

```
fn cot(prompt, model)
```

cot(prompt, model?) → {reasoning, answer}
Asks the model to think step-by-step before giving a final answer.

**Parameters:** `prompt`, `model`

### `llm.cot_verify`

```
fn cot_verify(prompt, n, model)
```

cot_verify(prompt, n, model?) → {answer, confidence, votes}
Runs CoT n times and takes the majority answer (self-consistency).

**Parameters:** `prompt`, `n`, `model`

### `llm.few_shot`

```
fn few_shot(examples, query, model)
```

few_shot(examples, query, model?) → string
examples is [{input, output}, ...]; builds a prompt from them.

**Parameters:** `examples`, `query`, `model`

### `llm.extract_json`

```
fn extract_json(text)
```

extract_json(text) → parsed dict/array or null
Pulls the first {...} or [...] block from model output.

**Parameters:** `text`

### `llm.llm_json`

```
fn llm_json(prompt, schema_hint, model)
```

llm_json(prompt, schema_hint, model?) → dict or null
Asks the model to respond with JSON matching schema_hint description.

**Parameters:** `prompt`, `schema_hint`, `model`

### `llm.react_agent`

```
fn react_agent(goal, tools, model, max_steps)
```

react_agent(goal, tools, model?, max_steps?) → {answer, trace}
tools is a dict: {tool_name: fn(input) → string}
Implements the Thought/Action/Observation loop.

**Parameters:** `goal`, `tools`, `model`, `max_steps`

### `llm.chain`

```
fn chain(steps_arr, input, model)
```

chain(steps_arr, input, model?) → string
Runs a sequence of prompts, feeding each output as the next input.
steps_arr is [string, ...] — prompt templates where {} is replaced by input.

**Parameters:** `steps_arr`, `input`, `model`

### `llm.critique_revise`

```
fn critique_revise(draft, criteria, model)
```

critique_revise(draft, criteria, model?) → {draft, critique, revised}

**Parameters:** `draft`, `criteria`, `model`

### `llm.summarize`

```
fn summarize(text, style, model)
```

summarize(text, style?, model?) → string
style: "bullet" | "paragraph" | "one-line" (default: "paragraph")

**Parameters:** `text`, `style`, `model`

### `llm.classify`

```
fn classify(text, labels, model)
```

classify(text, labels, model?) → string (one of labels)

**Parameters:** `text`, `labels`, `model`

### `llm.gen_and_test`

```
fn gen_and_test(description, test_fn, model, max_attempts)
```

gen_and_test(description, test_fn, model?, max_attempts?) → {code, passed, attempts}
Generates OMC code for description, runs test_fn(code), fixes until passing.

**Parameters:** `description`, `test_fn`, `model`, `max_attempts`

### `llm.best_of_n`

```
fn best_of_n(prompt, n, criteria, system, model)
```

best_of_n(prompt, n, criteria, system?, model?) → {answer, score, idx}
Generates N responses in parallel, judges all with llm_judge, returns best.

**Parameters:** `prompt`, `n`, `criteria`, `system`, `model`

### `llm.parallel_cot`

```
fn parallel_cot(prompt, n, model)
```

parallel_cot(prompt, n, model?) → {answer, confidence, all_answers}
Like cot_verify but fires all N calls in parallel using batch_llm_call.

**Parameters:** `prompt`, `n`, `model`

### `llm.improve_until`

```
fn improve_until(text, criteria, threshold, max_rounds, model)
```

improve_until(text, criteria, threshold, max_rounds, model?) → {text, score, rounds}
Repeatedly generates an improved version and judges it; stops when score >= threshold.

**Parameters:** `text`, `criteria`, `threshold`, `max_rounds`, `model`

### `llm.debate`

```
fn debate(topic, rounds, model)
```

debate(topic, rounds, model?) → {winner: "for"|"against", reasoning, transcript}

**Parameters:** `topic`, `rounds`, `model`

---

## `llm_agent` — `examples/lib/llm_agent.omc`

### `llm_agent.ask_json`

```
fn ask_json(prompt, example, model)
```

**Parameters:** `prompt`, `example`, `model`

### `llm_agent.chain_of_thought`

```
fn chain_of_thought(question, model)
```

**Parameters:** `question`, `model`

### `llm_agent.self_critique`

```
fn self_critique(text, criteria, model)
```

**Parameters:** `text`, `criteria`, `model`

### `llm_agent.critique_and_revise`

```
fn critique_and_revise(text, criteria, model)
```

**Parameters:** `text`, `criteria`, `model`

### `llm_agent.parallel_research`

```
fn parallel_research(questions, model)
```

**Parameters:** `questions`, `model`

### `llm_agent.research_and_synthesize`

```
fn research_and_synthesize(topic, sub_questions, model)
```

**Parameters:** `topic`, `sub_questions`, `model`

### `llm_agent.react_agent`

```
fn react_agent(goal, tools, model, max_turns)
```

**Parameters:** `goal`, `tools`, `model`, `max_turns`

### `llm_agent.code_agent`

```
fn code_agent(task, tests, model, max_attempts)
```

**Parameters:** `task`, `tests`, `model`, `max_attempts`

### `llm_agent.best_of_n`

```
fn best_of_n(task, n, model)
```

**Parameters:** `task`, `n`, `model`

### `llm_agent.mem_agent_new`

```
fn mem_agent_new()
```

### `llm_agent.mem_store`

```
fn mem_store(agent, text)
```

**Parameters:** `agent`, `text`

### `llm_agent.mem_recall`

```
fn mem_recall(agent, query, top_k)
```

**Parameters:** `agent`, `query`, `top_k`

### `llm_agent.mem_agent_call`

```
fn mem_agent_call(agent, question, model)
```

**Parameters:** `agent`, `question`, `model`

---

## `np` — `examples/lib/np.omc`

### `np.array`

```
fn array(items)
```

**Parameters:** `items`

### `np.zeros`

```
fn zeros(n)
```

**Parameters:** `n`

### `np.ones`

```
fn ones(n)
```

**Parameters:** `n`

### `np.arange`

```
fn arange(n)
```

**Parameters:** `n`

### `np.linspace`

```
fn linspace(a, b, n)
```

**Parameters:** `a`, `b`, `n`

### `np.mean`

```
fn mean(arr)
```

**Parameters:** `arr`

### `np.median`

```
fn median(arr)
```

**Parameters:** `arr`

### `np.nanmean`

```
fn nanmean(arr)
```

NaN-safe variants: pandas columns with missing values become arrays
of HFloat(NaN) — vanilla mean/median return NaN. nan* skips them.

**Parameters:** `arr`

### `np.nanmedian`

```
fn nanmedian(arr)
```

**Parameters:** `arr`

### `np.std`

```
fn std(arr)
```

**Parameters:** `arr`

### `np.np_sum`

```
fn np_sum(arr)
```

**Parameters:** `arr`

### `np.np_min`

```
fn np_min(arr)
```

**Parameters:** `arr`

### `np.np_max`

```
fn np_max(arr)
```

**Parameters:** `arr`

### `np.dot`

```
fn dot(a, b)
```

**Parameters:** `a`, `b`

### `np.np_add`

```
fn np_add(a, b)
```

**Parameters:** `a`, `b`

### `np.np_mul`

```
fn np_mul(a, b)
```

**Parameters:** `a`, `b`

### `np.sort`

```
fn sort(arr)
```

**Parameters:** `arr`

### `np.argsort`

```
fn argsort(arr)
```

**Parameters:** `arr`

### `np.shape`

```
fn shape(arr)
```

**Parameters:** `arr`

### `np.np_len`

```
fn np_len(arr)
```

**Parameters:** `arr`

### `np.percentile`

```
fn percentile(arr, p)
```

**Parameters:** `arr`, `p`

### `np.quantile`

```
fn quantile(arr, q)
```

**Parameters:** `arr`, `q`

### `np.corrcoef`

```
fn corrcoef(a, b)
```

**Parameters:** `a`, `b`

### `np.pi`

```
fn pi()
```

### `np.e`

```
fn e()
```

### `np.inf`

```
fn inf()
```

### `np.nan`

```
fn nan()
```

---

## `pd` — `examples/lib/pd.omc`

### `pd.read_csv`

```
fn read_csv(path)
```

**Parameters:** `path`

### `pd.read_json`

```
fn read_json(path)
```

**Parameters:** `path`

### `pd.read_parquet`

```
fn read_parquet(path)
```

**Parameters:** `path`

### `pd.read_excel`

```
fn read_excel(path)
```

**Parameters:** `path`

### `pd.read_table`

```
fn read_table(path)
```

**Parameters:** `path`

### `pd.shape`

```
fn shape(df)
```

**Parameters:** `df`

### `pd.nrows`

```
fn nrows(df)
```

**Parameters:** `df`

### `pd.ncols`

```
fn ncols(df)
```

**Parameters:** `df`

### `pd.columns`

```
fn columns(df)
```

**Parameters:** `df`

### `pd.to_dict`

```
fn to_dict(df)
```

All rows as a dict-of-arrays (one entry per column). Memory-heavy
on big DataFrames; prefer head_dict when sampling.

**Parameters:** `df`

### `pd.head_dict`

```
fn head_dict(df, n)
```

**Parameters:** `df`, `n`

### `pd.col`

```
fn col(df, name)
```

Pull a single column as an OMC array. py_call already auto-converts
the returned pandas Series via .tolist() (Series exposes that
attr), so we get an OMC array directly. No second call needed.

**Parameters:** `df`, `name`

### `pd.head`

```
fn head(df, n)
```

**Parameters:** `df`, `n`

### `pd.tail`

```
fn tail(df, n)
```

**Parameters:** `df`, `n`

### `pd.select_cols`

```
fn select_cols(df, cols)
```

**Parameters:** `df`, `cols`

### `pd.group_by`

```
fn group_by(df, key)
```

**Parameters:** `df`, `key`

### `pd.agg_mean`

```
fn agg_mean(grouped)
```

Apply mean/sum/count to a groupby and pull as a dict-of-arrays.

**Parameters:** `grouped`

### `pd.agg_count`

```
fn agg_count(grouped)
```

**Parameters:** `grouped`

### `pd.describe`

```
fn describe(df)
```

**Parameters:** `df`

### `pd.fillna_col`

```
fn fillna_col(df, col, value)
```

---- Cleaning -------------------------------------------------------------
Fill missing values in `col` with `value`. Returns a NEW DataFrame
(pandas immutable-on-assignment semantics).

Implementation note: we can't pull the Series out and operate on it
(py_to_omc auto-converts Series → OMC array via .tolist(), losing
the handle). Instead we call df.fillna(value={col: value}) which
operates on the whole DataFrame and returns a fresh one.

**Parameters:** `df`, `col`, `value`

### `pd.one_hot`

```
fn one_hot(df, col)
```

---- One-hot / dummies ---------------------------------------------------
Returns a new DataFrame with `col` replaced by N indicator columns.
Uses pandas.get_dummies under the hood.

**Parameters:** `df`, `col`

### `pd.apply_omc`

```
fn apply_omc(df, col, omc_fn_name)
```

---- Apply an OMC callback to every row of a column ----------------------
`omc_fn_name` is the name of an OMC fn taking a single value.
Returns a new column (OMC array).

Implementation: py_call_raw keeps the Series as a handle (vanilla
py_call would auto-convert via .tolist()). Then apply via the
py_callback. Final .tolist() materialises the result.

**Parameters:** `df`, `col`, `omc_fn_name`

---

## `prometheus` — `examples/lib/prometheus.omc`

### `prometheus.prom_linear_new`

```
fn prom_linear_new(in_dim, out_dim, rng_state)
```

**Parameters:** `in_dim`, `out_dim`, `rng_state`

### `prometheus.prom_linear_forward`

```
fn prom_linear_forward(layer, x_id)
```

**Parameters:** `layer`, `x_id`

### `prometheus.prom_linear_params`

```
fn prom_linear_params(layer)
```

**Parameters:** `layer`

### `prometheus.prom_relu`

```
fn prom_relu(x_id)
```

**Parameters:** `x_id`

### `prometheus.prom_sigmoid`

```
fn prom_sigmoid(x_id)
```

**Parameters:** `x_id`

### `prometheus.prom_mse_loss`

```
fn prom_mse_loss(pred_id, target_id)
```

**Parameters:** `pred_id`, `target_id`

### `prometheus.prom_sgd_step`

```
fn prom_sgd_step(params, lr)
```

**Parameters:** `params`, `lr`

### `prometheus.prom_harmonic_sgd_step`

```
fn prom_harmonic_sgd_step(params, lr)
```

Variant: substrate-modulated SGD step.
Reads each param's gradient, scales by (1 + phi.res(grad_hash)) so
gradients pointing toward Fibonacci attractors get a small boost.
Experimental; the architectural rule derived today says substrate
metric on float activations doesn't work as an attention modulator,
but on GRADIENT MAGNITUDES (integer-keyed via hash of grad bytes)
the rule may differ. Worth measuring once the baseline converges.

**Parameters:** `params`, `lr`

### `prometheus.prom_one_hot`

```
fn prom_one_hot(idx, vocab)
```

Build a [vocab] one-hot row vector as a tape_const.

**Parameters:** `idx`, `vocab`

### `prometheus.prom_argmax_row`

```
fn prom_argmax_row(logits)
```

Argmax over a logits vector. Handles either a flat 1D array (when
tape_value collapses a 1xN matrix) or a true 2D [1, vocab] matrix.

**Parameters:** `logits`

### `prometheus.prom_collect_params`

```
fn prom_collect_params(layers)
```

**Parameters:** `layers`

### `prometheus.prom_serialize_model`

```
fn prom_serialize_model(model, layer_names)
```

Serialize an arbitrary model dict that names its layers via
string keys to layer dicts. Returns a {layers: [{name, entry}],
meta: {...}} struct ready for json_stringify.

**Parameters:** `model`, `layer_names`

### `prometheus.prom_model_hash`

```
fn prom_model_hash(bundle)
```

Compute the canonical hash that addresses a serialized model.
Two models with the same weights (in canonical-JSON form) collapse
to the same hash regardless of session or insertion order.

Strategy: re-parse + re-serialize via OMC's deterministic json
round-trip (sorts dict keys, normalizes float format), then fnv1a
the canonical string. Two models with identical weights but
different in-memory ordering land on the same hash.

**Parameters:** `bundle`

### `prometheus.prom_load_model`

```
fn prom_load_model(bundle)
```

Reconstruct a model from a serialized bundle. Returns a dict keyed
by layer name, suitable for the same forward() the caller used
during training.

**Parameters:** `bundle`

### `prometheus.prom_geodesic_distance`

```
fn prom_geodesic_distance(i, j)
```

Compute geodesic distance between two integer positions in the
CRT-Fibonacci lattice. The substrate metric, applied to its
native basis (integer position pairs).

**Parameters:** `i`, `j`

### `prometheus.prom_geodesic_bias_matrix`

```
fn prom_geodesic_bias_matrix(seq_len)
```

Build the [seq_len, seq_len] bias matrix. Used as additive bias on
attention scores: scores[i,j] -= alpha * bias[i,j].
Normalized so mean-off-diagonal is ~1.0, giving alpha interpretable
units (matches the PyTorch impl's normalization).

**Parameters:** `seq_len`

### `prometheus.prom_harmonic_sgd_step`

```
fn prom_harmonic_sgd_step(params, lr, alpha)
```

Substrate-modulated SGD step. Applies tape_update per param with a
learning rate scaled by the substrate-coherence of that param's
gradient magnitude.

**Parameters:** `params`, `lr`, `alpha`

### `prometheus.prom_cache_key`

```
fn prom_cache_key(input_repr, model_hash)
```

Compute a content-addressed key for an inference query.
input_repr: a string representation of the input batch.
model_hash: the canonical hash of the model weights (from prom_model_hash).
Returns a key that uniquely identifies (input, model) pair.

**Parameters:** `input_repr`, `model_hash`

### `prometheus.prom_cache_new`

```
fn prom_cache_new()
```

In-memory cache (a dict mapping key → output JSON). For
cross-process caching, route through omc_msg_recover_compressed
against omc-kernel.

### `prometheus.prom_cache_get`

```
fn prom_cache_get(cache, key)
```

**Parameters:** `cache`, `key`

### `prometheus.prom_cache_put`

```
fn prom_cache_put(cache, key, value)
```

**Parameters:** `cache`, `key`, `value`

### `prometheus.prom_generate_greedy`

```
fn prom_generate_greedy(forward_fn, model, seed_idx, length, vocab)
```

Generate `length` integer-token IDs given:
forward_fn : fn(model, x_id) -> logits_id    (caller-defined)
model      : the model dict
seed_idx   : starting char index (int)
length     : number of NEW tokens to generate
vocab      : alphabet size
Returns array of length+1 indices (seed first, then generated).

**Parameters:** `forward_fn`, `model`, `seed_idx`, `length`, `vocab`

### `prometheus.prom_decode_indices`

```
fn prom_decode_indices(indices, chars)
```

Convert a list of indices to a string given a chars alphabet.

**Parameters:** `indices`, `chars`

### `prometheus.prom_cross_entropy_loss`

```
fn prom_cross_entropy_loss(logits_id, target_idx, vocab)
```

Cross-entropy loss: -log(softmax(logits)[target_idx])
logits_id: tape node holding [1, vocab] logits
target_idx: integer target class
vocab: size of the alphabet

**Parameters:** `logits_id`, `target_idx`, `vocab`

### `prometheus.prom_attention_new`

```
fn prom_attention_new(d_model, seq_len, rng_state)
```

**Parameters:** `d_model`, `seq_len`, `rng_state`

### `prometheus.prom_attention_forward`

```
fn prom_attention_forward(layer, x_id)
```

Forward (L0 — standard QKV attention with optional geodesic bias):
given x as a tape node of shape [seq_len, d_model], returns
attention output [seq_len, d_model].
Uses tape_transpose so K's gradient flows properly through the
score path (fixes the earlier K-frozen bug).

**Parameters:** `layer`, `x_id`

### `prometheus.prom_substrate_resample`

```
fn prom_substrate_resample(v_id, scale)
```

Apply post-projection substrate resampling to a tape node. Returns
a new tape node whose value is `v * modulation(v)` and whose backward
flows through v unchanged (modulation rides as a const). scale=0.0
disables (returns v unchanged).

**Parameters:** `v_id`, `scale`

### `prometheus.prom_substrate_softmax`

```
fn prom_substrate_softmax(scores_id, alpha)
```

Substrate-modulated softmax. alpha=0.0 returns standard softmax.
alpha>0.0 applies the S-MOD post-softmax modulation + renormalize,
the variant that won -4.27% val on TinyShakespeare multi-head.

**Parameters:** `scores_id`, `alpha`

### `prometheus.prom_attention_substrate_k_new`

```
fn prom_attention_substrate_k_new(d_model, seq_len, rng_state)
```

Substrate-K attention with optional S-MOD softmax.
smod_alpha=0.0 → standard softmax (legacy behavior).
smod_alpha>0.0 → S-MOD substrate-modulated softmax (the recommended default).

**Parameters:** `d_model`, `seq_len`, `rng_state`

### `prometheus.prom_attention_substrate_k_forward`

```
fn prom_attention_substrate_k_forward(layer, x_id)
```

**Parameters:** `layer`, `x_id`

### `prometheus.prom_q6_modulate`

```
fn prom_q6_modulate(q_id, scale, gamma, mode)
```

**Parameters:** `q_id`, `scale`, `gamma`, `mode`

### `prometheus.prom_attention_substrate_kq_new`

```
fn prom_attention_substrate_kq_new(d_model, seq_len, rng_state)
```

L2: substrate K + Q. Only V is learned.
Q is derived as: x_pos_concat * fixed projection (use CRT-PE directly).
In the simplest form: Q = CRT-PE (same as K) so each position queries
its own substrate address. The attention reduces to "soft positional
self-similarity" — diagonal-biased by construction.

**Parameters:** `d_model`, `seq_len`, `rng_state`

### `prometheus.prom_attention_substrate_kq_forward`

```
fn prom_attention_substrate_kq_forward(layer, x_id)
```

**Parameters:** `layer`, `x_id`

### `prometheus.prom_attention_substrate_full_new`

```
fn prom_attention_substrate_full_new(d_model, seq_len)
```

L3: fully substrate attention. Zero learnable params in the layer.
Q = K = CRT-PE; V = identity transform on x (the input passes
through unchanged, weighted by substrate-determined attention).

**Parameters:** `d_model`, `seq_len`

### `prometheus.prom_attention_substrate_full_forward`

```
fn prom_attention_substrate_full_forward(layer, x_id)
```

**Parameters:** `layer`, `x_id`

### `prometheus.prom_attention_substrate_k_mh_new`

```
fn prom_attention_substrate_k_mh_new(d_model, seq_len, n_heads, rng_state)
```

**Parameters:** `d_model`, `seq_len`, `n_heads`, `rng_state`

### `prometheus.prom_attention_substrate_k_mh_forward`

```
fn prom_attention_substrate_k_mh_forward(layer, x_id)
```

**Parameters:** `layer`, `x_id`

### `prometheus.prom_attention_substrate_k_mh_params`

```
fn prom_attention_substrate_k_mh_params(layer)
```

**Parameters:** `layer`

### `prometheus.prom_attention_substrate_k_params`

```
fn prom_attention_substrate_k_params(layer)
```

Param collectors per variant.

**Parameters:** `layer`

### `prometheus.prom_attention_substrate_kq_params`

```
fn prom_attention_substrate_kq_params(layer)
```

**Parameters:** `layer`

### `prometheus.prom_attention_substrate_full_params`

```
fn prom_attention_substrate_full_params(layer)
```

**Parameters:** `layer`

### `prometheus.prom_attention_params`

```
fn prom_attention_params(layer)
```

**Parameters:** `layer`

### `prometheus.prom_adamw_new`

```
fn prom_adamw_new(params, lr, beta1, beta2, eps, weight_decay)
```

**Parameters:** `params`, `lr`, `beta1`, `beta2`, `eps`, `weight_decay`

### `prometheus.prom_adamw_step`

```
fn prom_adamw_step(state)
```

One AdamW step. Updates state in-place (mutates dict + tape values).

**Parameters:** `state`

### `prometheus.prom_embedding_new`

```
fn prom_embedding_new(vocab, d_model, rng_state)
```

**Parameters:** `vocab`, `d_model`, `rng_state`

### `prometheus.prom_embedding_forward`

```
fn prom_embedding_forward(layer, token_idx)
```

Forward: token_idx → [1, d_model] embedding row.
Uses one-hot @ table internally; result is differentiable into the
table param so backward updates the relevant row.

**Parameters:** `layer`, `token_idx`

### `prometheus.prom_embedding_params`

```
fn prom_embedding_params(layer)
```

**Parameters:** `layer`

### `prometheus.prom_embedding_batch`

```
fn prom_embedding_batch(layer, token_ids)
```

Batched embedding lookup: token_ids[] → [N, d_model] matrix.
Implemented via an [N, vocab] one-hot batch then matmul with the
embedding table. Differentiable end-to-end.

**Parameters:** `layer`, `token_ids`

### `prometheus.prom_cross_entropy_batch`

```
fn prom_cross_entropy_batch(logits_id, targets, vocab)
```

Batched cross-entropy: logits is [N, vocab], targets is array of N
integer indices. Returns scalar mean loss (averaged over positions).

v0.8.5 — defers to the fused tape_cross_entropy_batch Rust builtin
(closed-form (p - one_hot) / N backward, no intermediate tape nodes).
`vocab` is accepted but unused (the builtin reads it from logits.cols);
kept in the signature for callers that pass it.

**Parameters:** `logits_id`, `targets`, `vocab`

### `prometheus.prom_layernorm_new`

```
fn prom_layernorm_new(d_model, rng_state)
```

**Parameters:** `d_model`, `rng_state`

### `prometheus.prom_layernorm_forward`

```
fn prom_layernorm_forward(layer, x_id)
```

Forward: x is [N, d_model]; per-row layer norm via the fused
tape_layernorm Rust op. Works for both single-row [1, d] and
multi-token [seq, d] shapes — same code path.

**Parameters:** `layer`, `x_id`

### `prometheus.prom_layernorm_params`

```
fn prom_layernorm_params(layer)
```

**Parameters:** `layer`

### `prometheus.prom_crt_pe_matrix`

```
fn prom_crt_pe_matrix(seq_len, d_model)
```

**Parameters:** `seq_len`, `d_model`

### `prometheus.prom_sequential`

```
fn prom_sequential(layers)
```

**Parameters:** `layers`

### `prometheus.prom_sequential_forward`

```
fn prom_sequential_forward(model, x_id)
```

**Parameters:** `model`, `x_id`

### `prometheus.prom_relu_layer`

```
fn prom_relu_layer()
```

Activation pseudo-layers — let users put them inline in a Sequential.

### `prometheus.prom_sigmoid_layer`

```
fn prom_sigmoid_layer()
```

### `prometheus.prom_collect_params_v2`

```
fn prom_collect_params_v2(layers)
```

Collect params from all layers (extends to embedding + layernorm too).

**Parameters:** `layers`

---

## `requests` — `examples/lib/requests.omc`

### `requests.get`

```
fn get(url)
```

**Parameters:** `url`

### `requests.post`

```
fn post(url, data)
```

**Parameters:** `url`, `data`

### `requests.put`

```
fn put(url, data)
```

**Parameters:** `url`, `data`

### `requests.delete`

```
fn delete(url)
```

**Parameters:** `url`

### `requests.head`

```
fn head(url)
```

**Parameters:** `url`

### `requests.status`

```
fn status(resp)
```

**Parameters:** `resp`

### `requests.text`

```
fn text(resp)
```

**Parameters:** `resp`

### `requests.json`

```
fn json(resp)
```

**Parameters:** `resp`

### `requests.headers`

```
fn headers(resp)
```

**Parameters:** `resp`

### `requests.url_of`

```
fn url_of(resp)
```

**Parameters:** `resp`

### `requests.ok`

```
fn ok(resp)
```

**Parameters:** `resp`

### `requests.fetch_json`

```
fn fetch_json(url)
```

---- Convenience: one-line GET-and-parse-JSON ----------------------------
Returns the parsed JSON as an OMC dict/array, or null on non-200.

**Parameters:** `url`

### `requests.fetch_text`

```
fn fetch_text(url)
```

Returns response text or null.

**Parameters:** `url`

---

## `schema` — `examples/lib/schema.omc`

### `schema.schema_type_of`

```
fn schema_type_of(v)
```

**Parameters:** `v`

### `schema.validate`

```
fn validate(value, schema)
```

validate(value, schema) → {ok: bool, errors: [string]}

**Parameters:** `value`, `schema`

### `schema.coerce`

```
fn coerce(value, schema)
```

coerce(value, schema) → value (best-effort type coercion)

**Parameters:** `value`, `schema`

### `schema.schema_string`

```
fn schema_string(min_len, max_len)
```

**Parameters:** `min_len`, `max_len`

### `schema.schema_number`

```
fn schema_number(min_val, max_val)
```

**Parameters:** `min_val`, `max_val`

### `schema.schema_object`

```
fn schema_object(required_fields, properties)
```

**Parameters:** `required_fields`, `properties`

### `schema.schema_array`

```
fn schema_array(item_schema, min_items, max_items)
```

**Parameters:** `item_schema`, `min_items`, `max_items`

### `schema.schema_enum`

```
fn schema_enum(values)
```

**Parameters:** `values`

### `schema.assert_valid`

```
fn assert_valid(value, schema)
```

assert_valid(value, schema) — raises error if invalid

**Parameters:** `value`, `schema`

---

## `sklearn` — `examples/lib/sklearn.omc`

### `sklearn.kmeans`

```
fn kmeans(n_clusters)
```

**Parameters:** `n_clusters`

### `sklearn.linear_regression`

```
fn linear_regression()
```

### `sklearn.logistic_regression`

```
fn logistic_regression()
```

### `sklearn.random_forest_classifier`

```
fn random_forest_classifier(n_estimators)
```

**Parameters:** `n_estimators`

### `sklearn.random_forest_regressor`

```
fn random_forest_regressor(n_estimators)
```

**Parameters:** `n_estimators`

### `sklearn.fit`

```
fn fit(model, X, y)
```

**Parameters:** `model`, `X`, `y`

### `sklearn.predict`

```
fn predict(model, X)
```

**Parameters:** `model`, `X`

### `sklearn.score`

```
fn score(model, X, y)
```

**Parameters:** `model`, `X`, `y`

### `sklearn.train_test_split`

```
fn train_test_split(X, y, test_size)
```

---- Train/test split ----------------------------------------------------
Returns [X_train, X_test, y_train, y_test] as OMC arrays.
test_size is a kwarg in sklearn's API — passing it positionally
would make it a third array. py_call_fn_kw handles the split.

**Parameters:** `X`, `y`, `test_size`

### `sklearn.standard_scaler`

```
fn standard_scaler()
```

### `sklearn.fit_transform`

```
fn fit_transform(scaler, X)
```

**Parameters:** `scaler`, `X`

### `sklearn.transform`

```
fn transform(scaler, X)
```

**Parameters:** `scaler`, `X`

### `sklearn.accuracy_score`

```
fn accuracy_score(y_true, y_pred)
```

**Parameters:** `y_true`, `y_pred`

### `sklearn.r2_score`

```
fn r2_score(y_true, y_pred)
```

**Parameters:** `y_true`, `y_pred`

### `sklearn.confusion_matrix`

```
fn confusion_matrix(y_true, y_pred)
```

**Parameters:** `y_true`, `y_pred`

### `sklearn.load_iris`

```
fn load_iris()
```

### `sklearn.load_wine`

```
fn load_wine()
```

### `sklearn.load_breast_cancer`

```
fn load_breast_cancer()
```

---

## `sqlite` — `examples/lib/sqlite.omc`

### `sqlite.connect`

```
fn connect(path)
```

`path` can be ":memory:" for an in-memory database or a real file path.

**Parameters:** `path`

### `sqlite.close`

```
fn close(conn)
```

**Parameters:** `conn`

### `sqlite.commit`

```
fn commit(conn)
```

**Parameters:** `conn`

### `sqlite.rollback`

```
fn rollback(conn)
```

**Parameters:** `conn`

### `sqlite.execute`

```
fn execute(conn, sql)
```

Execute a SQL statement that doesn't return rows (DDL, INSERT/UPDATE/DELETE
without RETURNING). Returns null. Caller is responsible for commit.

**Parameters:** `conn`, `sql`

### `sqlite.execute_with`

```
fn execute_with(conn, sql, params)
```

Same but with bound parameters. Pass an OMC array; sqlite3 binds via "?".

**Parameters:** `conn`, `sql`, `params`

### `sqlite.execute_many`

```
fn execute_many(conn, sql, rows)
```

Execute many: one SQL with a list of parameter rows.

**Parameters:** `conn`, `sql`, `rows`

### `sqlite.query`

```
fn query(conn, sql)
```

Run SELECT, return all rows as an OMC array of arrays.

**Parameters:** `conn`, `sql`

### `sqlite.query_with`

```
fn query_with(conn, sql, params)
```

Same with bound parameters.

**Parameters:** `conn`, `sql`, `params`

### `sqlite.query_one`

```
fn query_one(conn, sql)
```

Single-row helper: returns the first row or null.

**Parameters:** `conn`, `sql`

### `sqlite.tables`

```
fn tables(conn)
```

Schema introspection — useful for "show me what's in this DB" workflows.

**Parameters:** `conn`

---

## `substrate` — `examples/lib/substrate.omc`

### `substrate.s_new`

```
fn s_new()
```

Build an empty substrate-friendly sorted-int container. Just an alias
for an empty array — used to signal intent to other readers.

### `substrate.s_add`

```
fn s_add(container, value)
```

Add a value into a sorted container, keeping it sorted. Returns the
insertion index.

**Parameters:** `container`, `value`

### `substrate.s_find`

```
fn s_find(container, value)
```

Substrate-routed exact lookup: returns the index or -1.

**Parameters:** `container`, `value`

### `substrate.s_count_range`

```
fn s_count_range(container, lo, hi)
```

Substrate-routed range count: how many elements in [lo, hi).

**Parameters:** `container`, `lo`, `hi`

### `substrate.s_slice_range`

```
fn s_slice_range(container, lo, hi)
```

Substrate-routed range slice: array of elements in [lo, hi).

**Parameters:** `container`, `lo`, `hi`

### `substrate.s_nearest`

```
fn s_nearest(container, value)
```

Substrate-routed nearest value lookup.

**Parameters:** `container`, `value`

### `substrate.s_median`

```
fn s_median(container)
```

Substrate-routed median (50th percentile).

**Parameters:** `container`

### `substrate.s_percentile`

```
fn s_percentile(container, p)
```

Substrate-routed percentile [0, 100].

**Parameters:** `container`, `p`

### `substrate.h_snap`

```
fn h_snap(value)
```

Snap a value to the nearest Fibonacci attractor.

**Parameters:** `value`

### `substrate.h_residual`

```
fn h_residual(value)
```

Residual after substrate alignment (non-attractor component).

**Parameters:** `value`

### `substrate.h_distance`

```
fn h_distance(a, b)
```

Substrate-distance: ln(|a-b|+1) / (pi * ln(phi)).

**Parameters:** `a`, `b`

### `substrate.h_hash`

```
fn h_hash(value)
```

Substrate-aligned hash for keying into dict/bloom structures.

**Parameters:** `value`

### `substrate.h_weight`

```
fn h_weight(n)
```

Zeckendorf weight: number of Fibonacci terms in n's decomposition.

**Parameters:** `n`

### `substrate.h_score`

```
fn h_score(arr)
```

Substrate-coherence score for an array (0.0 .. 1.0).

**Parameters:** `arr`

### `substrate.s_from_array`

```
fn s_from_array(arr)
```

Build a sorted set from an unsorted array via substrate operations.
Result is sorted, deduplicated, and ready for s_find / s_count_range.

**Parameters:** `arr`

---

## `test` — `examples/lib/test.omc`

### `test.assert_eq`

```
fn assert_eq(actual, expected, msg)
```

**Parameters:** `actual`, `expected`, `msg`

### `test.assert_ne`

```
fn assert_ne(actual, unexpected, msg)
```

**Parameters:** `actual`, `unexpected`, `msg`

### `test.assert_true`

```
fn assert_true(cond, msg)
```

**Parameters:** `cond`, `msg`

### `test.assert_false`

```
fn assert_false(cond, msg)
```

**Parameters:** `cond`, `msg`

### `test.assert_near`

```
fn assert_near(actual, expected, eps, msg)
```

Float equality within an epsilon tolerance. Default eps = 1e-9
(tight for typical FP work; pass a looser eps for ML/stats tests).

**Parameters:** `actual`, `expected`, `eps`, `msg`

### `test.assert_throws`

```
fn assert_throws(body_fn, msg)
```

Assertion: invoking nullary fn body_fn raises ANY error.
Use for "this should fail" tests without coupling to the exact
error message.

**Parameters:** `body_fn`, `msg`

### `test.assert_throws_with`

```
fn assert_throws_with(body_fn, expected_substr, msg)
```

Like assert_throws but also requires the error message to contain
a given substring. Use to assert a specific failure mode.

**Parameters:** `body_fn`, `expected_substr`, `msg`

### `test.assert_contains`

```
fn assert_contains(haystack, needle, msg)
```

Convenience: assert string contains substring.

**Parameters:** `haystack`, `needle`, `msg`

### `test.assert_len`

```
fn assert_len(arr, expected_len, msg)
```

Convenience: assert array length.

**Parameters:** `arr`, `expected_len`, `msg`

---

## `text` — `examples/lib/text.omc`

### `text.tokenize`

```
fn tokenize(text)
```

tokenize(text) → [tokens]
Simple whitespace + punctuation tokenizer; lowercases.

**Parameters:** `text`

### `text.term_freq`

```
fn term_freq(tokens)
```

term_freq(tokens) → {token: count}

**Parameters:** `tokens`

### `text.tfidf_corpus`

```
fn tfidf_corpus(docs)
```

tfidf_corpus(docs) → corpus object for scoring
docs: [string, ...]

**Parameters:** `docs`

### `text.tfidf_score`

```
fn tfidf_score(corpus, query)
```

tfidf_score(corpus, query) → [{doc_idx, score}, ...] sorted desc

**Parameters:** `corpus`, `query`

### `text.bm25_corpus`

```
fn bm25_corpus(docs, k1, b)
```

bm25_corpus(docs, k1?, b?) → corpus object

**Parameters:** `docs`, `k1`, `b`

### `text.bm25_score`

```
fn bm25_score(corpus, query)
```

bm25_score(corpus, query) → [{doc_idx, score}, ...] sorted desc

**Parameters:** `corpus`, `query`

### `text.levenshtein`

```
fn levenshtein(a, b)
```

levenshtein(a, b) → int

**Parameters:** `a`, `b`

### `text.text_similarity`

```
fn text_similarity(a, b)
```

similarity(a, b) → 0.0..1.0 (1 = identical)

**Parameters:** `a`, `b`

### `text.closest`

```
fn closest(query, candidates)
```

closest(query, candidates) → {match, score, idx}

**Parameters:** `query`, `candidates`

### `text.chunk_fixed`

```
fn chunk_fixed(text, size, overlap)
```

chunk_fixed(text, size, overlap?) → [string chunks]

**Parameters:** `text`, `size`, `overlap`

### `text.chunk_sentences`

```
fn chunk_sentences(text, max_per_chunk)
```

chunk_sentences(text, max_per_chunk?) → [string chunks]

**Parameters:** `text`, `max_per_chunk`

### `text.ngrams`

```
fn ngrams(tokens, n)
```

ngrams(tokens, n) → [[token, ...], ...]

**Parameters:** `tokens`, `n`

---

## `torch` — `examples/lib/torch.omc`

### `torch.tensor`

```
fn tensor(items)
```

**Parameters:** `items`

### `torch.zeros`

```
fn zeros(shape)
```

**Parameters:** `shape`

### `torch.ones`

```
fn ones(shape)
```

**Parameters:** `shape`

### `torch.randn`

```
fn randn(shape)
```

**Parameters:** `shape`

### `torch.add`

```
fn add(a, b)
```

**Parameters:** `a`, `b`

### `torch.sub`

```
fn sub(a, b)
```

**Parameters:** `a`, `b`

### `torch.mul`

```
fn mul(a, b)
```

**Parameters:** `a`, `b`

### `torch.matmul`

```
fn matmul(a, b)
```

**Parameters:** `a`, `b`

### `torch.t_sum`

```
fn t_sum(t)
```

**Parameters:** `t`

### `torch.t_mean`

```
fn t_mean(t)
```

**Parameters:** `t`

### `torch.tolist`

```
fn tolist(t)
```

**Parameters:** `t`

### `torch.item`

```
fn item(t)
```

**Parameters:** `t`

### `torch.t_shape`

```
fn t_shape(t)
```

**Parameters:** `t`

### `torch.linear`

```
fn linear(in_features, out_features)
```

**Parameters:** `in_features`, `out_features`

### `torch.forward`

```
fn forward(model, x)
```

**Parameters:** `model`, `x`

### `torch.sgd`

```
fn sgd(model, lr)
```

**Parameters:** `model`, `lr`

### `torch.adam`

```
fn adam(model, lr)
```

**Parameters:** `model`, `lr`

### `torch.step`

```
fn step(opt)
```

**Parameters:** `opt`

### `torch.zero_grad`

```
fn zero_grad(opt)
```

**Parameters:** `opt`

### `torch.mse_loss`

```
fn mse_loss(pred, target)
```

**Parameters:** `pred`, `target`

### `torch.backward`

```
fn backward(loss)
```

**Parameters:** `loss`

---

