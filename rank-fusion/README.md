# Exact Weighted Rank Fusion over Paginated Directory APIs

A research-data portal exposes per-directory listings that are only available
**sorted by score, high-to-low with cursor pagination**, plus **point lookups
by item id**. Each directory scores items on relevance, recency and quality.
This project fuses several such directories into one **exact** weighted
top-K ranking — no "take each source's top-K and merge" approximation, and
**no floating point anywhere on the scoring path**.

Everything is pure Python standard library (`fractions.Fraction`,
`asyncio`, `http.server`-style raw sockets). No third-party dependencies.

## What is delivered

```
rank-fusion/
├── src/
│   ├── rankfusion/          # fusion product: engine, client, service, CLI
│   │   ├── rational.py      # exact rational parsing; floats are rejected
│   │   ├── protocol.py      # response parsing + observable contract checks
│   │   ├── client.py        # bounded-async HTTP client with 429/503 retries
│   │   ├── engine.py        # the exact fusion algorithm + stop certificate
│   │   ├── request.py       # request validation
│   │   ├── service.py       # POST /fuse HTTP service
│   │   ├── cli.py           # single-run command line
│   │   ├── types.py         # result / boundary / stats data structures
│   │   └── errors.py
│   └── datasim/             # local HTTP directory simulator
│       ├── catalog.py       # static catalog, deterministic page plan, faults
│       └── server.py        # /page and /item endpoints with snapshot binding
├── tests/
│   ├── oracle.py            # INDEPENDENT oracle; imports no product code
│   └── test_fusion.py       # 21 unittest cases over real HTTP
├── drills/
│   ├── scenarios/*.json     # static catalogs + request templates
│   └── out/                 # saved results and full request traces
└── scripts/
    ├── run_drills.py        # reproducible end-to-end drill + oracle check
    └── run_module.py        # isolated (-I) module launcher
```

## The exact fusion rule

A request fixes a `catalog_snapshot`, a list of `source_id`s each with a
**positive rational weight**, and a non-negative integer `K`.

* Each source holds a finite candidate set of `(item_id, exact rational score
  in [0,1])`. Scores travel over the wire as strings like `"3/5"` or
  `"0.25"` and are parsed straight into `Fraction`. A JSON float is rejected.
* An item absent from a source scores **exactly 0** there.
* The candidate universe is the **union** of the source sets. Unknown all-zero
  ids are never invented.
* The total is the exact weighted sum `Σ_s w_s · score_s(i)`.
* Ranking is exact total **descending**, ties broken by **item_id Unicode
  code-point ascending** (plain Python string comparison).
* Return `min(K, universe size)` items. `K = 0` returns an empty list and
  performs **no pagination and no point lookups at all**.

## Why the early stop is an exact certificate

For every seen item the engine keeps a lower/upper range over its still
unknown components, using each source's last promised `upper_bound` (the
bound for all unread pages; it never increases, and a source not yet read is
conservatively bounded by the domain maximum `1`):

```
lower(i) = Σ_known_s  w_s · score_s(i)
upper(i) = lower(i) + Σ_unknown_s w_s · bound_s
H0       = Σ_non-exhausted_s w_s · bound_s      # best total any fully
                                                 # unseen item could reach
```

A top-K is only certified when **all** hold:

1. at least K items have **every** component confirmed by point lookup or
   exhaustion (so their totals are exact);
2. every unresolved *seen* item is strictly behind the K-th exact item under
   the full ordering key `(-total, item_id)` even for its most favorable
   possible completion;
3. every *unseen* item is strictly behind, i.e. `H0 < kth_total`.

**Equality never certifies.** If `H0 == kth_total`, an unseen item could tie
the score and win the id tie-break with a smaller id, so paging continues
until either the bound drops strictly below or the terminal cursor proves the
source is exhausted (which drives its bound to 0). The `tail_tie` drill
records explicit `stop_deferred` trace events for exactly this situation.

Only a **null `next_cursor`** proves exhaustion. An empty page may carry a
`next_cursor` and must be followed; it never terminates a source by itself.

## Observable contract enforcement and the trust boundary

The engine validates everything it can observe:

* every response is bound to the requested snapshot/source — a wrong snapshot
  is a hard failure (`snapshot_mismatch`), responses from different snapshots
  are never mixed into one ranking;
* intra-page ordering (score desc, id asc on ties);
* non-increasing bounds across pages, and that a page's first record never
  exceeds the bound promised by the previous page;
* cursor advancement rules (a non-empty page must advance);
* point-lookup score consistency with the score previously seen on a sorted
  page (`score_inconsistency`);
* retryable transport/status failures are retried with bounded backoff.

**Trust boundary (read this).** The engine can only police *observable*
violations. A malicious directory that *under-reports* an upper bound but is
never actually probed past the stopping point cannot be detected: if its
promised bound already certifies the answer, no further page is read, so the
lie is invisible. Such an undetectable low-ball is explicitly **outside** the
trust boundary and is not defended against. Any bound/ordering/inconsistency
that *is* observed, any snapshot mismatch, and any exhausted retry budget
aborts the run with `status: "incomplete"` and an empty `items` list —
partial candidates are never repackaged as a final top-K.

## Bounded asynchronous scheduling

* A single semaphore (`max_concurrency`) bounds requests **across all
  sources**; the scheduler does not pull every directory in lock-step.
* Each round issues at most one new sorted-access page per source, and only
  while that source's bound can still change the decision. Random-access
  point lookups (to confirm unknown components of already-seen items) are
  scheduled first.
* Retries are idempotent: a request's response is applied exactly once when
  it finally succeeds, so a score is never double-counted and point lookups
  are never attributed to the wrong candidate.
* Per-source accounting is retained: pages read, empty pages, point lookups,
  HTTP attempts and retries, exhaustion flag, current bound and its weighted
  form. The full request/response trace is saved on every run.

## Running it

All commands use `-I` (isolated mode) so a planted `json.py` next to a data
file cannot hijack imports; `scripts/run_module.py` puts `src/` on
`sys.path` without weakening that isolation.

### 1. Start the directory simulator

```bash
python3 -I scripts/run_module.py datasim.server \
    --catalog drills/scenarios/main.json --port 8123
```

The simulator supports, per source, configurable `page_size`,
`empty_page_slots`, and `faults` (latency, 429/503 status, deliberately bad
`upper_bound`, forced absent lookups).

### 2. Run one fusion via the CLI

```bash
sed 's#${BASE_URL}#http://127.0.0.1:8123#g' \
    drills/scenarios/main.request.json > /tmp/req.json
python3 -I scripts/run_module.py rankfusion.cli \
    --request /tmp/req.json \
    --out /tmp/result.json --trace-out /tmp/trace.json
```

### 3. Run the HTTP fusion service

```bash
python3 -I scripts/run_module.py rankfusion.service --port 8090
# POST a fusion request JSON to http://127.0.0.1:8090/fuse
```

### 4. Reproducible drill suite (starts its own simulators)

```bash
python3 -I scripts/run_drills.py
```

It runs every scenario over real HTTP, recomputes each expected answer with
the independent oracle, and writes `drills/out/*.result.json` plus
`*.trace.json` and a `summary.json`. Exit code is non-zero on any mismatch.

### 5. Tests

```bash
python3 -I -m unittest discover -s tests -v
```

## The headline example

Two equal-weight sources: `a=(1,0)`, `b=(0,1)`, `c=(3/5,3/5)`, K=1.
Each source's own leader is `a` or `b` with total 1, but the exact fused
winner is **`c` with total `6/5`**. The drill certifies this while **unread
pages still remain in both sources** (`union_fully_enumerated: false`,
`unseen_potential: 1/5 < 6/5`).

## Scenario coverage

| Scenario | What it proves |
|---|---|
| `main` | fused winner `c=6/5`, certified with unread pages in every source |
| `tail_tie` | `H0 == kth_total` with a smaller-id tail item present; engine defers stop and pages on |
| `single_source` | candidate in only one source; mirror source empty |
| `all_empty` | every source empty → empty ranking, terminal cursors only |
| `paging` | consecutive empty pages, different page sizes, explicit absent (zero) lookups |
| `near_tie` | two totals float64 conflates but exact arithmetic keeps apart (proven in `float_conflation_proof.json`) |
| `retry_429` | injected latency, 429 and 503 retries, no double counting, out-of-order responses |
| `bad_bound` | observable bound violation → `incomplete`, no ranking returned |
| `kzero` | K=0 returns empty and reads no pages / does no lookups |

Additional fault cases (persistent 503 exhausting attempts, wrong snapshot,
record overshooting a promised bound, point-vs-page score disagreement) live
in `tests/test_fusion.py`.

## API sketch

`POST /fuse` body:

```json
{
  "catalog_snapshot": "snap-main-2026-11-16",
  "k": 1,
  "weights": {"src-a": "1", "src-b": "1"},
  "sources": [
    {"source_id": "src-a", "base_url": "http://127.0.0.1:8123"},
    {"source_id": "src-b", "base_url": "http://127.0.0.1:8123"}
  ],
  "max_concurrency": 4,
  "max_attempts": 5
}
```

All rationals are strings; floats are rejected. The response carries the exact
ranking with per-source components, per-source request statistics, and a
`stop_boundary` explaining precisely why termination was sound.
