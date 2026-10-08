# Segcore Grep: ripgrep Integration and a Memory-Only C API

- **Created:** 2026-10-08
- **Status:** Draft; dependency and integration approach selected, implementation pending
- **Components:** Segcore, Rust binding, query execution
- **Repository baseline:** `6e6b5c462dedecb55e2f95f7604e9882aece1924`
- **Upstream baseline:** ripgrep `15.2.0`, commit `e89fff89ac9af12e8d4ce9d5fd07beb408ca730f`

## Summary

Add an in-process Grep component that searches text already loaded by Milvus.
It compiles patterns once, searches independent field values in batches, and
returns record selection, matching lines with context, or matching fragments.

The implementation uses a new `milvus-grep-core` Rust crate. The existing
`tantivy-binding` crate consumes it through a local `path` dependency and
exports a small C API. C++ calls that API through a Segcore adapter. The final
Rust artifact remains `libtantivy_binding.a`, linked into `milvus_core`.

Version one uses the complete upstream `grep` facade and its default
dependencies from ripgrep 15.2.0. It does not trim upstream source code or
reorganize the Rust build into a Cargo workspace. The C boundary exposes only
the memory-search functionality needed by Milvus. Source-level trimming can be
evaluated after correctness, integration, and performance measurements.

This document specifies the internal search component and its integration
contract. It does not claim that the public Grep operation, protocol fields,
Rust exports, or C++ adapter already exist.

## 1. Motivation and Current Behavior

Milvus already has RE2-based string matching in
[`RegexQuery.h`](../../../internal/core/src/common/RegexQuery.h). Its full and
partial matchers primarily return Boolean results and enable `dot_nl`.
That behavior is useful for existing filters, but is not the same contract as
line-oriented Grep with line numbers, context, matching fragments, and an
explicit multiline mode. Existing filter semantics must remain unchanged.

The repository also has an established Rust-to-C++ integration:

- [`tantivy-binding/Cargo.toml`](../../../internal/core/thirdparty/tantivy/tantivy-binding/Cargo.toml)
  declares Rust dependencies and produces `staticlib` and `rlib` artifacts.
- [`tantivy/CMakeLists.txt`](../../../internal/core/thirdparty/tantivy/CMakeLists.txt)
  invokes `cargo +1.89`, imports the resulting static library, and installs its
  headers.
- [`build.rs`](../../../internal/core/thirdparty/tantivy/tantivy-binding/build.rs)
  runs cbindgen. Its current
  [configuration](../../../internal/core/thirdparty/tantivy/tantivy-binding/cbindgen.toml)
  generates a C++ header.
- [`src/CMakeLists.txt`](../../../internal/core/src/CMakeLists.txt) links
  `tantivy_binding` into `milvus_core`. The
  [Segcore target](../../../internal/core/src/segcore/CMakeLists.txt) is an
  object library, not the final shared-library link target.

This design reuses that build path while keeping Grep logic separate from
Tantivy index objects.

## 2. Goals and Scope

### Goals

1. Search a batch of in-memory UTF-8 field values without opening files,
   issuing network requests, or executing external commands.
2. Support regex and literal patterns, multiple patterns combined with OR,
   case control, whole-line matching, line-oriented and multiline search,
   LF/CRLF handling, context, and per-field `-m` semantics.
3. Return structured selection, line, and occurrence data with original-field
   byte coordinates.
4. Reuse compiled patterns and task-local execution buffers.
5. Define ownership, cancellation, error propagation, and resource limits
   before implementation.
6. Keep the initial C API limited to implemented Milvus functionality rather
   than mirror ripgrep's CLI or internal configuration surface.

### Outside Version One

The C header has no file, path, file-descriptor, Reader/Writer callback,
encoding, BOM, mmap, thread-count, color, JSON, CLI-argument, engine-selection,
PCRE2, Unicode-toggle, replacement, whole-word, stop-on-nonmatch, inverted
selection, count, or request-level `exists` option. It has no generic extension
dictionary or reserved feature flags.

PCRE2, counts, inverted selection, NGRAM planning, soft detail truncation, and
request-level existence results require separate implementations and contracts
before being added. Version one does not provide a resumable scan or a stable
cross-version plugin ABI.

The component does not own Collection/Partition selection, authorization,
visibility, primary keys, field names, database `limit`/`offset`, result-column
projection, storage, or task scheduling. Full GNU grep, POSIX, and ripgrep CLI
compatibility are not acceptance criteria; the supported subset is explicit.

## 3. Selected Dependency and Build Integration

### 3.1 Version and License

Use the **15.2.0 release commit**, not upstream `master`. The release declares
Rust 1.85 as its minimum version and is compatible with the selected Milvus
Rust 1.89 toolchain at the independent-build level.

| Crate | Version in the selected source commit | Use |
|---|---|---|
| `grep` | 0.4.1 | Facade used by `milvus-grep-core` |
| `grep-matcher` | 0.1.9 | Matching interface and occurrence enumeration |
| `grep-regex` | 0.1.14 | Pattern compilation and matching |
| `grep-searcher` | 0.1.17 | In-memory search and line/context events |
| `grep-cli` | 0.1.12 | Retained default dependency |
| `grep-printer` | 0.3.1 | Retained default dependency |
| `grep-pcre2` | 0.1.10 | Optional; not enabled in version one |

These are the package versions in the release source, not the minimum version
ranges appearing in individual dependency declarations. Pinning only the
crates.io facade version would not pin all of its transitive dependencies.

ripgrep is dual-licensed `Unlicense OR MIT`. Select MIT, retain its copyright
and license text, and review notices for the final resolved dependency set.
This is not a completed audit of every dependency in the Milvus product.

### 3.2 Proposed Layout

```text
internal/core/thirdparty/grep/milvus-grep-core/
  Cargo.toml
  src/lib.rs
  src/plan.rs
  src/session.rs
  src/sink.rs
  src/result.rs
  src/error.rs
  tests/

internal/core/thirdparty/tantivy/tantivy-binding/
  Cargo.toml                 # Add local path dependency.
  Cargo.lock                 # Resolve and lock the integrated dependency graph.
  build.rs                   # Generate the narrowly scoped C-compatible header.
  src/lib.rs                 # Register grep_c.
  src/grep_c.rs              # C ABI types, handles, validation, panic boundary.
  include/milvus_grep.h      # Generated during implementation.

internal/core/src/segcore/
  GrepRunner.h/.cpp          # Selection, field lifetime, batching, row mapping.
  GrepRustBridge.h/.cpp      # RAII and structured result/error conversion.
```

`milvus-grep-core` is an `rlib`. Add this dependency to `tantivy-binding`:

```toml
milvus-grep-core = { path = "../../grep/milvus-grep-core" }
```

In `milvus-grep-core`, use the pinned upstream source:

```toml
[dependencies]
grep = { git = "https://github.com/BurntSushi/ripgrep.git", rev = "e89fff89ac9af12e8d4ce9d5fd07beb408ca730f" }
```

Use `grep::matcher`, `grep::regex`, and `grep::searcher`. Keep the default
dependency set; do not use `--all-features` to imply completeness. Merely
including the facade does not launch the CLI or expose all its behaviors.

### 3.3 Build Changes

Keep the existing Rust artifact name, toolchain, and release profile, including
`panic = "unwind"`. Cargo builds Grep as a dependency of the binding; there is
no separate Grep static library or independent Cargo invocation from CMake.

Merge new dependencies into the existing binding lockfile. Do not replace it
with ripgrep's lockfile or perform an unrelated full dependency update. Submit
the reviewed lockfile and use `--locked` in the integrated build.

Generate a separate **C-compatible** `milvus_grep.h`; the existing C++-oriented
binding header should not expose Grep's internal Rust types. Configure exports
and common includes so the two headers can coexist without duplicate type
definitions. Add header installation and generation dependencies to CMake,
including dependencies needed before consumers compile in a clean parallel
build. The final link remains the existing `milvus_core` link.

Update build-context packaging, CI cache inputs, and source-change tracking to
include the sibling Grep directory. Changes in that directory must rebuild the
Rust artifact. No workspace root, lockfile relocation, or binding rename is
part of this work.

## 4. Architecture and Query Integration

```text
Query validation / typed Grep specification
                    |
      Segcore snapshot + visibility + Filter
                    |
       GrepRunner: retain candidate row mapping
                    |
       bounded batches of non-NULL field values
                    |
       GrepRustBridge -> eight-function C API
                    |
       milvus-grep-core
         Plan: compiled matcher and immutable configuration
         Session: execution buffers and cancellation state
         Sink: structured in-memory results
                    |
       positional results -> original segment rows
                    |
       selected records -> reduction -> offset/limit
```

The intended public direction is a Grep operation in the Query execution path.
Public protocol, SDK syntax, computed-result columns, and any new query-stage
enum remain integration work; this document does not assert that an
`L0_QUERY` stage already exists. An unsupported operation must fail explicitly
rather than run an ordinary query while ignoring Grep.

The current
[`FilterBitsNode`](../../../internal/core/src/exec/operator/FilterBitsNode.cpp)
produces an exclusion bitset: `1` means filtered out, and NULL predicates are
excluded. The adapter must honor this polarity and the established visibility
rules. A Grep `selected` value is not directly a bit in that exclusion mask.

[`ExecPlanNodeVisitor`](../../../internal/core/src/query/ExecPlanNodeVisitor.cpp)
captures the read snapshot and, for bitmap retrieve results, applies
`find_first_n(node.limit_, ...)`. Grep selection must occur before this result
truncation. Applying the record limit first and then searching those records
would miss valid matches. Detailed results must remain aligned with their
selected rows through later reduction and projection.

C++ pins field storage through the synchronous call, removes database NULLs,
and stores `batch position -> segment row` mappings outside Rust. It owns
request deadlines, remaining budgets, and scheduling. Rust receives neither a
Segment pointer nor a database identifier and does not create an inner thread
pool.

Version one can scan the full eligible Filter/visibility subset. NGRAM is a
later optimization: a candidate filter must be a proven necessary condition
under the same matching semantics. An unproven optimization falls back to the
full eligible set. No `prefilter_view`, index pointer, or `MatchKnowledge`
parameter is reserved in the C API now.

## 5. C API

Appendix A contains the complete proposed header. It has three opaque handle
types, typed POD options/results, and eight functions:

| Function | Responsibility |
|---|---|
| `mg_grep_compile` | Compile patterns and copy configuration into a Plan |
| `mg_grep_plan_free` | Release the caller's Plan reference |
| `mg_grep_session_create` | Create task-local mutable execution state |
| `mg_grep_session_cancel` | Request sticky cancellation of a Session |
| `mg_grep_session_free` | Release a quiescent Session |
| `mg_grep_execute_batch` | Search a batch synchronously |
| `mg_grep_result_view` | Borrow a read-only view of a Result |
| `mg_grep_result_free` | Release the Result and all owned data |

### 5.1 Matching Options

| Field | Values / meaning |
|---|---|
| `pattern_kind` | Regex or literal |
| `case_mode` | Sensitive, insensitive, or smart-case |
| `scope` | Line-oriented or multiline within one field |
| `newline` | LF or CRLF handling |
| `whole_line` | Require whole-line matching |
| `dot_all` | Let dot match newline; valid only for regex + multiline |

The zero-initialized matching configuration is case-sensitive regex search,
line-oriented LF, with `whole_line` and `dot_all` disabled. Boolean fields accept
only 0/1; unknown values are rejected. Do not silently ignore invalid option
combinations.

Unicode matching is enabled by policy, without a public toggle. Version one
rejects inline Unicode-disabling constructs through syntax-aware validation;
it must not return arbitrary mid-codepoint fragments under a UTF-8 contract.
Other accepted pattern syntax and flag precedence follow the pinned matcher.

Matcher `multi_line(true)` controls line anchors; Searcher multiline mode
controls whether a match may cross lines. The adapter must reproduce the
intended combination, not equate the two setters. CRLF handling preserves raw
bytes and also accepts LF line endings. Fields are never concatenated into a
single search input.

### 5.2 Input Contract

`MgGrepBytes` contains a pointer and a length, not a NUL-terminated string.
Patterns and texts must be valid UTF-8; embedded NUL bytes in text are valid.
BOM bytes are preserved. The C layer does not split, unescape, or execute CLI
argument strings.

The pattern array must have at least one entry. Empty patterns are valid;
multiple patterns have OR semantics. Compilation occurs before checking whether
there are any eligible database records, so an empty candidate set does not
hide a malformed request pattern.

Database NULL is handled by C++. `{NULL, 0}` is an empty non-NULL text; a
nonzero length paired with NULL is invalid. An empty batch accepts a NULL array
pointer and returns an empty Result. Successful results always have exactly
one entry per input, including unselected inputs, in the same order.

Validate descriptor counts, arithmetic, aggregate input size, and necessary
result metadata before allocating for a batch. Validate all text UTF-8 before
searching, including SELECT and `-m 0`, so early exits cannot hide malformed
batch inputs. This validation consumes the execution time budget.

As with any in-process C ABI, non-NULL pointer validity and the claimed memory
extent remain caller obligations. The implementation can reject NULL/length
mismatches and overflow, but cannot safely probe arbitrary invalid addresses.

### 5.3 Output Modes

| Mode | Result | Context |
|---|---|---|
| `LINES` | Selection, scan completion, selected/context lines, occurrences | Before/after context allowed |
| `MATCHES` | Selection, scan completion, nonempty occurrences and exact text | Both context values must be zero |
| `SELECT` | Selection and scan completion only | Both context values must be zero |

`max_selected_units` is the per-field `-m` limit, not a database record limit or
the number of occurrences. Zero means no search; `UINT64_MAX` means no `-m`
limit. Use the pinned Searcher's `max_matches` behavior: line-oriented search
counts selected lines; a multiline match spanning lines is counted once
according to its search events. Multiple occurrences within one selected line
are not separately charged as selected lines. Requested trailing context is
still collected after the limit.

The following result invariants are part of the contract:

- `selected` is the only record-retention decision. It cannot be derived from
  `match_count`: a zero-length match can select a line while MATCHES returns no
  fragments.
- Empty text has no logical lines and is unselected in this API. A text
  consisting of a newline has an empty-content line that an empty pattern can
  select. Explicitly enforce and test this policy.
- `scan_complete` describes complete line-selection examination, not merely
  whether `selected` is known. SELECT may stop after finding a match; `-m` may
  stop before the end. Such success may have `scan_complete=0`. `-m 0` returns
  `selected=0`, `scan_complete=0`, and empty arrays.
- LINES returns unique lines in original line-number order, with SELECTED
  taking precedence over CONTEXT. Line text includes its original terminator.
  MATCHES returns no lines; SELECT returns neither array.
- Occurrences use the engine's non-overlapping enumeration order over the
  combined patterns, not separate per-pattern scans with duplicate results.
  Only selected line/group occurrences are business matches; context alone
  does not add matches.
- Byte coordinates are zero-based, half-open intervals in the original field.
  Line numbers are one-based. For nonempty matches, `end_line` is the line of
  the final consumed byte, with a terminator belonging to that line. An empty
  match belongs to its selected logical line; do not invent a line after EOF.
- MATCHES omits empty occurrences. LINES may retain empty ranges for selected
  lines. Each returned match text is the exact original byte interval.
- Enumerate with the original search boundary and offsets. Re-running a regex
  on an arbitrary extracted fragment can change anchors and other semantics.

C++ can construct business-level snippets from adjacent returned lines and
coordinates. The C view does not duplicate snippets, field metadata, request
mode, or counts that are already represented by array lengths.

## 6. Ownership and Concurrency

Patterns and configuration are copied or owned by the Plan before compilation
returns. A Session retains a strong reference to its Plan, so freeing the
caller's Plan handle does not invalidate existing Sessions. Sharing a Plan
must be backed by verified Matcher concurrency/cache behavior; no mutable
Searcher or Sink is shared between concurrently executing tasks.

One Session may execute multiple batches sequentially. It cannot execute two
batches concurrently. Cancellation is the only operation allowed concurrently
with execution on the same Session. Different Sessions may run in parallel
under Milvus scheduling.

Input text is borrowed only until `execute_batch` returns. A Result owns its
arrays and text storage. All nested view pointers remain stable until
`result_free`, independent of the input, Session, and Plan lifetimes. Empty
arrays have NULL pointers. Read-only Result views may be shared, provided
release is synchronized with all readers.

Only Rust release functions free Rust objects. C++ wraps handles in RAII;
it never calls `free`/`delete` on returned storage. NULL release is a no-op.
Plan release must not race with Session creation using that handle; Session
release waits for execution and cancellation callbacks; Result release waits
for all view users.

Example lifecycle:

```text
compile -> plan
session_create(plan) -> session
plan_free(plan)                         # Session retains ownership.
execute_batch(session, texts, ...) -> result
session_free(session)                   # No execution/cancel call remains.
result_view(result) -> copy/convert     # Input/session may already be gone.
result_free(result)
```

The installed header and library are delivered together for the same build and
architecture. Use C POD, fixed-width integers, `size_t`, and pointers; no Rust
`String`/`Vec`, C++ containers, packed structures, or cross-language `bool`.
Rust uses `repr(C)` structures and integer aliases. Do not add unused
`abi_version`, `struct_size`, or reserved fields for hypothetical consumers.
An incompatible future ABI change requires an explicit versioning decision.

## 7. Budgets, Cancellation, and Failure Semantics

### 7.1 Resource Limits

| Limit | Meaning |
|---|---|
| `max_patterns` | Pattern-count cap, checked before dereferencing the pattern array |
| `max_pattern_bytes` | Sum of pattern byte lengths |
| `max_regex_bytes` | Engine compiled-program limit |
| `max_dfa_cache_bytes` | Engine DFA-cache limit |
| `max_input_bytes` | Sum of batch text byte lengths |
| `max_result_bytes` | Logical result structures/arrays and owned text, including unselected-row metadata |
| `timeout_ms` | Remaining time allowance for this batch, starting with input validation |

All limits are strictly positive. They are supplied by the service, not raw CLI
switches. Input/output arithmetic is checked for overflow; zero-length texts
still consume descriptor/result metadata. Test the metadata-only exhaustion
case rather than checking only text bytes.

These are not aggregate RSS guarantees. Engine cache limits may apply to
multiple caches; logical result bytes exclude allocator overhead and temporary
engine storage. C++ also bounds batch size, concurrent Sessions, and the sum of
retained results, and passes remaining request budgets on each call.

Version one has **whole-batch success or failure**. Resource exhaustion,
invalid input, timeout, and cancellation discard that call's output and return
`*out_result=NULL`. No details are silently truncated to fit a budget. Success
returns complete detail/context for the requested scope, including explicit
`-m`; it need not mean the entire field was searched. If a later batch fails,
the caller fails the request rather than publishing earlier batches as a
complete success. Soft detail truncation is deferred.

### 7.2 Cancellation and Timeout

`session_cancel` sets Rust-owned atomic state. It is idempotent and sticky:
cancellation before execution cannot be lost, and future executions on that
Session also return CANCELLED. There is no reset operation; use a new Session
for a new task. C++ does not share an atomic memory layout with Rust.

Check cancellation/time at validation loops, text boundaries, Matcher call
boundaries, and Sink events. The final cancellation check before publishing a
result is the success linearization point; later cancellation does not revoke
an already completed result. A cancellation callback must retain the Session
until it returns. Freeing a handle is not cancellation.

Cancellation and deadlines are cooperative. They cannot immediately interrupt
every individual regex call. Pattern compilation has no Session cancellation
handle; bound input and engine resources and check the request deadline before
and after compilation. Hard preemptive timeout is not promised by this ABI.

### 7.3 Errors and Milvus Classification

The C API returns a `uint32_t` status and optionally fills a caller-owned
512-byte UTF-8 diagnostic buffer. Success clears the buffer; failure may
truncate at a character boundary and always terminates it with NUL. Control
flow uses the status code, never error-string parsing. No error allocator,
`error_free`, global `last_error`, or I/O error category is needed.

The eight values in Appendix A are **private Grep statuses**, not Milvus
`ErrorCode` values or wire codes. The C++ adapter must classify by the origin
of a failure before constructing a typed Segcore error:

| Origin | Required treatment |
|---|---|
| Malformed user pattern or unsupported user pattern syntax | Request-input failure |
| Invalid user pattern UTF-8 | Request-input failure |
| Invalid UTF-8 in a persisted field | Data-integrity/system failure; do not blame the current request |
| Invalid descriptor, enum, output pointer, or internally supplied configuration | Internal adapter/plan contract failure unless proven to originate from request input |
| Deterministic pattern/result policy limit | Preserve the budget failure and its policy origin; do not blindly map to a transient resource shortage |
| Request cancellation or deadline | Preserve cancellation/deadline semantics; do not translate to success or empty results |
| Recovered Rust panic | Internal failure, never invalid input |

Use the repository's
[error handling guide](../../dev/error_handling_guide.md) and
[casebook](../../dev/error_handling_casebook.md). Do not invent numeric 20xx
codes or cast a Grep status into `CStatus.error_code`. Audit producers and the
actual consumer path before choosing the existing Milvus code mapping.

Existing typed boundary patterns are visible in
[`tantivy-error.h`](../../../internal/core/thirdparty/tantivy/tantivy-error.h),
[`CGoCatch.h`](../../../internal/core/src/common/CGoCatch.h), and
[`merr/segcore.go`](../../../pkg/util/merr/segcore.go). The future Grep adapter
must preserve its chosen classification through executor wrapping and the
cgo boundary; implementation and end-to-end error mapping remain unverified.

Every failed handle-producing function sets a valid output slot to NULL. A
recoverable ordinary error clears per-call state. An internally panicked
Session is poisoned and returns INTERNAL_ERROR on later executions until it
is freed. Unwinding panics are caught at the ABI boundary; C++ exceptions do
not cross it. Abort and allocation failure cannot all be claimed recoverable.

## 8. Memory-Only Execution and Upstream Dependencies

Use `search_slice` with explicit encoding/BOM processing disabled, binary
detection disabled, and file mapping disabled. Keep the builder private;
neither clients nor C callers can supply I/O configuration. The Sink stores
structured results in memory and does not print or write files.

The pinned searcher selects an in-memory line or multiline path. Even its
transcoding fallback passes the original slice as a Reader, rather than
interpreting text as a file path. Disabling that fallback preserves original
bytes and avoids unnecessary decoding; the `Read` abstraction itself does not
imply disk I/O.

The facade retains I/O-related source and dependencies. `encoding_rs`,
`encoding_rs_io`, and `memmap2` are non-optional Searcher dependencies in this
version; a downstream `default-features=false` does not remove them. Cargo
feature unification can also re-enable shared capabilities through other
consumers. Do not promise that all unused code disappears from the final
shared library or use linker elimination as a security boundary.

Constructors still have memory costs: the pinned Searcher allocates an 8 KiB
decode buffer and a default 64 KiB line buffer. Reuse Session state rather than
constructing a Searcher for every record. Do not set `heap_limit=0` as a way to
disable buffers; it has different semantics.

The library uses Rust `log`; trace/debug messages may contain patterns or
extracted literals. Integrate with the existing process logger, suppress these
module-specific payload logs, and do not register a second global logger or
disable all Milvus logging. At the Go boundary, follow the repository's
[`mlog` guidance](../../agent_guides/observability/logging.md) and carry the real
request context. No new logging callbacks or telemetry knobs belong in this C
API. Metrics use low-cardinality labels rather than patterns or record IDs.

The contract forbids active external I/O by the matcher. It does not promise
that the entire query avoids storage reads: Milvus loads fields, mapped input
can fault pages, and allocators may use anonymous mmap. Verify the isolated
search path separately from the whole query's I/O behavior.

## 9. Compatibility and Alternatives

### Compatibility

- Existing RE2 filters and their semantics remain unchanged.
- This component does not change persisted data, indexes, WAL, manifests, or
  storage paths. It introduces no data migration.
- Existing requests keep their behavior. Public Grep exposure requires explicit
  plan/protocol support and mixed-version capability checks; unsupported nodes
  must reject the operation rather than ignore it.
- NULL rows are excluded by C++; empty fields and zero-length matches have the
  explicit rules above. One stored chunk is one independent document.
- MIT and transitive notices accompany distribution. Full dependency-license
  and security review is based on the integrated lockfile, not the facade alone.

### Alternatives Considered

| Alternative | Decision |
|---|---|
| Launch `rg` as a subprocess | Adds process/file/output-protocol work to an in-memory operation; not selected |
| Extend the existing RE2 Boolean matcher | Requires implementing the Grep line/context contract separately; retain RE2 for existing filters |
| Depend directly on only three search crates | Viable later reduction; first establish the complete facade baseline |
| Fork and remove I/O source immediately | Creates maintenance work before measured benefit; defer |
| Build a second Rust `staticlib` | Requires extra runtime/dependency/symbol coordination; use one binding artifact |
| Create a Cargo workspace now | Useful for shared independent testing/lockfile management, but adds unrelated build migration; use a path dependency first |
| Return partial details or callbacks across C | Expands lifetime, failure, and continuation contracts; use owned all-or-error results in version one |

The path-dependency build already resolves Grep under the binding lockfile and
root profile. Workspace adoption is not needed to produce one integrated
artifact. Independent crate tests may resolve a different graph, so binding
integration tests are mandatory even when the crate's own tests pass.

## 10. Implementation Plan

1. Add `milvus-grep-core` and the pinned facade dependency. Merge the binding
   lockfile and prove Rust 1.89 clean builds with the retained default features.
2. Implement pattern validation, immutable plans, and batched selection. Then
   implement LINES/MATCHES, context, multiline/CRLF, and `-m` combinations.
3. Implement the eight C exports, owned Result views, panic/error boundaries,
   cancellation, budgets, and the generated C header. Add ABI layout checks.
4. Add C++ RAII, field-lifetime and row-mapping adapters. Verify the final
   `milvus_core` link alongside existing Tantivy functionality.
5. Wire query execution before result truncation and add the public operation's
   explicit capability validation and structured result plumbing. Expose only
   option combinations with completed semantic and resource tests.
6. Measure production-target overhead and throughput. Consider dependency or
   source trimming only after comparing the integrated artifacts and workload.

No Rust implementation, generated production header, Cargo lockfile, CMake
change, or query behavior is delivered by this design-document change.

## 11. Verification Plan

### Evidence Already Collected

| Check | Result and boundary |
|---|---|
| ripgrep 15.2.0 facade build | `cargo +1.89 build --locked -p grep` succeeded on macOS ARM64, using the release source and its lockfile, default features, dev profile |
| Proposed C declarations and usage example | Strict C11 and C++17 syntax checks passed; no Rust ABI or runtime link was exercised |
| Earlier input-behavior probe | Three BOM/NUL tests passed on 14.1.1; historical research only, not 15.2.0 acceptance evidence |

The independent upstream build does not prove compatibility with Milvus's
merged lockfile, Release/LTO, Linux linking, or query semantics.

### Required Implementation Tests

| Area | Required cases |
|---|---|
| Pattern semantics | Regex/literal, multiple-pattern OR, case modes, whole-line, invalid pattern, forbidden Unicode disabling, empty pattern |
| Text boundaries | UTF-8, invalid UTF-8, NUL, BOM, empty field, blank line, final unterminated line, LF/CRLF, multiline restricted to one field |
| Results | Three mode shapes, zero-length selection with no MATCHES payload, exact offsets/text, multiple occurrences per line, overlapping context windows, context/selected deduplication |
| Limits | `-m 0/1/unlimited`, multiline groups, trailing context, pattern count/bytes, compile/cache limits, metadata-only batches, input/result byte limits, checked arithmetic |
| Lifecycle | Release Plan before Session, release inputs/Session before reading Result, independent Sessions, synchronized cancellation/free, sticky pre-call and between-batch cancellation |
| Failure | One invalid input fails the batch; no partial result after limit/deadline/cancel; panic poisons Session; no exception crosses ABI |
| Query integration | Filter exclusion polarity, snapshot and NULL handling, selection before limit, result/row alignment through reduction, unsupported-version rejection |
| Error propagation | Trace each producer through Rust status, C++ typed exception, executor, cgo, and Go consumer; confirm input/system classification and retry behavior |
| Build | Header coexistence, C/Rust layout, clean parallel header generation, final symbols, default feature graph, Linux targets, Release/LTO, existing Tantivy regression |

Use a standalone search probe to trace file/network/process activity after
initialization, with relevant detailed logs suppressed. Separate loader,
allocator, and input page-fault effects from active external operations.

Benchmark selection and detail modes across short/long records, low/high match
density, multiple batch sizes, multiline inputs, and concurrent Sessions.
Measure pattern compilation separately from steady-state matching. Compare
final `libmilvus_core` size, build time, peak memory, result-copy cost, and
throughput against the same toolchain/profile baseline. Do not infer runtime
improvement from a smaller dependency count.

The repository's adversarial review and failure-path verification apply before
shipping behavior. Syntax checks and a successful upstream build are not
substitutes for these tests.

## 12. Remaining Implementation Decisions

- Service defaults and hard ceilings for batch size, compilation/cache budgets,
  output bytes, and concurrent Sessions must be established from measurements.
- Select existing Milvus error mappings after auditing producers and consumers;
  a malformed internal descriptor must not become a user parameter error.
- Complete public Query plan/result/capability plumbing and verify its exact
  insertion point before limit, rather than claim it follows automatically
  from the C adapter.
- Validate cancellation latency on long individual matcher calls. If cooperative
  checking is insufficient for the service objective, address that explicitly.

These do not reopen the selected upstream version, path-dependency integration,
single Rust static artifact, or memory-only C boundary.

## References

- [ripgrep 15.2.0 release manifest](https://github.com/BurntSushi/ripgrep/blob/15.2.0/Cargo.toml)
- [Facade and default dependencies](https://github.com/BurntSushi/ripgrep/blob/15.2.0/crates/grep/Cargo.toml)
- [Searcher implementation](https://github.com/BurntSushi/ripgrep/blob/15.2.0/crates/searcher/src/searcher/mod.rs)
- [CLI matcher/searcher configuration](https://github.com/BurntSushi/ripgrep/blob/15.2.0/crates/core/flags/hiargs.rs)
- [Dual-license selection](https://github.com/BurntSushi/ripgrep/blob/15.2.0/COPYING)
- [MIT license](https://github.com/BurntSushi/ripgrep/blob/15.2.0/LICENSE-MIT)
- [Cargo dependency and path rules](https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html)
- [Rust foreign-code linking](https://doc.rust-lang.org/reference/linkage.html#mixed-rust-and-foreign-codebases)

## Appendix A. Proposed C Header

The following declarations are the reviewable contract, not generated
implementation output. Integer constants and structure layouts must match the
eventual Rust definitions. The header is intentionally independent of Tantivy
types and database metadata.

```c
/*
 * Milvus Grep C ABI v1 -- design contract, not an implementation.
 * Backend: ripgrep 15.2.0, default Rust regex engine.
 * See the accompanying Segcore Grep design document.
 * All objects belong to the same build/architecture. This is not a wire format.
 */
#ifndef MILVUS_GREP_H
#define MILVUS_GREP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MgGrepPlan MgGrepPlan;
typedef struct MgGrepSession MgGrepSession;
typedef struct MgGrepResult MgGrepResult;

typedef uint32_t MgGrepStatus;
#define MG_GREP_OK               UINT32_C(0)
#define MG_GREP_INVALID_ARGUMENT UINT32_C(1)
#define MG_GREP_INVALID_PATTERN  UINT32_C(2)
#define MG_GREP_INVALID_UTF8     UINT32_C(3)
#define MG_GREP_RESOURCE_LIMIT   UINT32_C(4)
#define MG_GREP_CANCELLED        UINT32_C(5)
#define MG_GREP_DEADLINE_EXCEEDED UINT32_C(6)
#define MG_GREP_INTERNAL_ERROR   UINT32_C(7)

/* Caller-owned, optional error output. Always NUL-terminated when supplied.
 * Messages may be truncated at a UTF-8 boundary; use the returned status code.
 * No allocation/free API is needed for errors. */
typedef struct MgGrepError {
    char message[512];
} MgGrepError;

/* Length-delimited bytes, not a C string. data may be NULL only if len == 0.
 * Input is borrowed for the duration of a call. Output belongs to its result. */
typedef struct MgGrepBytes {
    const uint8_t *data;
    size_t len;
} MgGrepBytes;

#define MG_GREP_PATTERN_REGEX   UINT32_C(0)
#define MG_GREP_PATTERN_LITERAL UINT32_C(1)

#define MG_GREP_CASE_SENSITIVE   UINT32_C(0)
#define MG_GREP_CASE_INSENSITIVE UINT32_C(1)
#define MG_GREP_CASE_SMART       UINT32_C(2)

#define MG_GREP_SCOPE_LINE      UINT32_C(0)
#define MG_GREP_SCOPE_MULTILINE UINT32_C(1)

#define MG_GREP_NEWLINE_LF   UINT32_C(0)
#define MG_GREP_NEWLINE_CRLF UINT32_C(1)

/* All fields are validated. Boolean fields accept only 0 or 1.
 * Zero initialization selects regex, case-sensitive, line-oriented LF search.
 * dot_all == 1 requires MULTILINE and REGEX. */
typedef struct MgGrepMatchOptions {
    uint32_t pattern_kind;
    uint32_t case_mode;
    uint32_t scope;
    uint32_t newline;
    uint32_t whole_line;
    uint32_t dot_all;
} MgGrepMatchOptions;

/* Service-supplied compilation limits, all strictly positive.
 * Regex/cache limits map to engine limits, not total process memory caps. */
typedef struct MgGrepCompileLimits {
    uint32_t max_patterns;
    uint64_t max_pattern_bytes;       /* Total bytes of all input patterns. */
    uint64_t max_regex_bytes;         /* Approximate compiled-program limit. */
    uint64_t max_dfa_cache_bytes;     /* Engine cache limit. */
} MgGrepCompileLimits;

#define MG_GREP_OUTPUT_LINES   UINT32_C(0)
#define MG_GREP_OUTPUT_MATCHES UINT32_C(1)
#define MG_GREP_OUTPUT_SELECT  UINT32_C(2)
#define MG_GREP_NO_LIMIT UINT64_MAX

/* Context is accepted only for LINES.
 * max_selected_units is the per-input rg -m limit: 0 means no search;
 * MG_GREP_NO_LIMIT means unlimited. It is not a database record limit.
 * In multiline mode, counting follows ripgrep 15.2.0 Searcher semantics. */
typedef struct MgGrepOutputOptions {
    uint32_t mode;
    uint32_t before_context;
    uint32_t after_context;
    uint64_t max_selected_units;
} MgGrepOutputOptions;

/* All values are strictly positive; caller supplies remaining request budgets.
 * Limits apply to one execute_batch call. timeout_ms is cooperative, not a
 * preemptive wall-clock guarantee. Exceeding any limit fails the whole batch. */
typedef struct MgGrepExecutionLimits {
    uint64_t max_input_bytes;   /* Sum of text lengths, checked before scanning. */
    uint64_t max_result_bytes;  /* Result payload including views and text. */
    uint64_t timeout_ms;
} MgGrepExecutionLimits;

#define MG_GREP_LINE_SELECTED UINT32_C(0)
#define MG_GREP_LINE_CONTEXT  UINT32_C(1)

/* Original field coordinates: 1-based line number, 0-based byte offset.
 * text includes the original line terminator when one is present. */
typedef struct MgGrepLine {
    uint64_t line_number;
    uint64_t start_byte;
    MgGrepBytes text;
    uint32_t role;
} MgGrepLine;

/* Original field byte interval [start_byte, end_byte).
 * text is the exact matched bytes. MATCHES omits empty matches; LINES may
 * contain an empty interval. Line numbers are 1-based (see contract). */
typedef struct MgGrepMatch {
    uint64_t start_byte;
    uint64_t end_byte;
    uint64_t start_line;
    uint64_t end_line;
    MgGrepBytes text;
} MgGrepMatch;

/* One entry per input, including unselected inputs. No database identifiers.
 * selected and scan_complete are 0 or 1. selected never depends on match_count.
 * scan_complete == 0 is permitted on successful SELECT/-m early termination.
 * Successful detail results are complete for the requested -m scope; v1 does
 * not silently truncate details to fit a resource budget. */
typedef struct MgGrepTextResult {
    uint32_t selected;
    uint32_t scan_complete;
    const MgGrepLine *lines;
    size_t line_count;
    const MgGrepMatch *matches;
    size_t match_count;
} MgGrepTextResult;

typedef struct MgGrepBatchView {
    const MgGrepTextResult *texts;
    size_t text_count;
} MgGrepBatchView;

/* pattern_count >= 1; patterns are OR-ed. Empty patterns are valid.
 * Copies pattern/configuration data before returning. *out_plan is NULL on
 * failure. Required: options, limits, out_plan and the pattern array.
 * Compilation limits are enforced before expensive matching setup. */
MgGrepStatus mg_grep_compile(
    const MgGrepBytes *patterns,
    size_t pattern_count,
    const MgGrepMatchOptions *options,
    const MgGrepCompileLimits *limits,
    MgGrepPlan **out_plan,
    MgGrepError *error);

/* Releases the caller's plan reference; existing sessions retain ownership.
 * NULL is accepted. Must not race with session_create on this handle. */
void mg_grep_plan_free(MgGrepPlan *plan);

/* Creates mutable execution state from an immutable compiled plan.
 * *out_session is NULL on failure. */
MgGrepStatus mg_grep_session_create(
    const MgGrepPlan *plan,
    MgGrepSession **out_session,
    MgGrepError *error);

/* The only operation permitted concurrently with execute_batch on a session.
 * Idempotent, sticky cancellation: it also cancels future calls on this session.
 * Does not revoke results from calls that already completed. NULL is a no-op. */
void mg_grep_session_cancel(MgGrepSession *session);

/* NULL is accepted. All execution/cancellation calls must have returned. */
void mg_grep_session_free(MgGrepSession *session);

/* Synchronous. Inputs must remain valid through return. Database NULL rows
 * are filtered by C++; {NULL, 0} represents an empty, non-NULL text.
 * text_count == 0 accepts texts == NULL and returns an empty result.
 * One session cannot execute concurrent batches.
 * On failure, *out_result == NULL; no partial batch is returned.
 * Required: session, output, limits, out_result; error is optional. */
MgGrepStatus mg_grep_execute_batch(
    MgGrepSession *session,
    const MgGrepBytes *texts,
    size_t text_count,
    const MgGrepOutputOptions *output,
    const MgGrepExecutionLimits *limits,
    MgGrepResult **out_result,
    MgGrepError *error);

/* NULL input yields NULL. A non-NULL view and all nested pointers remain valid
 * until result_free, independently of input, session and plan lifetimes. */
const MgGrepBatchView *mg_grep_result_view(const MgGrepResult *result);

/* NULL is accepted. Results can only be freed with this function. */
void mg_grep_result_free(MgGrepResult *result);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MILVUS_GREP_H */
```
