# Grep NGRAM Index Acceleration

Created: 2026-11-03. Updated: 2026-11-04.

This document describes the current workspace implementation and is the single maintained reference for Grep/NGRAM index design. Grep arguments, matching results, and the overall QueryScan protocol are covered by the [Grep integration design](20261008-segcore-grep-ripgrep-integration.md). Historical proposals, development logs, and performance reports are excluded.

## 1. Current Scope

- TEXT supports explicitly created NGRAM indexes through the existing scalar index build, persistence, and load paths.
- Grep extracts necessary literal conditions from Rust regex HIR and transfers them to C++ through Protobuf.
- Sealed VARCHAR/TEXT segments can filter candidates through the dedicated `INgramReader` condition-tree interface, then read original values and perform exact matching in Rust.
- Eligible wide-OR queries with small limits first probe the input, then use a configurable match ratio to choose continued scanning or index lookup.
- This design does not introduce TEXT AUTOINDEX/HYBRID, integrate FM-index, add an index file format, or enable ordinary LIKE, `=~`, and `!~` filters on TEXT.

Long, mostly distinct documents are a target workload, not a data prerequisite. Short texts, duplicate values, NULLs, empty strings, Unicode, embedded NULs, and newlines must remain correct. Distinct documents may share many grams, so full-value cardinality does not establish substring selectivity.

## 2. TEXT Index Creation and Building

### 2.1 Admission Rules

| TEXT CreateIndex request | `scalarAutoIndex.enable=false` | `scalarAutoIndex.enable=true` |
|---|---|---|
| Explicit NGRAM with valid gram parameters | Create NGRAM | Create NGRAM, preserving the explicit type |
| AUTOINDEX | Reject | Reject |
| Missing or empty index_type | Reject | Reject |
| Another user scalar index type | Reject | Reject |

Both `min_gram` and `max_gram` must be integers satisfying `0 < min_gram <= max_gram`. Length follows the existing tokenizer's Unicode-character semantics, not UTF-8 byte length. No TEXT-specific gram defaults are introduced.

`ScalarAutoIndexEnable` does not automatically select NGRAM for TEXT. Proxy checks the explicit type before automatic index handling, excludes TEXT from scalar automatic resolution, and verifies that the resolved type remains NGRAM with `isAutoIndex=false`. VARCHAR automatic index policy is unchanged; its global policy can override an explicitly requested index type.

“Only NGRAM” refers to user scalar indexes and does not exclude existing full-text search features. NGRAM indexes raw text with its own gram tokenizer; the field's full-text analyzer does not determine its case handling, stop words, or stemming.

Code: [Proxy parameter resolution](../../../internal/proxy/ddl/task_index.go), [NGRAM parameter validation](../../../internal/util/indexparamcheck/ngram_index_checker.go).

### 2.2 Native Data Path

Index definitions, task scheduling, and publication use existing paths. The field schema remains TEXT rather than being relabeled VARCHAR. This change adds no independent metadata write path, DDL message, or index file format.

For Milvus-written manifest-backed TEXT, `BuildSession` uses `VisitTextFieldDataFromManifest` to decode inline/LOB cells before supplying text to the scalar NGRAM builder. External mapped columns use the existing generic read path. Encoded LOB references must not be indexed as document contents. NULLs, empty strings, and duplicate texts retain their original row coordinates, and index row count must match segment row count.

The existing NGRAM builder/loader and heap/mmap loading capabilities are reused. Successful creation or `state=Finished` alone does not establish a complete build: functional verification also requires full indexed-row progress, actual index files, and query results after loading.

Code: [BuildSession](../../../internal/core/src/indexbuilder/BuildSession.cpp), [input materialization](../../../internal/core/src/indexbuilder/BuildInputMaterializer.cpp), [NgramIndexBuilder](../../../internal/core/src/index/scalar/ngram/NgramIndexBuilder.cpp), [NgramIndexLoader](../../../internal/core/src/index/scalar/ngram/NgramIndexLoader.cpp).

## 3. Necessary Condition Tree

The required invariant is:

```text
Rust Grep matches a value => the value satisfies the index condition
```

The converse does not hold. The condition tree does not encode final match order, same-line requirements, boundaries, line numbers, or spans, and cannot replace Rust matching results.

The tree contains `True`, `Literal`, `And`, and `Or`. A Literal denotes a case-sensitive byte substring; True means no pruning. Multiple `-e` patterns are combined with OR. Literal-mode (`-F`) patterns are escaped before extraction. Case options and scoped inline flags are handled through Rust HIR. A case-insensitive pattern must not become a single lowercase literal.

| HIR structure | Conservative condition |
|---|---|
| Literal | A substring that must occur |
| Concatenation | AND of child conditions, with bounded combination of adjacent literal sequences |
| Alternation | OR covering every branch |
| Capture | Inner condition |
| Repetition with minimum zero | True |
| Repetition with minimum at least one | Inner necessary condition; the repetition count may be omitted |
| Empty expression, assertion, or character class without a useful bounded expansion | True |

Small character classes and case variants may be enumerated within the budget. Dropping some AND constraints is safe; dropping a viable OR branch is not. If an OR branch cannot be represented, the condition must be conservatively weakened. Regex validation and final matcher compilation still run: extraction fallback must not make an invalid pattern acceptable.

### 3.1 Extraction and Encoding Budgets

| Layer | Current bound | Behavior on exhaustion |
|---|---|---|
| Rust condition extraction | At most 127 nodes; literals are valid UTF-8 substrings of 2..64 bytes | Weaken conditions, using True when necessary |
| HIR traversal | 4,096 visits and 64 recursive levels | Weaken unanalyzed portions |
| Finite sequence expansion | At most 32 strings, each at most 64 bytes | Stop combination or discard the sequence cover |
| Additional literal-mode HIR preparation | 4,096 estimated bytes | Use True |
| Rust Protobuf encoding | 16 KiB and 192 Protobuf message levels | Encode a complete True root |
| NGRAM condition planning | More than 127 nodes, or literals unsuitable for min_gram | Conservatively weaken the tree or affected conditions |

These are optimization budgets, not additional regex request rejection criteria. Actual allocation failures, malformed internal trees, and matcher compilation errors remain errors. Protobuf depth includes intermediate messages such as `Conditions`; it is distinct from logical tree depth.

Code: [Rust Plan](../../../internal/core/thirdparty/grep/src/plan.rs), [condition extraction](../../../internal/core/thirdparty/grep/src/condition.rs), [encoding](../../../internal/core/thirdparty/grep/src/condition_proto.rs).

## 4. Rust-to-C++ Protocol and Ownership

The shared schema is in [cgo_msg.proto](../../../pkg/proto/cgo_msg.proto):

```proto
message LiteralConditionTree {
  LiteralCondition root = 1;
}
message LiteralCondition {
  oneof kind {
    AlwaysTrue always_true = 1;
    bytes literal = 2;
    Conditions all = 3;
    Conditions any = 4;
  }
}
message AlwaysTrue {}
message Conditions {
  repeated LiteralCondition children = 1;
}
```

This is in-process binary Protobuf, not JSON, regex text, or a persisted index format. There is no version field, reserved field, custom header, or separate Grep proto file. Rust and C++ generate their own bindings from the same schema; bytes are transferred without sharing language-specific object layouts.

Rust Plan caches the matcher, `max_count`, condition tree, and encoded payload. The C ABI exposes a view of the Plan-owned payload. C++ `Plan::ConditionPayload()` uses that view while the Plan remains alive, and `DecodeLiteralConditionTree` copies it into a `RegexLiteralCondition` that owns its literals and children. The reader borrows the tree only for the call and retains neither Rust pointers nor Session state.

C++ checks complete Protobuf parsing, root/kind presence, nonempty literals, and at least two children for AND/OR. It trusts Rust's size, depth, and UTF-8 guarantees rather than adding duplicate Grep payload limits; the Protobuf recursion limit is relaxed to accept valid Rust output. Structural errors report `DataFormatBroken` rather than becoming True. A structurally valid tree exceeding the planning budget can decode successfully and subsequently be rejected for pruning by the planner.

Standard Protobuf unknown-field and merge semantics apply. The decoder does not promise to detect every concatenation of valid fields. The pointer/length supplied by the call defines the frame.

Code: [Rust FFI](../../../internal/core/thirdparty/grep/src/ffi.rs), [C++ Plan wrapper](../../../internal/core/src/grep/Grep.cpp), [C++ decoding](../../../internal/core/src/grep/IndexCondition.cpp), [Rust binding generation](../../../internal/core/thirdparty/grep/build.rs).

## 5. NGRAM Reader and Candidate Bitmap

Grep uses dedicated methods on [INgramReader](../../../internal/core/src/index/contracts/query/INgramReader.h):

```cpp
bool CanHandleConditionTree(const RegexLiteralCondition& condition) const;
void ConditionTreeCandidates(const RegexLiteralCondition& condition,
                             TargetBitmap& candidates) const;
```

The reader receives a C++ condition tree, not Protobuf. Grep adds no PatternOp and does not use RE2 to parse Rust regexes. Ordinary filters using `RegexMatch` and Grep retain their respective matcher semantics.

`PlanNgramRegexCondition` restricts conditions to lookups supported by min_gram. Short literals become True. AND may omit True children, whereas OR containing True cannot query only the remaining branches. `ApplyNgramRegexPlan` applies intersections and branch unions to the incoming candidate set. Direct literal children of an AND are passed together to the reader; each OR branch starts from the incoming set.

Candidates use a bitmap over the full segment row domain and are intersected with the caller's input range. This is not a streaming interface that returns only enough candidates to satisfy limit. Grep currently has no lightweight candidate-estimation API or posting iterator; smaller offsets windows do not guarantee less underlying posting work. Regardless of candidate precision, Grep verifies values in Rust and generates complete matching details.

Code: [condition planning](../../../internal/core/src/index/scalar/ngram/NgramRegex.cpp), [tree evaluation](../../../internal/core/src/index/scalar/ngram/NgramRegex.h), [NgramIndexReader](../../../internal/core/src/index/scalar/ngram/NgramIndexReader.cpp).

## 6. Segment Routing and Execution

### 6.1 Index Eligibility

`PrepareGrepIndex` considers only:

- Sealed segments with nonempty input offsets.
- Fields with a whole-field NGRAM registration and an empty JSON path.
- Non-True conditions that the reader can handle.
- Capabilities providing ngram candidates, with no nesting and a matching field type.
- Pinned readers in the Row coordinate domain, with Count equal to segment row count and a matching ValueType.

Missing eligible indexes, growing segments, True conditions, and unsuitable domains or coverage use scanning. Actual errors during loading, pinning, decoding, index lookup, or reading propagate rather than being hidden as fallback. There is no additional explicit generation-ID comparison here; sealed read leases/snapshots and index pins protect access under the existing lifecycle contracts.

### 6.2 Probe and Match-Ratio Configuration

Probe requires all of the following: an eligible NGRAM reader, an incoming condition rooted at OR, at least 32 Literal leaves in the whole tree, and a local execution limit at most 128. Other queries with an eligible NGRAM reader query the index directly.

```text
sampleRows = min(input_count, clamp(2 * limit, 32, 256))

Read and match small batches:
    limit satisfied or inputs exhausted => return
Once sampleRows inputs have been sampled:
    matches / sampled_rows >= probeMatchRatio => keep scanning; no later index switch
    otherwise => query the index once for the unread suffix
```

```yaml
function:
  grep:
    probeMatchRatio: 0.05
    batchSize: 128
```

| Parameter or boundary | Behavior |
|---|---|
| probeMatchRatio | Default 0.05; finite range [0,1]; invalid values, NaN, and Inf fall back to the default |
| Configuration updates | Refreshable; snapshotted when preparing GrepOp and applied to newly prepared operators |
| Ratio equal to threshold | Continue scanning |
| Threshold 0 / 1 | Always scan after sampling / require all sampled inputs to match |
| L1, L10 | Sample cap 32; the default requires at least 2 matches to continue scanning after a full sample |
| L100 | Sample cap 200; the default requires at least 10 matches |
| L128 | Sample cap 256; the default requires at least 13 matches |
| Fewer inputs than the sample cap | Scan to input exhaustion without an additional index lookup |

The match ratio counts input record positions, not matching lines or spans. NULLs count as nonmatches, and repeated offsets count at each input position. The first TEXT batch contains at most one row; the first VARCHAR batch contains at most 32. Subsequent batches are at most twice the largest prior batch, further bounded by the original read window, remaining sample size, and remaining match quota. A first batch with zero matches does not immediately trigger index lookup.

There is no Probe byte budget, fixed total sample of 32 rows, or `8 * limit` completion estimate. The sample cap limits observation before choosing a path; the query need not finish within that sample. An unrepresentative prefix can still select a slower path, and larger samples increase read work for low-hit-rate long texts. The configurable 5% default is an initial choice, not a proven universal optimum.

Configuration flows through [paramtable](../../../pkg/util/paramtable/function_param.go), [GrepOp](../../../internal/util/function/chain/operator_grep.go), and the [Go native adapter](../../../internal/util/segcore/grep.go) to `CGrepScanOptions.probe_match_ratio`. The C ABI is built together with Go; structure changes require rebuilding both. No RPC parameter is added.

### 6.3 Ordering, Reads, and Resource Lifetime

- `BuildGrepCandidates` initializes the row-domain bitmap from the unread input suffix before calling the reader. A physical row appearing again in the suffix must not be removed merely because it was read in the prefix.
- Execution follows the original offsets array rather than sorting by row offset. Returned positions refer to the original input, preserving duplicate-record semantics.
- `GrepReadCount` validates the original window and handles VARCHAR chunk boundaries. Probe sub-batches do not reset those validation boundaries.
- The Index path collects candidate offsets per window rather than materializing every candidate offset at once. Ordinary Scan/Index raw reads still use window-sized batches and are not uniformly capped by the remaining limit; Rust matching stops according to the remaining quota.
- Owned TEXT reads deduplicate repeated physical offsets within a batch, then map the fetched values back to all input positions.
- After candidate generation, `index.reset()` releases the reader pin while the candidate bitmap remains independently owned. Choosing continued scanning also releases the pin. RAII releases resources on early return or exceptions; the sealed read lease continues to cover raw reads.
- `-m` caps matching lines per record, not index candidate count or the Query limit. `-m 0` returns an empty result after validating and compiling the pattern, without index lookup or raw reads. Positive values retain the routing described above.

Code: [SegmentGrep.cpp](../../../internal/core/src/grep/SegmentGrep.cpp).

### 6.4 Limits Across Segments

[QueryScan](../../../internal/querynodev2/segments/query_scan.go) runs an independent chain for each segment with the same request limit, waits for completion, and then merges results. Reaching the global limit does not immediately cancel other segments. Therefore, at most one NGRAM lookup per native call does not mean at most one lookup per request.

## 7. Functional Verification and Remaining Work

Functional verification covers explicit TEXT NGRAM admission, actual metadata persistence, DataNode building, index file existence, load/reload/restart, and complete-result equality between indexed and raw Grep paths. This does not establish acceptance for every deployment mode or failure scenario.

Primary test entry points:

- [TEXT index admission](../../../internal/proxy/ddl/task_index_text_test.go) and [NGRAM parameter validation](../../../internal/util/indexparamcheck/ngram_index_checker_test.go).
- [Native TEXT building](../../../internal/core/src/indexbuilder/NgramTextBuildSessionTest.cpp).
- [Rust extraction correctness](../../../internal/core/thirdparty/grep/src/condition_tests.rs).
- [NGRAM condition trees](../../../internal/core/src/index/scalar/ngram/NgramGrepTest.cpp) and [segment execution](../../../internal/core/src/grep/SegmentGrepTest.cpp): candidate supersets, original ordering, repeated offsets, NULLs, LOBs, limits, cancellation, index/read errors, and Probe transitions.
- [Configuration](../../../pkg/util/paramtable/function_param_test.go), [operator snapshots](../../../internal/util/function/chain/operator_grep_test.go), and the [Go/native adapter](../../../internal/util/segcore/grep_test.go).

Boundaries and follow-up work:

- Ordinary TEXT LIKE, `=~`, and `!~` filters remain subject to the [parser's TEXT restriction](../../../internal/parser/planparserv2/parser_visitor.go). Creating NGRAM does not enable these filters. Existing full-text operators are separate from this restriction.
- TEXT AUTOINDEX/HYBRID, FM-index Grep integration, index-estimation APIs, limit pushdown into postings, and uniformly limit-aware raw reads are not implemented by this design.
- Complete CDC, compaction, remote object storage, and fault-injection acceptance remains outstanding. Reusing existing index definitions and lifecycle mechanisms is not evidence that these scenarios have been verified.
- Native build tests have encountered teardown failures, and embedded standalone instances have encountered shutdown problems. Passing functional assertions does not establish clean shutdown; index admission and Probe changes do not resolve those issues.

## 8. Documentation Corrections

- Removed obsolete claims that Grep has no index stage or that routing and public-query validation are still awaiting implementation.
- Removed superseded byte budgets, `8 * limit` estimates, immediate fallback after a first miss, and fixed total samples of 32 rows. Only the current limit-aware sampling policy is maintained.
- Distinguished Rust extraction/encoding budgets, C++ structural checks, and NGRAM planning limits. The 127-node and 64-level bounds are not additional protocol rejection rules.
- Included Rust Plan's `max_count` and `-m 0` behavior, distinguishing Query limits, per-record line caps, and index candidate counts.
- Clarified that the implementation has no additional generation-ID comparison, global-limit early cancellation, or uniform remaining-limit cap on raw read batches.
- Removed performance content and separate reports. Latency changes are not used to infer actual index routing for a request or its segments.
