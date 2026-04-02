// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include "common/type_c.h"
#include "segcore/segment_c.h"
#include "segcore/plan_c.h"

// Forward declarations for Arrow C Data Interface
struct ArrowArray;
struct ArrowSchema;

// Export a per-segment SearchResult as an Arrow RecordBatch via the C Data Interface.
//
// This function performs:
// 1. FilterInvalidSearchResult: removes rows with offset == -1
// 2. FillPrimaryKeys: reads PKs from the segment
// 3. Reads extra fields (e.g., for L0 rerank) via bulk_subscript if extra_field_ids is provided
// 4. Builds Arrow RecordBatch with columns: $id (int64/string), $score (float32), $seg_offset (int64),
//    plus any extra fields requested
// 5. Exports via Arrow C Data Interface (zero-copy)
//
// The RecordBatch rows are ordered by NQ (NQ=0 results first, then NQ=1, etc.).
// The per-NQ row counts are stored as comma-separated values in Arrow schema metadata
// under key "topk_per_nq" (e.g., "3,2,5" means NQ=0 has 3 rows, NQ=1 has 2, NQ=2 has 5).
//
// extra_field_ids: optional array of field IDs to export (e.g., fields needed by L0 rerank).
//                  Pass NULL with num_extra_fields=0 if no extra fields are needed.
// out_array and out_schema must point to caller-allocated ArrowArray/ArrowSchema structs.
// The caller is responsible for releasing them (via ArrowArrayRelease/ArrowSchemaRelease).
// If the result is empty (0 valid rows), out_array and out_schema are still valid (0-row batch).
CStatus
ExportSearchResultAsArrow(CSearchResult c_search_result,
                          CSearchPlan c_plan,
                          const int64_t* extra_field_ids,
                          int64_t num_extra_fields,
                          struct ArrowArray* out_array,
                          struct ArrowSchema* out_schema);

// Fill output fields for multiple segments in a single call, producing
// results in the specified output order.
//
// This replaces the per-segment GetFieldDataByOffsets + Go-side scatter pattern.
// The caller provides the reduce result as parallel arrays:
//   result_seg_indices[i] = which search_results[] this row came from
//   result_seg_offsets[i] = seg_offset within that segment
// The output proto FieldsData is assembled in the order of these arrays.
//
// Internally: groups by segment, calls FillTargetEntry per segment,
// then uses MergeDataArray to produce the correctly-ordered output.
CStatus
FillOutputFieldsOrdered(CSearchResult* search_results,
                        int64_t num_search_results,
                        CSearchPlan c_plan,
                        const int32_t* result_seg_indices,
                        const int64_t* result_seg_offsets,
                        int64_t total_rows,
                        CProto* out_result);

#ifdef __cplusplus
}
#endif
