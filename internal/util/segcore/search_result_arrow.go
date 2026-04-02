// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package segcore

/*
#cgo pkg-config: milvus_core

#include <stdlib.h>
#include <stdint.h>
#include "common/type_c.h"
#include "segcore/segment_c.h"
#include "segcore/plan_c.h"

// Arrow C Data Interface structs (standard ABI, layout matches Go cdata package)
#ifndef ARROW_C_DATA_INTERFACE
#define ARROW_C_DATA_INTERFACE

struct ArrowSchema {
    const char* format;
    const char* name;
    const char* metadata;
    int64_t flags;
    int64_t n_children;
    struct ArrowSchema** children;
    struct ArrowSchema* dictionary;
    void (*release)(struct ArrowSchema*);
    void* private_data;
};

struct ArrowArray {
    int64_t length;
    int64_t null_count;
    int64_t offset;
    int64_t n_buffers;
    int64_t n_children;
    const void** buffers;
    struct ArrowArray** children;
    struct ArrowArray* dictionary;
    void (*release)(struct ArrowArray*);
    void* private_data;
};

#endif  // ARROW_C_DATA_INTERFACE

CStatus
ExportSearchResultAsArrow(CSearchResult c_search_result,
                          CSearchPlan c_plan,
                          const int64_t* extra_field_ids,
                          int64_t num_extra_fields,
                          struct ArrowArray* out_array,
                          struct ArrowSchema* out_schema);

CStatus
FillOutputFieldsOrdered(CSearchResult* search_results,
                        int64_t num_search_results,
                        CSearchPlan c_plan,
                        const int32_t* result_seg_indices,
                        const int64_t* result_seg_offsets,
                        int64_t total_rows,
                        CProto* out_result);
*/
import "C"

import (
	"strconv"
	"strings"
	"unsafe"

	"github.com/apache/arrow/go/v17/arrow"
	"github.com/apache/arrow/go/v17/arrow/cdata"
	"github.com/cockroachdb/errors"
)

// ExportSearchResultAsArrow exports a per-segment C++ SearchResult as an Arrow RecordBatch.
// The RecordBatch has columns: $id (int64/string), $score (float32), $seg_offset (int64),
// plus any extra fields specified by extraFieldIDs (e.g., fields needed by L0 rerank).
// Per-NQ row counts are in schema metadata under key "topk_per_nq".
func ExportSearchResultAsArrow(result *SearchResult, plan *SearchPlan, extraFieldIDs []int64) (arrow.Record, error) {
	if result == nil {
		return nil, errors.New("nil search result")
	}
	if plan == nil || plan.cSearchPlan == nil {
		return nil, errors.New("nil search plan")
	}

	var cArr C.struct_ArrowArray
	var cSchema C.struct_ArrowSchema

	var extraPtr *C.int64_t
	if len(extraFieldIDs) > 0 {
		extraPtr = (*C.int64_t)(unsafe.Pointer(&extraFieldIDs[0]))
	}

	status := C.ExportSearchResultAsArrow(
		result.cSearchResult,
		plan.cSearchPlan,
		extraPtr,
		C.int64_t(len(extraFieldIDs)),
		&cArr,
		&cSchema,
	)
	if err := ConsumeCStatusIntoError(&status); err != nil {
		return nil, errors.Wrap(err, "ExportSearchResultAsArrow failed")
	}

	// Import into Go Arrow RecordBatch via C Data Interface
	goCArr := (*cdata.CArrowArray)(unsafe.Pointer(&cArr))
	goCSchema := (*cdata.CArrowSchema)(unsafe.Pointer(&cSchema))
	defer func() {
		cdata.ReleaseCArrowArray(goCArr)
		cdata.ReleaseCArrowSchema(goCSchema)
	}()

	record, err := cdata.ImportCRecordBatch(goCArr, goCSchema)
	if err != nil {
		return nil, errors.Wrap(err, "failed to import Arrow RecordBatch")
	}

	return record, nil
}

// ParseTopkPerNQ extracts the per-NQ row counts from Arrow schema metadata.
// Returns the counts parsed from the "topk_per_nq" metadata key.
func ParseTopkPerNQ(schema *arrow.Schema) ([]int64, error) {
	md := schema.Metadata()
	idx := md.FindKey("topk_per_nq")
	if idx < 0 {
		return nil, errors.New("topk_per_nq metadata not found in Arrow schema")
	}
	val := md.Values()[idx]
	if val == "" {
		return nil, nil // empty result
	}

	parts := strings.Split(val, ",")
	counts := make([]int64, len(parts))
	for i, p := range parts {
		v, err := strconv.ParseInt(strings.TrimSpace(p), 10, 64)
		if err != nil {
			return nil, errors.Wrapf(err, "invalid topk_per_nq value %q at index %d", p, i)
		}
		counts[i] = v
	}
	return counts, nil
}

// FillOutputFieldsOrdered reads output fields from multiple segments in a single CGO call,
// producing results in the specified output order.
//
// segIndices[i] specifies which results[] element the i-th output row came from.
// segOffsets[i] specifies the segment-internal offset for that row.
// Returns serialized schemapb.SearchResultData proto with only FieldsData populated.
func FillOutputFieldsOrdered(
	results []*SearchResult,
	plan *SearchPlan,
	segIndices []int32,
	segOffsets []int64,
) ([]byte, error) {
	if plan == nil || plan.cSearchPlan == nil {
		return nil, errors.New("nil search plan")
	}
	if len(segIndices) == 0 {
		return nil, nil
	}

	cResults := make([]C.CSearchResult, len(results))
	for i, r := range results {
		cResults[i] = r.cSearchResult
	}

	var cProto C.CProto
	status := C.FillOutputFieldsOrdered(
		&cResults[0],
		C.int64_t(len(results)),
		plan.cSearchPlan,
		(*C.int32_t)(unsafe.Pointer(&segIndices[0])),
		(*C.int64_t)(unsafe.Pointer(&segOffsets[0])),
		C.int64_t(len(segIndices)),
		&cProto,
	)
	if err := ConsumeCStatusIntoError(&status); err != nil {
		return nil, errors.Wrap(err, "FillOutputFieldsOrdered failed")
	}

	if cProto.proto_size == 0 {
		return nil, nil
	}
	return getCProtoBlob(&cProto), nil
}
