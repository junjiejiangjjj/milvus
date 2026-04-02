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

#ifndef ARROW_C_STREAM_INTERFACE
#define ARROW_C_STREAM_INTERFACE

struct ArrowArrayStream {
    int (*get_schema)(struct ArrowArrayStream*, struct ArrowSchema*);
    int (*get_next)(struct ArrowArrayStream*, struct ArrowArray*);
    const char* (*get_last_error)(struct ArrowArrayStream*);
    void (*release)(struct ArrowArrayStream*);
    void* private_data;
};

#endif  // ARROW_C_STREAM_INTERFACE

CStatus
ExportSearchResultAsArrowStream(CSearchResult c_search_result,
                                CSearchPlan c_plan,
                                const int64_t* extra_field_ids,
                                int64_t num_extra_fields,
                                struct ArrowArrayStream* out_stream);

CStatus
FillOutputFieldsOrdered(CSearchResult* search_results,
                        int64_t num_search_results,
                        CSearchPlan c_plan,
                        const int32_t* result_seg_indices,
                        const int64_t* result_seg_offsets,
                        int64_t total_rows,
                        CProto* out_result);

void
GetSearchResultMetadata(CSearchResult c_search_result,
                        bool* has_group_by,
                        int64_t* group_size,
                        int64_t* scanned_remote_bytes,
                        int64_t* scanned_total_bytes);
*/
import "C"

import (
	"runtime"
	"unsafe"

	"github.com/apache/arrow/go/v17/arrow/arrio"
	"github.com/apache/arrow/go/v17/arrow/cdata"
	"github.com/cockroachdb/errors"
)

// ExportSearchResultAsArrowStream exports a per-segment C++ SearchResult as a stream
// of Arrow RecordBatches, one per NQ query.
// Each RecordBatch has columns: $id (int64/string), $score (float32), $seg_offset (int64),
// plus any extra fields specified by extraFieldIDs (e.g., fields needed by L0 rerank).
// The caller should iterate with reader.Read() until io.EOF.
func ExportSearchResultAsArrowStream(result *SearchResult, plan *SearchPlan, extraFieldIDs []int64) (arrio.Reader, error) {
	if result == nil {
		return nil, errors.New("nil search result")
	}
	if plan == nil || plan.cSearchPlan == nil {
		return nil, errors.New("nil search plan")
	}

	var extraPtr *C.int64_t
	if len(extraFieldIDs) > 0 {
		extraPtr = (*C.int64_t)(unsafe.Pointer(&extraFieldIDs[0]))
	}

	var cStream C.struct_ArrowArrayStream
	status := C.ExportSearchResultAsArrowStream(
		result.cSearchResult,
		plan.cSearchPlan,
		extraPtr,
		C.int64_t(len(extraFieldIDs)),
		&cStream,
	)
	runtime.KeepAlive(extraFieldIDs)
	runtime.KeepAlive(result)
	runtime.KeepAlive(plan)
	if err := ConsumeCStatusIntoError(&status); err != nil {
		return nil, errors.Wrap(err, "ExportSearchResultAsArrowStream failed")
	}

	goStream := (*cdata.CArrowArrayStream)(unsafe.Pointer(&cStream))
	reader, err := cdata.ImportCRecordReader(goStream, nil)
	if err != nil {
		return nil, errors.Wrap(err, "failed to import Arrow RecordBatchReader")
	}

	return reader, nil
}

// FillOutputFieldsOrdered reads output fields from multiple segments in a single CGO call,
// producing results in the specified output order.
// Storage cost is accumulated in the original SearchResult objects.
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

	cResults := make([]C.CSearchResult, len(results))
	for i, r := range results {
		cResults[i] = r.cSearchResult
	}

	var segIndicesPtr *C.int32_t
	var segOffsetsPtr *C.int64_t
	if len(segIndices) > 0 {
		segIndicesPtr = (*C.int32_t)(unsafe.Pointer(&segIndices[0]))
		segOffsetsPtr = (*C.int64_t)(unsafe.Pointer(&segOffsets[0]))
	}

	var cProto C.CProto
	status := C.FillOutputFieldsOrdered(
		&cResults[0],
		C.int64_t(len(results)),
		plan.cSearchPlan,
		segIndicesPtr,
		segOffsetsPtr,
		C.int64_t(len(segIndices)),
		&cProto,
	)
	runtime.KeepAlive(segIndices)
	runtime.KeepAlive(segOffsets)
	runtime.KeepAlive(cResults)
	runtime.KeepAlive(results)
	runtime.KeepAlive(plan)
	if err := ConsumeCStatusIntoError(&status); err != nil {
		return nil, errors.Wrap(err, "FillOutputFieldsOrdered failed")
	}

	if cProto.proto_size == 0 {
		return nil, nil
	}
	// Copy to Go heap and free the C malloc'd buffer immediately.
	// Do NOT use getCProtoBlob here — it calls cgoconverter.Extract which
	// removes the pointer from the lease registry without calling C.free,
	// leaking the buffer.
	goBytes := C.GoBytes(cProto.proto_blob, C.int(cProto.proto_size))
	C.free(cProto.proto_blob)
	return goBytes, nil
}
