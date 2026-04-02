/*
 * Licensed to the LF AI & Data foundation under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package tasks

import (
	"github.com/apache/arrow/go/v17/arrow"
	"github.com/apache/arrow/go/v17/arrow/array"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/util/function/chain"
	"github.com/milvus-io/milvus/pkg/v2/util/merr"
)

// MarshalReduceResult converts a ReduceResult into a serialized SearchResultData proto.
//
// This is the Go replacement for the C++ ReduceHelper::GetSearchResultDataSlice + Marshal.
// It takes the reduced DataFrame ($id + $score columns) from HeapMergeReduce
// and produces the final schemapb.SearchResultData.
//
// Output field data from Late Materialization should be added to the result
// after this function returns, by merging fieldData per segment.
//
// The function handles both int64 and varchar PK types.
func MarshalReduceResult(reduceResult *ReduceResult) (*schemapb.SearchResultData, error) {
	if reduceResult == nil || reduceResult.DF == nil {
		return nil, merr.WrapErrServiceInternal("nil reduce result")
	}

	df := reduceResult.DF

	result := &schemapb.SearchResultData{
		NumQueries: int64(df.NumChunks()),
		TopK:       maxChunkSize(df),
		Topks:      df.ChunkSizes(),
		Scores:     make([]float32, 0, df.NumRows()),
		Ids:        &schemapb.IDs{},
	}

	// Export IDs
	idCol := df.Column(idFieldName)
	if idCol == nil {
		return nil, merr.WrapErrServiceInternal("MarshalReduceResult: $id column not found")
	}

	// Detect PK type from Arrow data type
	isInt64PK := idCol.DataType() == arrow.PrimitiveTypes.Int64

	if isInt64PK {
		ids := make([]int64, 0, df.NumRows())
		for _, chunk := range idCol.Chunks() {
			arr := chunk.(*array.Int64)
			for i := 0; i < arr.Len(); i++ {
				ids = append(ids, arr.Value(i))
			}
		}
		result.Ids = &schemapb.IDs{
			IdField: &schemapb.IDs_IntId{
				IntId: &schemapb.LongArray{Data: ids},
			},
		}
	} else {
		ids := make([]string, 0, df.NumRows())
		for _, chunk := range idCol.Chunks() {
			arr := chunk.(*array.String)
			for i := 0; i < arr.Len(); i++ {
				ids = append(ids, arr.Value(i))
			}
		}
		result.Ids = &schemapb.IDs{
			IdField: &schemapb.IDs_StrId{
				StrId: &schemapb.StringArray{Data: ids},
			},
		}
	}

	// Export Scores
	scoreCol := df.Column(scoreFieldName)
	if scoreCol == nil {
		return nil, merr.WrapErrServiceInternal("MarshalReduceResult: $score column not found")
	}
	for _, chunk := range scoreCol.Chunks() {
		arr := chunk.(*array.Float32)
		for i := 0; i < arr.Len(); i++ {
			result.Scores = append(result.Scores, arr.Value(i))
		}
	}

	return result, nil
}

// GroupBySegment groups SegmentSources by their input (segment) index.
// Returns a map from inputIdx → list of (SegOffset, position in result).
// Used by Late Materialization to batch CGO calls per segment.
func GroupBySegment(sources [][]SegmentSource) map[int][]OffsetPosition {
	groups := make(map[int][]OffsetPosition)
	globalPos := 0
	for _, chunkSources := range sources {
		for _, src := range chunkSources {
			groups[src.InputIdx] = append(groups[src.InputIdx], OffsetPosition{
				SegOffset:   src.SegOffset,
				ResultPos:   globalPos,
				OriginalIdx: src.OriginalIdx,
			})
			globalPos++
		}
	}
	return groups
}

// OffsetPosition tracks a segment offset and its position in the final result.
type OffsetPosition struct {
	SegOffset   int64 // Offset within the C++ segment
	ResultPos   int   // Position in the final flat result arrays
	OriginalIdx int   // Original index in the input segment's chunk
}

func maxChunkSize(df *chain.DataFrame) int64 {
	var max int64
	for _, size := range df.ChunkSizes() {
		if size > max {
			max = size
		}
	}
	return max
}
