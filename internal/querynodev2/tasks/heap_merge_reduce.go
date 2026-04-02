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
	"container/heap"
	"fmt"

	"github.com/apache/arrow/go/v17/arrow"
	"github.com/apache/arrow/go/v17/arrow/array"
	"github.com/apache/arrow/go/v17/arrow/memory"

	"github.com/milvus-io/milvus/internal/util/function/chain"
	"github.com/milvus-io/milvus/pkg/v2/util/merr"
)

const (
	idFieldName    = "$id"
	scoreFieldName = "$score"
)

// GroupByOptions configures GroupBy mode for HeapMergeReduce.
type GroupByOptions struct {
	GroupByFieldName string // column name in the DataFrame
	GroupSize        int64  // max results per group
}

// SegmentSource records the origin of each result row (for Late Materialization).
type SegmentSource struct {
	InputIdx    int   // which input DataFrame
	SegOffset   int64 // original segment offset (-1 if not available)
	OriginalIdx int   // original row index in the source chunk array
}

// ReduceResult contains the merged result and source tracking info.
type ReduceResult struct {
	DF      *chain.DataFrame  // merged result with $id + $score [+ $group_by]
	Sources [][]SegmentSource // per-chunk (NQ) sources for Late Materialization
}

const (
	// SegOffsetCol is the column name for segment offset in Arrow exports.
	SegOffsetCol = "$seg_offset"
)

// HeapMergeReduce merges multiple per-segment DataFrames using k-way heap merge.
// Corresponds to C++ ReduceHelper::ReduceResultData + ReduceSearchResultForOneNQ.
//
// Each input DataFrame must have $id and $score columns, with the same number of chunks (NQ).
// Optionally $seg_offset for Late Materialization tracking.
//
// The function performs PK deduplication: if the same PK appears in multiple segments,
// only the highest-scoring occurrence is kept.
func HeapMergeReduce(
	pool memory.Allocator,
	inputs []*chain.DataFrame,
	topK int64,
	groupByOpts *GroupByOptions,
) (*ReduceResult, error) {
	if len(inputs) == 0 {
		return nil, merr.WrapErrServiceInternal("HeapMergeReduce: no inputs")
	}

	numChunks := inputs[0].NumChunks()
	for i, df := range inputs {
		if df.NumChunks() != numChunks {
			return nil, merr.WrapErrServiceInternal(
				fmt.Sprintf("HeapMergeReduce: input %d has %d chunks, expected %d", i, df.NumChunks(), numChunks))
		}
	}

	// Detect PK type from first input
	idCol := inputs[0].Column(idFieldName)
	if idCol == nil {
		return nil, merr.WrapErrServiceInternal("HeapMergeReduce: $id column not found")
	}

	hasSegOffset := inputs[0].HasColumn(SegOffsetCol)
	hasGroupBy := groupByOpts != nil && groupByOpts.GroupByFieldName != "" &&
		inputs[0].HasColumn(groupByOpts.GroupByFieldName)

	if idCol.DataType().ID() == arrow.STRING {
		return heapMergeReduceImpl[string](pool, inputs, topK, groupByOpts, numChunks, hasSegOffset, hasGroupBy, true)
	}
	return heapMergeReduceImpl[int64](pool, inputs, topK, groupByOpts, numChunks, hasSegOffset, hasGroupBy, false)
}

// heapMergeReduceImpl is the generic implementation parameterized by PK type.
func heapMergeReduceImpl[PK comparable](
	pool memory.Allocator,
	inputs []*chain.DataFrame,
	topK int64,
	groupByOpts *GroupByOptions,
	numChunks int,
	hasSegOffset bool,
	hasGroupBy bool,
	isStringPK bool,
) (*ReduceResult, error) {
	chunkSizes := make([]int64, numChunks)
	allSources := make([][]SegmentSource, numChunks)

	// Determine output columns
	outCols := []string{idFieldName, scoreFieldName}
	groupByFieldName := ""
	if hasGroupBy {
		groupByFieldName = groupByOpts.GroupByFieldName
		outCols = append(outCols, groupByFieldName)
	}

	collector := chain.NewChunkCollector(outCols, numChunks)
	defer collector.Release()

	for chunkIdx := 0; chunkIdx < numChunks; chunkIdx++ {
		entries := buildMergeEntries(inputs, chunkIdx, hasSegOffset, hasGroupBy, groupByFieldName, isStringPK)

		var resultSources []SegmentSource
		if isStringPK {
			resultSources = mergeChunkString(pool, collector, entries, chunkIdx, topK, groupByOpts,
				hasGroupBy, groupByFieldName, inputs)
		} else {
			resultSources = mergeChunkInt64(pool, collector, entries, chunkIdx, topK, groupByOpts,
				hasGroupBy, groupByFieldName, inputs)
		}

		chunkSizes[chunkIdx] = int64(len(resultSources))
		allSources[chunkIdx] = resultSources
	}

	// Build output DataFrame
	builder := chain.NewDataFrameBuilder()
	defer builder.Release()
	builder.SetChunkSizes(chunkSizes)

	for _, colName := range outCols {
		if err := builder.AddColumnFromChunks(colName, collector.Consume(colName)); err != nil {
			return nil, err
		}
		builder.CopyFieldMetadata(inputs[0], colName)
	}
	builder.CopyAllMetadata(inputs[0])

	return &ReduceResult{
		DF:      builder.Build(),
		Sources: allSources,
	}, nil
}

// buildMergeEntries creates mergeEntry for each input's chunk.
func buildMergeEntries(
	inputs []*chain.DataFrame,
	chunkIdx int,
	hasSegOffset bool,
	hasGroupBy bool,
	groupByFieldName string,
	isStringPK bool,
) []*mergeEntry {
	entries := make([]*mergeEntry, 0, len(inputs))
	for inputIdx, df := range inputs {
		idChunked := df.Column(idFieldName)
		scoreChunked := df.Column(scoreFieldName)
		if idChunked == nil || scoreChunked == nil {
			continue
		}
		idChunk := idChunked.Chunk(chunkIdx)
		scoreChunk := scoreChunked.Chunk(chunkIdx)
		if idChunk.Len() == 0 {
			continue
		}

		scoreArr := scoreChunk.(*array.Float32)
		entry := &mergeEntry{
			inputIdx: inputIdx,
			cursor:   0,
			scoreArr: scoreArr,
		}

		if isStringPK {
			idArr := idChunk.(*array.String)
			entry.idString = idArr
			entry.indices = buildSortedIndicesString(idArr, scoreArr)
		} else {
			idArr := idChunk.(*array.Int64)
			entry.idInt64 = idArr
			entry.indices = buildSortedIndicesInt64(idArr, scoreArr)
		}

		if hasSegOffset {
			if soCol := df.Column(SegOffsetCol); soCol != nil {
				entry.segOffsetArr = soCol.Chunk(chunkIdx).(*array.Int64)
			}
		}
		if hasGroupBy {
			if gbCol := df.Column(groupByFieldName); gbCol != nil {
				entry.groupByArr = gbCol.Chunk(chunkIdx)
			}
		}

		if len(entry.indices) > 0 {
			entries = append(entries, entry)
		}
	}
	return entries
}

// mergeChunkInt64 performs the k-way merge for one chunk with int64 PK.
func mergeChunkInt64(
	pool memory.Allocator,
	collector *chain.ChunkCollector,
	entries []*mergeEntry,
	chunkIdx int,
	topK int64,
	groupByOpts *GroupByOptions,
	hasGroupBy bool,
	groupByFieldName string,
	inputs []*chain.DataFrame,
) []SegmentSource {
	h := &mergeHeapInt64{}
	heap.Init(h)
	for _, e := range entries {
		heap.Push(h, e)
	}

	var ids []int64
	var scores []float32
	var sources []SegmentSource

	if groupByOpts != nil && hasGroupBy {
		ids, scores, sources = mergeInt64GroupBy(h, topK, groupByOpts.GroupSize)
	} else {
		ids, scores, sources = mergeInt64Standard(h, topK)
	}

	// Build output arrays
	idBuilder := array.NewInt64Builder(pool)
	idBuilder.AppendValues(ids, nil)
	collector.Set(idFieldName, chunkIdx, idBuilder.NewArray())
	idBuilder.Release()

	scoreBuilder := array.NewFloat32Builder(pool)
	scoreBuilder.AppendValues(scores, nil)
	collector.Set(scoreFieldName, chunkIdx, scoreBuilder.NewArray())
	scoreBuilder.Release()

	if hasGroupBy {
		gbArr := pickGroupByValues(pool, inputs, groupByFieldName, chunkIdx, sources)
		collector.Set(groupByFieldName, chunkIdx, gbArr)
	}

	return sources
}

// mergeChunkString performs the k-way merge for one chunk with string PK.
func mergeChunkString(
	pool memory.Allocator,
	collector *chain.ChunkCollector,
	entries []*mergeEntry,
	chunkIdx int,
	topK int64,
	groupByOpts *GroupByOptions,
	hasGroupBy bool,
	groupByFieldName string,
	inputs []*chain.DataFrame,
) []SegmentSource {
	h := &mergeHeapString{}
	heap.Init(h)
	for _, e := range entries {
		heap.Push(h, e)
	}

	var ids []string
	var scores []float32
	var sources []SegmentSource

	if groupByOpts != nil && hasGroupBy {
		ids, scores, sources = mergeStringGroupBy(h, topK, groupByOpts.GroupSize)
	} else {
		ids, scores, sources = mergeStringStandard(h, topK)
	}

	idBuilder := array.NewStringBuilder(pool)
	idBuilder.AppendValues(ids, nil)
	collector.Set(idFieldName, chunkIdx, idBuilder.NewArray())
	idBuilder.Release()

	scoreBuilder := array.NewFloat32Builder(pool)
	scoreBuilder.AppendValues(scores, nil)
	collector.Set(scoreFieldName, chunkIdx, scoreBuilder.NewArray())
	scoreBuilder.Release()

	if hasGroupBy {
		gbArr := pickGroupByValues(pool, inputs, groupByFieldName, chunkIdx, sources)
		collector.Set(groupByFieldName, chunkIdx, gbArr)
	}

	return sources
}

// mergeInt64Standard performs standard k-way merge for one NQ (int64 PK).
// Corresponds to C++ ReduceHelper::ReduceSearchResultForOneNQ (Reduce.cpp:349-407).
func mergeInt64Standard(h *mergeHeapInt64, topK int64) ([]int64, []float32, []SegmentSource) {
	pkSet := make(map[int64]struct{}, topK)
	ids := make([]int64, 0, topK)
	scores := make([]float32, 0, topK)
	sources := make([]SegmentSource, 0, topK)

	for int64(len(ids)) < topK && h.Len() > 0 {
		e := heap.Pop(h).(*mergeEntry)
		pk := e.idInt64Val()

		if _, dup := pkSet[pk]; !dup {
			ids = append(ids, pk)
			scores = append(scores, e.scoreVal())
			pkSet[pk] = struct{}{}
			sources = append(sources, SegmentSource{
				InputIdx:    e.inputIdx,
				SegOffset:   e.segOffsetVal(),
				OriginalIdx: e.indices[e.cursor],
			})
		}

		if e.advance() {
			heap.Push(h, e)
		}
	}
	return ids, scores, sources
}

// mergeInt64GroupBy performs GroupBy-aware k-way merge for one NQ (int64 PK).
// Corresponds to C++ GroupReduceHelper::ReduceSearchResultForOneNQ (GroupReduce.cpp:122-211).
func mergeInt64GroupBy(h *mergeHeapInt64, topK int64, groupSize int64) ([]int64, []float32, []SegmentSource) {
	totalLimit := topK * groupSize
	pkSet := make(map[int64]struct{}, totalLimit)
	groupMap := make(map[string]int64)

	ids := make([]int64, 0, totalLimit)
	scores := make([]float32, 0, totalLimit)
	sources := make([]SegmentSource, 0, totalLimit)

	for int64(len(ids)) < totalLimit && h.Len() > 0 {
		e := heap.Pop(h).(*mergeEntry)
		pk := e.idInt64Val()

		if _, dup := pkSet[pk]; dup {
			if e.advance() {
				heap.Push(h, e)
			}
			continue
		}

		groupKey := extractGroupKey(e)
		currentGroupCount := groupMap[groupKey]

		// should_filtered logic from C++ GroupReduceHelper
		if int64(len(groupMap)) >= topK && currentGroupCount == 0 {
			if e.advance() {
				heap.Push(h, e)
			}
			continue
		}
		if currentGroupCount >= groupSize {
			if e.advance() {
				heap.Push(h, e)
			}
			continue
		}

		ids = append(ids, pk)
		scores = append(scores, e.scoreVal())
		pkSet[pk] = struct{}{}
		groupMap[groupKey]++
		sources = append(sources, SegmentSource{
			InputIdx:    e.inputIdx,
			SegOffset:   e.segOffsetVal(),
			OriginalIdx: e.indices[e.cursor],
		})

		if e.advance() {
			heap.Push(h, e)
		}
	}
	return ids, scores, sources
}

// mergeStringStandard performs standard k-way merge for one NQ (string PK).
func mergeStringStandard(h *mergeHeapString, topK int64) ([]string, []float32, []SegmentSource) {
	pkSet := make(map[string]struct{}, topK)
	ids := make([]string, 0, topK)
	scores := make([]float32, 0, topK)
	sources := make([]SegmentSource, 0, topK)

	for int64(len(ids)) < topK && h.Len() > 0 {
		e := heap.Pop(h).(*mergeEntry)
		pk := e.idStringVal()

		if _, dup := pkSet[pk]; !dup {
			ids = append(ids, pk)
			scores = append(scores, e.scoreVal())
			pkSet[pk] = struct{}{}
			sources = append(sources, SegmentSource{
				InputIdx:    e.inputIdx,
				SegOffset:   e.segOffsetVal(),
				OriginalIdx: e.indices[e.cursor],
			})
		}

		if e.advance() {
			heap.Push(h, e)
		}
	}
	return ids, scores, sources
}

// mergeStringGroupBy performs GroupBy-aware merge for one NQ (string PK).
func mergeStringGroupBy(h *mergeHeapString, topK int64, groupSize int64) ([]string, []float32, []SegmentSource) {
	totalLimit := topK * groupSize
	pkSet := make(map[string]struct{}, totalLimit)
	groupMap := make(map[string]int64)

	ids := make([]string, 0, totalLimit)
	scores := make([]float32, 0, totalLimit)
	sources := make([]SegmentSource, 0, totalLimit)

	for int64(len(ids)) < totalLimit && h.Len() > 0 {
		e := heap.Pop(h).(*mergeEntry)
		pk := e.idStringVal()

		if _, dup := pkSet[pk]; dup {
			if e.advance() {
				heap.Push(h, e)
			}
			continue
		}

		groupKey := extractGroupKey(e)
		currentGroupCount := groupMap[groupKey]

		if int64(len(groupMap)) >= topK && currentGroupCount == 0 {
			if e.advance() {
				heap.Push(h, e)
			}
			continue
		}
		if currentGroupCount >= groupSize {
			if e.advance() {
				heap.Push(h, e)
			}
			continue
		}

		ids = append(ids, pk)
		scores = append(scores, e.scoreVal())
		pkSet[pk] = struct{}{}
		groupMap[groupKey]++
		sources = append(sources, SegmentSource{
			InputIdx:    e.inputIdx,
			SegOffset:   e.segOffsetVal(),
			OriginalIdx: e.indices[e.cursor],
		})

		if e.advance() {
			heap.Push(h, e)
		}
	}
	return ids, scores, sources
}

// extractGroupKey extracts the group-by value as a string key for groupMap.
func extractGroupKey(e *mergeEntry) string {
	if e.groupByArr == nil {
		return ""
	}
	idx := e.indices[e.cursor]
	if e.groupByArr.IsNull(idx) {
		return "<null>"
	}
	switch arr := e.groupByArr.(type) {
	case *array.Int8:
		return fmt.Sprintf("%d", arr.Value(idx))
	case *array.Int16:
		return fmt.Sprintf("%d", arr.Value(idx))
	case *array.Int32:
		return fmt.Sprintf("%d", arr.Value(idx))
	case *array.Int64:
		return fmt.Sprintf("%d", arr.Value(idx))
	case *array.Boolean:
		if arr.Value(idx) {
			return "true"
		}
		return "false"
	case *array.String:
		return arr.Value(idx)
	default:
		return fmt.Sprintf("<%d>", idx)
	}
}

// pickGroupByValues builds a group-by output array by picking values from source entries.
// Uses SegmentSource.OriginalIdx to look up values from the original input chunk arrays.
func pickGroupByValues(
	pool memory.Allocator,
	inputs []*chain.DataFrame,
	groupByFieldName string,
	chunkIdx int,
	sources []SegmentSource,
) arrow.Array {
	// Preload chunk arrays per input
	chunkArrays := make([]arrow.Array, len(inputs))
	for i, df := range inputs {
		col := df.Column(groupByFieldName)
		if col != nil && chunkIdx < col.Len() {
			chunkArrays[i] = col.Chunk(chunkIdx)
		}
	}

	if len(sources) == 0 {
		// Empty: determine type and return empty array
		return buildEmptyGroupByArray(pool, inputs, groupByFieldName)
	}

	// Determine type from first valid source
	firstArr := chunkArrays[sources[0].InputIdx]
	if firstArr == nil {
		return buildEmptyGroupByArray(pool, inputs, groupByFieldName)
	}

	switch firstArr.(type) {
	case *array.Int8:
		return pickTyped(pool, chunkArrays, sources, func(arr arrow.Array, idx int) int8 {
			return arr.(*array.Int8).Value(idx)
		}, array.NewInt8Builder)
	case *array.Int16:
		return pickTyped(pool, chunkArrays, sources, func(arr arrow.Array, idx int) int16 {
			return arr.(*array.Int16).Value(idx)
		}, array.NewInt16Builder)
	case *array.Int32:
		return pickTyped(pool, chunkArrays, sources, func(arr arrow.Array, idx int) int32 {
			return arr.(*array.Int32).Value(idx)
		}, array.NewInt32Builder)
	case *array.Int64:
		return pickTyped(pool, chunkArrays, sources, func(arr arrow.Array, idx int) int64 {
			return arr.(*array.Int64).Value(idx)
		}, array.NewInt64Builder)
	case *array.Boolean:
		return pickTyped(pool, chunkArrays, sources, func(arr arrow.Array, idx int) bool {
			return arr.(*array.Boolean).Value(idx)
		}, array.NewBooleanBuilder)
	case *array.String:
		return pickTyped(pool, chunkArrays, sources, func(arr arrow.Array, idx int) string {
			return arr.(*array.String).Value(idx)
		}, array.NewStringBuilder)
	default:
		return buildEmptyGroupByArray(pool, inputs, groupByFieldName)
	}
}

type appendable[T any] interface {
	Append(T)
	AppendNull()
	NewArray() arrow.Array
	Release()
}

func pickTyped[T any, B appendable[T]](
	pool memory.Allocator,
	chunkArrays []arrow.Array,
	sources []SegmentSource,
	getValue func(arrow.Array, int) T,
	newBuilder func(memory.Allocator) B,
) arrow.Array {
	b := newBuilder(pool)
	defer b.Release()
	for _, src := range sources {
		arr := chunkArrays[src.InputIdx]
		if arr == nil || arr.IsNull(src.OriginalIdx) {
			b.AppendNull()
		} else {
			b.Append(getValue(arr, src.OriginalIdx))
		}
	}
	return b.NewArray()
}

func buildEmptyGroupByArray(pool memory.Allocator, inputs []*chain.DataFrame, groupByFieldName string) arrow.Array {
	// Try to determine the type
	for _, df := range inputs {
		col := df.Column(groupByFieldName)
		if col == nil {
			continue
		}
		switch col.DataType().ID() {
		case arrow.INT8:
			b := array.NewInt8Builder(pool)
			a := b.NewArray()
			b.Release()
			return a
		case arrow.INT16:
			b := array.NewInt16Builder(pool)
			a := b.NewArray()
			b.Release()
			return a
		case arrow.INT32:
			b := array.NewInt32Builder(pool)
			a := b.NewArray()
			b.Release()
			return a
		case arrow.INT64:
			b := array.NewInt64Builder(pool)
			a := b.NewArray()
			b.Release()
			return a
		case arrow.BOOL:
			b := array.NewBooleanBuilder(pool)
			a := b.NewArray()
			b.Release()
			return a
		case arrow.STRING:
			b := array.NewStringBuilder(pool)
			a := b.NewArray()
			b.Release()
			return a
		}
	}
	// fallback
	b := array.NewInt64Builder(pool)
	a := b.NewArray()
	b.Release()
	return a
}
