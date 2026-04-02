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
	"testing"

	"github.com/apache/arrow/go/v17/arrow/array"
	"github.com/apache/arrow/go/v17/arrow/memory"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/internal/util/function/chain"
)

// buildTestDF creates a simple test DataFrame with int64 PKs.
// ids and scores are per-chunk (NQ) arrays.
func buildTestDF(pool memory.Allocator, idsPerChunk [][]int64, scoresPerChunk [][]float32) *chain.DataFrame {
	numChunks := len(idsPerChunk)
	chunkSizes := make([]int64, numChunks)

	builder := chain.NewDataFrameBuilder()

	idChunks := make([]interface{ Release() }, 0)
	_ = idChunks

	// Build ID chunks
	idArrays := make([]interface{}, numChunks)
	scoreArrays := make([]interface{}, numChunks)
	_ = idArrays
	_ = scoreArrays

	// Use ChunkCollector pattern
	colNames := []string{idFieldName, scoreFieldName}
	collector := chain.NewChunkCollector(colNames, numChunks)

	for i := 0; i < numChunks; i++ {
		chunkSizes[i] = int64(len(idsPerChunk[i]))

		idBuilder := array.NewInt64Builder(pool)
		idBuilder.AppendValues(idsPerChunk[i], nil)
		collector.Set(idFieldName, i, idBuilder.NewArray())
		idBuilder.Release()

		scoreBuilder := array.NewFloat32Builder(pool)
		scoreBuilder.AppendValues(scoresPerChunk[i], nil)
		collector.Set(scoreFieldName, i, scoreBuilder.NewArray())
		scoreBuilder.Release()
	}

	builder.SetChunkSizes(chunkSizes)
	_ = builder.AddColumnFromChunks(idFieldName, collector.Consume(idFieldName))
	_ = builder.AddColumnFromChunks(scoreFieldName, collector.Consume(scoreFieldName))
	collector.Release()

	return builder.Build()
}

// buildTestDFWithGroupBy creates a test DataFrame with int64 PKs and int64 group-by values.
func buildTestDFWithGroupBy(pool memory.Allocator, idsPerChunk [][]int64, scoresPerChunk [][]float32, groupByPerChunk [][]int64, groupByFieldName string) *chain.DataFrame {
	numChunks := len(idsPerChunk)
	chunkSizes := make([]int64, numChunks)

	colNames := []string{idFieldName, scoreFieldName, groupByFieldName}
	collector := chain.NewChunkCollector(colNames, numChunks)

	for i := 0; i < numChunks; i++ {
		chunkSizes[i] = int64(len(idsPerChunk[i]))

		idBuilder := array.NewInt64Builder(pool)
		idBuilder.AppendValues(idsPerChunk[i], nil)
		collector.Set(idFieldName, i, idBuilder.NewArray())
		idBuilder.Release()

		scoreBuilder := array.NewFloat32Builder(pool)
		scoreBuilder.AppendValues(scoresPerChunk[i], nil)
		collector.Set(scoreFieldName, i, scoreBuilder.NewArray())
		scoreBuilder.Release()

		gbBuilder := array.NewInt64Builder(pool)
		gbBuilder.AppendValues(groupByPerChunk[i], nil)
		collector.Set(groupByFieldName, i, gbBuilder.NewArray())
		gbBuilder.Release()
	}

	builder := chain.NewDataFrameBuilder()
	builder.SetChunkSizes(chunkSizes)
	_ = builder.AddColumnFromChunks(idFieldName, collector.Consume(idFieldName))
	_ = builder.AddColumnFromChunks(scoreFieldName, collector.Consume(scoreFieldName))
	_ = builder.AddColumnFromChunks(groupByFieldName, collector.Consume(groupByFieldName))
	collector.Release()

	return builder.Build()
}

func TestHeapMergeReduce_BasicMerge(t *testing.T) {
	pool := memory.NewGoAllocator()

	// Segment 0: NQ=1, results [id=1,score=0.9], [id=2,score=0.7], [id=3,score=0.5]
	df0 := buildTestDF(pool, [][]int64{{1, 2, 3}}, [][]float32{{0.9, 0.7, 0.5}})
	defer df0.Release()

	// Segment 1: NQ=1, results [id=4,score=0.8], [id=5,score=0.6], [id=6,score=0.4]
	df1 := buildTestDF(pool, [][]int64{{4, 5, 6}}, [][]float32{{0.8, 0.6, 0.4}})
	defer df1.Release()

	result, err := HeapMergeReduce(pool, []*chain.DataFrame{df0, df1}, 4, nil)
	require.NoError(t, err)
	defer result.DF.Release()

	// Expected merged order: id=1(0.9), id=4(0.8), id=2(0.7), id=5(0.6)
	assert.Equal(t, 1, result.DF.NumChunks())
	assert.Equal(t, int64(4), result.DF.NumRows())

	idCol := result.DF.Column(idFieldName)
	scoreCol := result.DF.Column(scoreFieldName)

	ids := idCol.Chunk(0).(*array.Int64)
	scores := scoreCol.Chunk(0).(*array.Float32)

	assert.Equal(t, int64(1), ids.Value(0))
	assert.Equal(t, int64(4), ids.Value(1))
	assert.Equal(t, int64(2), ids.Value(2))
	assert.Equal(t, int64(5), ids.Value(3))

	assert.InDelta(t, float32(0.9), scores.Value(0), 0.001)
	assert.InDelta(t, float32(0.8), scores.Value(1), 0.001)
	assert.InDelta(t, float32(0.7), scores.Value(2), 0.001)
	assert.InDelta(t, float32(0.6), scores.Value(3), 0.001)
}

func TestHeapMergeReduce_PKDedup(t *testing.T) {
	pool := memory.NewGoAllocator()

	// Same PK=1 in both segments, different scores
	df0 := buildTestDF(pool, [][]int64{{1, 2}}, [][]float32{{0.9, 0.7}})
	defer df0.Release()

	df1 := buildTestDF(pool, [][]int64{{1, 3}}, [][]float32{{0.8, 0.6}})
	defer df1.Release()

	result, err := HeapMergeReduce(pool, []*chain.DataFrame{df0, df1}, 3, nil)
	require.NoError(t, err)
	defer result.DF.Release()

	// id=1 from df0 (score=0.9) wins, id=1 from df1 (score=0.8) is deduped
	// Expected: id=1(0.9), id=2(0.7), id=3(0.6)
	assert.Equal(t, int64(3), result.DF.NumRows())

	ids := result.DF.Column(idFieldName).Chunk(0).(*array.Int64)
	assert.Equal(t, int64(1), ids.Value(0))
	assert.Equal(t, int64(2), ids.Value(1))
	assert.Equal(t, int64(3), ids.Value(2))
}

func TestHeapMergeReduce_EqualScoreDeterminism(t *testing.T) {
	pool := memory.NewGoAllocator()

	// Multiple entries with the same score, should be ordered by PK ASC
	df0 := buildTestDF(pool, [][]int64{{5, 3}}, [][]float32{{0.9, 0.9}})
	defer df0.Release()

	df1 := buildTestDF(pool, [][]int64{{1, 7}}, [][]float32{{0.9, 0.9}})
	defer df1.Release()

	result, err := HeapMergeReduce(pool, []*chain.DataFrame{df0, df1}, 4, nil)
	require.NoError(t, err)
	defer result.DF.Release()

	// Equal score → PK ASC: id=1, id=3, id=5, id=7
	ids := result.DF.Column(idFieldName).Chunk(0).(*array.Int64)
	assert.Equal(t, int64(1), ids.Value(0))
	assert.Equal(t, int64(3), ids.Value(1))
	assert.Equal(t, int64(5), ids.Value(2))
	assert.Equal(t, int64(7), ids.Value(3))
}

func TestHeapMergeReduce_MultiNQ(t *testing.T) {
	pool := memory.NewGoAllocator()

	// NQ=2: chunk0=[id=1,0.9 | id=2,0.7], chunk1=[id=10,0.8 | id=11,0.6]
	df0 := buildTestDF(pool,
		[][]int64{{1, 2}, {10, 11}},
		[][]float32{{0.9, 0.7}, {0.8, 0.6}})
	defer df0.Release()

	df1 := buildTestDF(pool,
		[][]int64{{3, 4}, {12, 13}},
		[][]float32{{0.85, 0.65}, {0.75, 0.55}})
	defer df1.Release()

	result, err := HeapMergeReduce(pool, []*chain.DataFrame{df0, df1}, 3, nil)
	require.NoError(t, err)
	defer result.DF.Release()

	assert.Equal(t, 2, result.DF.NumChunks())

	// Chunk 0: id=1(0.9), id=3(0.85), id=2(0.7)
	ids0 := result.DF.Column(idFieldName).Chunk(0).(*array.Int64)
	assert.Equal(t, int64(1), ids0.Value(0))
	assert.Equal(t, int64(3), ids0.Value(1))
	assert.Equal(t, int64(2), ids0.Value(2))

	// Chunk 1: id=10(0.8), id=12(0.75), id=11(0.6)
	ids1 := result.DF.Column(idFieldName).Chunk(1).(*array.Int64)
	assert.Equal(t, int64(10), ids1.Value(0))
	assert.Equal(t, int64(12), ids1.Value(1))
	assert.Equal(t, int64(11), ids1.Value(2))
}

func TestHeapMergeReduce_EmptyInput(t *testing.T) {
	pool := memory.NewGoAllocator()

	// One segment has results, one is empty
	df0 := buildTestDF(pool, [][]int64{{1, 2}}, [][]float32{{0.9, 0.7}})
	defer df0.Release()

	df1 := buildTestDF(pool, [][]int64{{}}, [][]float32{{}})
	defer df1.Release()

	result, err := HeapMergeReduce(pool, []*chain.DataFrame{df0, df1}, 3, nil)
	require.NoError(t, err)
	defer result.DF.Release()

	// Only results from df0
	assert.Equal(t, int64(2), result.DF.NumRows())
	ids := result.DF.Column(idFieldName).Chunk(0).(*array.Int64)
	assert.Equal(t, int64(1), ids.Value(0))
	assert.Equal(t, int64(2), ids.Value(1))
}

func TestHeapMergeReduce_SingleSegment(t *testing.T) {
	pool := memory.NewGoAllocator()

	df0 := buildTestDF(pool, [][]int64{{3, 1, 2}}, [][]float32{{0.5, 0.9, 0.7}})
	defer df0.Release()

	result, err := HeapMergeReduce(pool, []*chain.DataFrame{df0}, 2, nil)
	require.NoError(t, err)
	defer result.DF.Release()

	// Should sort by score DESC and take top 2
	assert.Equal(t, int64(2), result.DF.NumRows())
	ids := result.DF.Column(idFieldName).Chunk(0).(*array.Int64)
	assert.Equal(t, int64(1), ids.Value(0)) // score=0.9
	assert.Equal(t, int64(2), ids.Value(1)) // score=0.7
}

func TestHeapMergeReduce_GroupByBasic(t *testing.T) {
	pool := memory.NewGoAllocator()
	gbField := "color"

	// Segment 0: group A has 3 items, group B has 1
	df0 := buildTestDFWithGroupBy(pool,
		[][]int64{{1, 2, 3, 4}},
		[][]float32{{0.9, 0.8, 0.7, 0.6}},
		[][]int64{{100, 100, 100, 200}}, // group 100 and 200
		gbField)
	defer df0.Release()

	// Segment 1: group B has 2 items
	df1 := buildTestDFWithGroupBy(pool,
		[][]int64{{5, 6}},
		[][]float32{{0.85, 0.75}},
		[][]int64{{200, 200}},
		gbField)
	defer df1.Release()

	result, err := HeapMergeReduce(pool, []*chain.DataFrame{df0, df1}, 2, &GroupByOptions{
		GroupByFieldName: gbField,
		GroupSize:        2,
	})
	require.NoError(t, err)
	defer result.DF.Release()

	// topK=2, groupSize=2 → max 4 results
	// group 100: id=1(0.9), id=2(0.8) → 2 items (full)
	// group 200: id=5(0.85), id=4(0.6) or id=6(0.75) → 2 items (full)
	// Expected order by score: 1(0.9), 5(0.85), 2(0.8), 6(0.75)
	assert.True(t, result.DF.NumRows() <= 4)

	ids := result.DF.Column(idFieldName).Chunk(0).(*array.Int64)
	assert.Equal(t, int64(1), ids.Value(0))
	assert.Equal(t, int64(5), ids.Value(1))
	assert.Equal(t, int64(2), ids.Value(2))
	assert.Equal(t, int64(6), ids.Value(3))

	// Verify group_by column exists
	assert.True(t, result.DF.HasColumn(gbField))
}

func TestHeapMergeReduce_GroupByMaxGroups(t *testing.T) {
	pool := memory.NewGoAllocator()
	gbField := "category"

	// 4 groups, but topK=2 → only 2 groups allowed
	df0 := buildTestDFWithGroupBy(pool,
		[][]int64{{1, 2, 3, 4}},
		[][]float32{{0.9, 0.8, 0.7, 0.6}},
		[][]int64{{10, 20, 30, 40}},
		gbField)
	defer df0.Release()

	result, err := HeapMergeReduce(pool, []*chain.DataFrame{df0}, 2, &GroupByOptions{
		GroupByFieldName: gbField,
		GroupSize:        1,
	})
	require.NoError(t, err)
	defer result.DF.Release()

	// topK=2, groupSize=1 → max 2 results, 2 groups
	assert.Equal(t, int64(2), result.DF.NumRows())
	ids := result.DF.Column(idFieldName).Chunk(0).(*array.Int64)
	assert.Equal(t, int64(1), ids.Value(0)) // group 10
	assert.Equal(t, int64(2), ids.Value(1)) // group 20
}

func TestHeapMergeReduce_SourceTracking(t *testing.T) {
	pool := memory.NewGoAllocator()

	df0 := buildTestDF(pool, [][]int64{{1, 2}}, [][]float32{{0.9, 0.7}})
	defer df0.Release()

	df1 := buildTestDF(pool, [][]int64{{3}}, [][]float32{{0.8}})
	defer df1.Release()

	result, err := HeapMergeReduce(pool, []*chain.DataFrame{df0, df1}, 3, nil)
	require.NoError(t, err)
	defer result.DF.Release()

	// Verify source tracking
	require.Len(t, result.Sources, 1) // 1 chunk
	sources := result.Sources[0]
	require.Len(t, sources, 3)

	// id=1(0.9) from input 0
	assert.Equal(t, 0, sources[0].InputIdx)
	// id=3(0.8) from input 1
	assert.Equal(t, 1, sources[1].InputIdx)
	// id=2(0.7) from input 0
	assert.Equal(t, 0, sources[2].InputIdx)
}

func TestHeapMergeReduce_LargeScale(t *testing.T) {
	pool := memory.NewGoAllocator()

	// 4 segments, each with 100 results
	numSegments := 4
	numResults := 100
	dfs := make([]*chain.DataFrame, numSegments)
	for seg := 0; seg < numSegments; seg++ {
		ids := make([]int64, numResults)
		scores := make([]float32, numResults)
		for i := 0; i < numResults; i++ {
			ids[i] = int64(seg*1000 + i)
			scores[i] = float32(numResults-i) / float32(numResults) // descending scores
		}
		dfs[seg] = buildTestDF(pool, [][]int64{ids}, [][]float32{scores})
		defer dfs[seg].Release()
	}

	result, err := HeapMergeReduce(pool, dfs, 50, nil)
	require.NoError(t, err)
	defer result.DF.Release()

	assert.Equal(t, int64(50), result.DF.NumRows())

	// Verify scores are in descending order
	scoreCol := result.DF.Column(scoreFieldName).Chunk(0).(*array.Float32)
	for i := 1; i < 50; i++ {
		assert.GreaterOrEqual(t, scoreCol.Value(i-1), scoreCol.Value(i),
			"scores should be descending at index %d", i)
	}
}

func TestHeapMergeReduce_NoInputs(t *testing.T) {
	pool := memory.NewGoAllocator()
	_, err := HeapMergeReduce(pool, nil, 10, nil)
	assert.Error(t, err)
}

func TestHeapMergeReduce_MismatchedChunks(t *testing.T) {
	pool := memory.NewGoAllocator()

	df0 := buildTestDF(pool, [][]int64{{1}, {2}}, [][]float32{{0.9}, {0.8}})
	defer df0.Release()

	df1 := buildTestDF(pool, [][]int64{{3}}, [][]float32{{0.7}})
	defer df1.Release()

	_, err := HeapMergeReduce(pool, []*chain.DataFrame{df0, df1}, 10, nil)
	assert.Error(t, err)
}
