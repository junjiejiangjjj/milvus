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

	"github.com/apache/arrow/go/v17/arrow"
	"github.com/apache/arrow/go/v17/arrow/array"
	"github.com/apache/arrow/go/v17/arrow/memory"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func buildTestRecord(pool memory.Allocator, ids []int64, scores []float32, segOffsets []int64) arrow.Record {
	schema := arrow.NewSchema([]arrow.Field{
		{Name: idFieldName, Type: arrow.PrimitiveTypes.Int64},
		{Name: scoreFieldName, Type: arrow.PrimitiveTypes.Float32},
		{Name: SegOffsetCol, Type: arrow.PrimitiveTypes.Int64},
	}, nil)

	idBuilder := array.NewInt64Builder(pool)
	defer idBuilder.Release()
	idBuilder.AppendValues(ids, nil)
	idArr := idBuilder.NewArray()
	defer idArr.Release()

	scoreBuilder := array.NewFloat32Builder(pool)
	defer scoreBuilder.Release()
	scoreBuilder.AppendValues(scores, nil)
	scoreArr := scoreBuilder.NewArray()
	defer scoreArr.Release()

	offsetBuilder := array.NewInt64Builder(pool)
	defer offsetBuilder.Release()
	offsetBuilder.AppendValues(segOffsets, nil)
	offsetArr := offsetBuilder.NewArray()
	defer offsetArr.Release()

	return array.NewRecord(schema, []arrow.Array{idArr, scoreArr, offsetArr}, int64(len(ids)))
}

func TestDataFrameFromArrowRecord_Basic(t *testing.T) {
	pool := memory.NewGoAllocator()
	// 2 NQs: NQ0 has 3 rows, NQ1 has 2 rows
	record := buildTestRecord(pool,
		[]int64{1, 2, 3, 4, 5},
		[]float32{0.9, 0.8, 0.7, 0.6, 0.5},
		[]int64{10, 20, 30, 40, 50},
	)
	defer record.Release()

	df, err := DataFrameFromArrowRecord(record, []int64{3, 2})
	require.NoError(t, err)
	defer df.Release()

	assert.Equal(t, 2, df.NumChunks())
	assert.Equal(t, int64(5), df.NumRows())
	assert.Equal(t, []int64{3, 2}, df.ChunkSizes())

	// Verify $id column
	idCol := df.Column(idFieldName)
	require.NotNil(t, idCol)
	assert.Equal(t, 2, len(idCol.Chunks())) // 2 chunks

	// Chunk 0: ids [1, 2, 3]
	idChunk0 := idCol.Chunk(0).(*array.Int64)
	assert.Equal(t, 3, idChunk0.Len())
	assert.Equal(t, int64(1), idChunk0.Value(0))
	assert.Equal(t, int64(2), idChunk0.Value(1))
	assert.Equal(t, int64(3), idChunk0.Value(2))

	// Chunk 1: ids [4, 5]
	idChunk1 := idCol.Chunk(1).(*array.Int64)
	assert.Equal(t, 2, idChunk1.Len())
	assert.Equal(t, int64(4), idChunk1.Value(0))
	assert.Equal(t, int64(5), idChunk1.Value(1))

	// Verify $score column
	scoreCol := df.Column(scoreFieldName)
	require.NotNil(t, scoreCol)
	scoreChunk0 := scoreCol.Chunk(0).(*array.Float32)
	assert.InDelta(t, float32(0.9), scoreChunk0.Value(0), 0.001)

	// Verify $seg_offset column
	offsetCol := df.Column(SegOffsetCol)
	require.NotNil(t, offsetCol)
	offsetChunk1 := offsetCol.Chunk(1).(*array.Int64)
	assert.Equal(t, int64(40), offsetChunk1.Value(0))
}

func TestDataFrameFromArrowRecord_SingleNQ(t *testing.T) {
	pool := memory.NewGoAllocator()
	record := buildTestRecord(pool,
		[]int64{100},
		[]float32{0.95},
		[]int64{42},
	)
	defer record.Release()

	df, err := DataFrameFromArrowRecord(record, []int64{1})
	require.NoError(t, err)
	defer df.Release()

	assert.Equal(t, 1, df.NumChunks())
	assert.Equal(t, int64(1), df.NumRows())
}

func TestDataFrameFromArrowRecord_EmptyChunks(t *testing.T) {
	pool := memory.NewGoAllocator()
	// 3 NQs: NQ0=2, NQ1=0, NQ2=1
	record := buildTestRecord(pool,
		[]int64{1, 2, 3},
		[]float32{0.9, 0.8, 0.7},
		[]int64{10, 20, 30},
	)
	defer record.Release()

	df, err := DataFrameFromArrowRecord(record, []int64{2, 0, 1})
	require.NoError(t, err)
	defer df.Release()

	assert.Equal(t, 3, df.NumChunks())
	assert.Equal(t, int64(3), df.NumRows())

	idCol := df.Column(idFieldName)
	// Chunk 1 (NQ=1) should be empty
	assert.Equal(t, 0, idCol.Chunk(1).Len())
}

func TestDataFrameFromArrowRecord_MismatchRows(t *testing.T) {
	pool := memory.NewGoAllocator()
	record := buildTestRecord(pool,
		[]int64{1, 2, 3},
		[]float32{0.9, 0.8, 0.7},
		[]int64{10, 20, 30},
	)
	defer record.Release()

	// topkPerNQ sums to 5 but record has 3 rows
	_, err := DataFrameFromArrowRecord(record, []int64{3, 2})
	assert.Error(t, err)
}

func TestDataFrameFromArrowRecord_NilRecord(t *testing.T) {
	_, err := DataFrameFromArrowRecord(nil, []int64{1})
	assert.Error(t, err)
}
