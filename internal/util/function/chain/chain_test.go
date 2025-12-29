/*
 * # Licensed to the LF AI & Data foundation under one
 * # or more contributor license agreements. See the NOTICE file
 * # distributed with this work for additional information
 * # regarding copyright ownership. The ASF licenses this file
 * # to you under the Apache License, Version 2.0 (the
 * # "License"); you may not use this file except in compliance
 * # with the License. You may obtain a copy of the License at
 * #
 * #     http://www.apache.org/licenses/LICENSE-2.0
 * #
 * # Unless required by applicable law or agreed to in writing, software
 * # distributed under the License is distributed on an "AS IS" BASIS,
 * # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * # See the License for the specific language governing permissions and
 * # limitations under the License.
 */

package chain

import (
	"fmt"
	"testing"

	"github.com/apache/arrow/go/v17/arrow"
	"github.com/apache/arrow/go/v17/arrow/array"
	"github.com/apache/arrow/go/v17/arrow/memory"
	"github.com/stretchr/testify/suite"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
)

// =============================================================================
// Test Suite
// =============================================================================

type ChainTestSuite struct {
	suite.Suite
	pool *memory.CheckedAllocator
}

func (s *ChainTestSuite) SetupTest() {
	s.pool = memory.NewCheckedAllocator(memory.NewGoAllocator())
}

func (s *ChainTestSuite) TearDownTest() {
	s.pool.AssertSize(s.T(), 0)
}

func TestChainTestSuite(t *testing.T) {
	suite.Run(t, new(ChainTestSuite))
}

// =============================================================================
// Helper Functions
// =============================================================================

func (s *ChainTestSuite) createTestDataFrame() *DataFrame {
	resultData := &schemapb.SearchResultData{
		NumQueries: 2,
		TopK:       5,
		Topks:      []int64{5, 4},
		Scores:     []float32{0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1},
		Ids: &schemapb.IDs{
			IdField: &schemapb.IDs_IntId{
				IntId: &schemapb.LongArray{
					Data: []int64{1, 2, 3, 4, 5, 6, 7, 8, 9},
				},
			},
		},
		FieldsData: []*schemapb.FieldData{
			{
				Type:      schemapb.DataType_Int64,
				FieldName: "age",
				FieldId:   100,
				Field: &schemapb.FieldData_Scalars{
					Scalars: &schemapb.ScalarField{
						Data: &schemapb.ScalarField_LongData{
							LongData: &schemapb.LongArray{
								Data: []int64{25, 30, 35, 40, 45, 50, 55, 60, 65},
							},
						},
					},
				},
			},
			{
				Type:      schemapb.DataType_VarChar,
				FieldName: "name",
				FieldId:   101,
				Field: &schemapb.FieldData_Scalars{
					Scalars: &schemapb.ScalarField{
						Data: &schemapb.ScalarField_StringData{
							StringData: &schemapb.StringArray{
								Data: []string{"alice", "bob", "charlie", "david", "eve", "frank", "grace", "henry", "ivy"},
							},
						},
					},
				},
			},
		},
	}

	df, err := FromSearchResultData(resultData)
	s.Require().NoError(err)
	return df
}

// =============================================================================
// FuncChain Basic Tests
// =============================================================================

func (s *ChainTestSuite) TestNewFuncChain() {
	fc := NewFuncChain()
	s.NotNil(fc)
	s.Empty(fc.operators)
	s.NotNil(fc.alloc)
}

func (s *ChainTestSuite) TestNewFuncChainWithAllocator() {
	fc := NewFuncChainWithAllocator(s.pool)
	s.NotNil(fc)
	s.Equal(s.pool, fc.alloc)
}

func (s *ChainTestSuite) TestFuncChainSetName() {
	fc := NewFuncChain().SetName("test-chain")
	s.Equal("test-chain", fc.name)
}

func (s *ChainTestSuite) TestFuncChainString() {
	fc := NewFuncChain().SetName("test-chain")
	str := fc.String()
	s.Contains(str, "FuncChain: test-chain")
}

// =============================================================================
// SelectOp Tests
// =============================================================================

func (s *ChainTestSuite) TestSelectOp() {
	df := s.createTestDataFrame()
	defer df.Release()

	result, err := NewFuncChainWithAllocator(s.pool).
		Select(IDFieldName, "age").
		Execute(df)
	s.Require().NoError(err)
	defer result.Release()

	// Verify only selected columns exist
	s.True(result.HasColumn(IDFieldName))
	s.True(result.HasColumn("age"))
	s.False(result.HasColumn("name"))
	s.False(result.HasColumn(ScoreFieldName))

	// Verify data integrity
	s.Equal(df.NumRows(), result.NumRows())
	s.Equal(df.NumChunks(), result.NumChunks())
}

func (s *ChainTestSuite) TestSelectOp_NonExistentColumn() {
	df := s.createTestDataFrame()
	defer df.Release()

	_, err := NewFuncChainWithAllocator(s.pool).
		Select("nonexistent").
		Execute(df)
	s.Error(err)
}

// =============================================================================
// FilterOp Tests
// =============================================================================

// MockFilterFunction creates a boolean column for filtering
type MockFilterFunction struct {
	threshold float32
}

func (f *MockFilterFunction) Name() string { return "MockFilter" }

func (f *MockFilterFunction) OutputDataTypes() []arrow.DataType {
	return []arrow.DataType{arrow.FixedWidthTypes.Boolean}
}

func (f *MockFilterFunction) IsRunnable(stage string) bool { return true }

func (f *MockFilterFunction) Execute(ctx *FuncContext, inputs []*arrow.Chunked) ([]*arrow.Chunked, error) {
	col := inputs[0]

	chunks := make([]arrow.Array, len(col.Chunks()))
	for i, chunk := range col.Chunks() {
		floatChunk := chunk.(*array.Float32)
		builder := array.NewBooleanBuilder(ctx.Pool())

		for j := range floatChunk.Len() {
			if floatChunk.IsNull(j) {
				builder.AppendNull()
			} else {
				builder.Append(floatChunk.Value(j) >= f.threshold)
			}
		}

		chunks[i] = builder.NewArray()
		builder.Release()
	}

	result := arrow.NewChunked(arrow.FixedWidthTypes.Boolean, chunks)
	// Release individual arrays after creating chunked
	for _, chunk := range chunks {
		chunk.Release()
	}

	return []*arrow.Chunked{result}, nil
}

func (s *ChainTestSuite) TestFilterOp() {
	df := s.createTestDataFrame()
	defer df.Release()

	// Add filter column and then filter
	// Column mapping is now at operator level
	result, err := NewFuncChainWithAllocator(s.pool).
		Map(&MockFilterFunction{threshold: 0.5}, []string{ScoreFieldName}, []string{"_filter"}).
		Filter("_filter").
		Execute(df)
	s.Require().NoError(err)
	defer result.Release()

	// Verify filtered results - scores >= 0.5 should remain
	// Chunk 0: 0.9, 0.8, 0.7, 0.6, 0.5 (all 5 pass)
	// Chunk 1: 0.4, 0.3, 0.2, 0.1 (none pass)
	s.Equal(int64(5), result.NumRows())
	s.Equal([]int64{5, 0}, result.ChunkSizes())
}

func (s *ChainTestSuite) TestFilterOp_NonExistentColumn() {
	df := s.createTestDataFrame()
	defer df.Release()

	_, err := NewFuncChainWithAllocator(s.pool).
		Filter("nonexistent").
		Execute(df)
	s.Error(err)
}

// =============================================================================
// SortOp Tests
// =============================================================================

func (s *ChainTestSuite) TestSortOp_Ascending() {
	df := s.createTestDataFrame()
	defer df.Release()

	result, err := NewFuncChainWithAllocator(s.pool).
		Sort("age", false). // ascending
		Execute(df)
	s.Require().NoError(err)
	defer result.Release()

	// Verify sorted - each chunk should be sorted independently
	ageCol, _ := result.Column("age")

	// Check chunk 0 is sorted ascending
	chunk0 := ageCol.Chunk(0).(*array.Int64)
	for i := 1; i < chunk0.Len(); i++ {
		s.LessOrEqual(chunk0.Value(i-1), chunk0.Value(i))
	}

	// Check chunk 1 is sorted ascending
	if len(ageCol.Chunks()) > 1 {
		chunk1 := ageCol.Chunk(1).(*array.Int64)
		for i := 1; i < chunk1.Len(); i++ {
			s.LessOrEqual(chunk1.Value(i-1), chunk1.Value(i))
		}
	}
}

func (s *ChainTestSuite) TestSortOp_Descending() {
	df := s.createTestDataFrame()
	defer df.Release()

	result, err := NewFuncChainWithAllocator(s.pool).
		Sort("age", true). // descending
		Execute(df)
	s.Require().NoError(err)
	defer result.Release()

	// Verify sorted descending
	ageCol, _ := result.Column("age")

	// Check chunk 0 is sorted descending
	chunk0 := ageCol.Chunk(0).(*array.Int64)
	for i := 1; i < chunk0.Len(); i++ {
		s.GreaterOrEqual(chunk0.Value(i-1), chunk0.Value(i))
	}
}

func (s *ChainTestSuite) TestSortOp_NonExistentColumn() {
	df := s.createTestDataFrame()
	defer df.Release()

	_, err := NewFuncChainWithAllocator(s.pool).
		Sort("nonexistent", false).
		Execute(df)
	s.Error(err)
}

// =============================================================================
// LimitOp Tests
// =============================================================================

func (s *ChainTestSuite) TestLimitOp() {
	df := s.createTestDataFrame()
	defer df.Release()

	result, err := NewFuncChainWithAllocator(s.pool).
		Limit(3).
		Execute(df)
	s.Require().NoError(err)
	defer result.Release()

	// Each chunk should be limited to 3 rows
	// Chunk 0: 5 -> 3
	// Chunk 1: 4 -> 3
	s.Equal([]int64{3, 3}, result.ChunkSizes())
	s.Equal(int64(6), result.NumRows())
}

func (s *ChainTestSuite) TestLimitOp_WithOffset() {
	df := s.createTestDataFrame()
	defer df.Release()

	result, err := NewFuncChainWithAllocator(s.pool).
		LimitWithOffset(2, 1).
		Execute(df)
	s.Require().NoError(err)
	defer result.Release()

	// Offset 1, Limit 2 for each chunk
	// Chunk 0: 5 rows, skip 1, take 2 -> 2 rows
	// Chunk 1: 4 rows, skip 1, take 2 -> 2 rows
	s.Equal([]int64{2, 2}, result.ChunkSizes())
	s.Equal(int64(4), result.NumRows())
}

func (s *ChainTestSuite) TestLimitOp_LargerThanChunk() {
	df := s.createTestDataFrame()
	defer df.Release()

	result, err := NewFuncChainWithAllocator(s.pool).
		Limit(100).
		Execute(df)
	s.Require().NoError(err)
	defer result.Release()

	// Limit larger than chunk size should return all rows
	s.Equal(df.ChunkSizes(), result.ChunkSizes())
	s.Equal(df.NumRows(), result.NumRows())
}

func (s *ChainTestSuite) TestLimitOp_OffsetBeyondChunk() {
	df := s.createTestDataFrame()
	defer df.Release()

	result, err := NewFuncChainWithAllocator(s.pool).
		LimitWithOffset(10, 100).
		Execute(df)
	s.Require().NoError(err)
	defer result.Release()

	// Offset beyond chunk size should return empty chunks
	s.Equal([]int64{0, 0}, result.ChunkSizes())
	s.Equal(int64(0), result.NumRows())
}

// =============================================================================
// MapOp Tests
// =============================================================================

// MockAddColumnFunction adds a constant column
// Note: This function needs to know the chunk sizes, so it uses a special approach
// by taking $id column as input to determine the chunk structure
type MockAddColumnFunction struct {
	value int64
}

func (f *MockAddColumnFunction) Name() string { return "MockAddColumn" }

func (f *MockAddColumnFunction) OutputDataTypes() []arrow.DataType {
	return []arrow.DataType{arrow.PrimitiveTypes.Int64}
}

func (f *MockAddColumnFunction) IsRunnable(stage string) bool { return true }

func (f *MockAddColumnFunction) Execute(ctx *FuncContext, inputs []*arrow.Chunked) ([]*arrow.Chunked, error) {
	// Use input column to determine chunk sizes
	idCol := inputs[0]

	chunks := make([]arrow.Array, len(idCol.Chunks()))
	for i, chunk := range idCol.Chunks() {
		builder := array.NewInt64Builder(ctx.Pool())
		for range chunk.Len() {
			builder.Append(f.value)
		}
		chunks[i] = builder.NewArray()
		builder.Release()
	}

	result := arrow.NewChunked(arrow.PrimitiveTypes.Int64, chunks)
	// Release individual arrays after creating chunked
	for _, chunk := range chunks {
		chunk.Release()
	}

	return []*arrow.Chunked{result}, nil
}

func (s *ChainTestSuite) TestMapOp() {
	df := s.createTestDataFrame()
	defer df.Release()

	// Column mapping is now at operator level
	result, err := NewFuncChainWithAllocator(s.pool).
		Map(&MockAddColumnFunction{value: 42}, []string{IDFieldName}, []string{"constant"}).
		Execute(df)
	s.Require().NoError(err)
	defer result.Release()

	// Verify new column exists
	s.True(result.HasColumn("constant"))

	// Verify all values are 42
	col, _ := result.Column("constant")
	for i := range len(col.Chunks()) {
		chunk := col.Chunk(i).(*array.Int64)
		for j := range chunk.Len() {
			s.Equal(int64(42), chunk.Value(j))
		}
	}
}

func (s *ChainTestSuite) TestMapOp_NilFunction() {
	df := s.createTestDataFrame()
	defer df.Release()

	// With new Map signature, nil function should error in NewMapOp
	_, err := NewFuncChainWithAllocator(s.pool).
		Map(nil, []string{}, []string{}).
		Execute(df)
	s.Error(err)
}

// =============================================================================
// Chained Operations Tests
// =============================================================================

func (s *ChainTestSuite) TestChainedOperations() {
	df := s.createTestDataFrame()
	defer df.Release()

	// Add filter column -> Filter -> Select -> Sort -> Limit
	// Column mapping is now at operator level
	result, err := NewFuncChainWithAllocator(s.pool).
		Map(&MockFilterFunction{threshold: 0.3}, []string{ScoreFieldName}, []string{"_filter"}).
		Filter("_filter").
		Select(IDFieldName, ScoreFieldName, "age").
		Sort(ScoreFieldName, true). // descending
		Limit(3).
		Execute(df)
	s.Require().NoError(err)
	defer result.Release()

	// Verify columns
	s.True(result.HasColumn(IDFieldName))
	s.True(result.HasColumn(ScoreFieldName))
	s.True(result.HasColumn("age"))
	s.False(result.HasColumn("name"))
	s.False(result.HasColumn("_filter"))
}

// =============================================================================
// Memory Leak Tests
// =============================================================================

func (s *ChainTestSuite) TestMemoryLeak_ChainedOperations() {
	for range 10 {
		df := s.createTestDataFrame()

		result, err := NewFuncChainWithAllocator(s.pool).
			Map(&MockAddColumnFunction{value: 1}, []string{IDFieldName}, []string{"temp"}).
			Select(IDFieldName, "age", "temp").
			Limit(3).
			Execute(df)
		s.Require().NoError(err)

		result.Release()
		df.Release()
	}
	// Memory leak check happens in TearDownTest
}

func (s *ChainTestSuite) TestMemoryLeak_FilterOperation() {
	for range 10 {
		df := s.createTestDataFrame()

		result, err := NewFuncChainWithAllocator(s.pool).
			Map(&MockFilterFunction{threshold: 0.5}, []string{ScoreFieldName}, []string{"_f"}).
			Filter("_f").
			Execute(df)
		s.Require().NoError(err)

		result.Release()
		df.Release()
	}
	// Memory leak check happens in TearDownTest
}

func (s *ChainTestSuite) TestMemoryLeak_SortOperation() {
	for range 10 {
		df := s.createTestDataFrame()

		result, err := NewFuncChainWithAllocator(s.pool).
			Sort("age", true).
			Execute(df)
		s.Require().NoError(err)

		result.Release()
		df.Release()
	}
	// Memory leak check happens in TearDownTest
}

// =============================================================================
// Operator String Tests
// =============================================================================

func (s *ChainTestSuite) TestOperatorStrings() {
	// MapOp
	mapOp := &MapOp{function: &MockAddColumnFunction{value: 1}}
	s.Contains(mapOp.String(), "Map(MockAddColumn)")

	mapOpNil := &MapOp{function: nil}
	s.Equal("Map(nil)", mapOpNil.String())

	// FilterOp
	filterOp := &FilterOp{column: "filter_col"}
	s.Equal("Filter(filter_col)", filterOp.String())

	// SelectOp
	selectOp := &SelectOp{columns: []string{"a", "b"}}
	s.Contains(selectOp.String(), "Select")

	// SortOp
	sortOpAsc := &SortOp{column: "col", desc: false}
	s.Equal("Sort(col ASC)", sortOpAsc.String())

	sortOpDesc := &SortOp{column: "col", desc: true}
	s.Equal("Sort(col DESC)", sortOpDesc.String())

	// LimitOp
	limitOp := &LimitOp{limit: 10}
	s.Equal("Limit(10)", limitOp.String())

	limitOpOffset := &LimitOp{limit: 10, offset: 5}
	s.Equal("Limit(10, offset=5)", limitOpOffset.String())
}

// =============================================================================
// FuncContext Tests
// =============================================================================

func (s *ChainTestSuite) TestNewFuncContext() {
	ctx := NewFuncContext(s.pool)
	s.Equal(s.pool, ctx.Pool())
}

func (s *ChainTestSuite) TestNewFuncContext_NilPool() {
	ctx := NewFuncContext(nil)
	s.NotNil(ctx.Pool())
	s.Equal(memory.DefaultAllocator, ctx.Pool())
}

// =============================================================================
// Edge Cases
// =============================================================================

func (s *ChainTestSuite) TestEmptyChain() {
	df := s.createTestDataFrame()
	defer df.Release()

	// Empty chain should return input as-is
	result, err := NewFuncChainWithAllocator(s.pool).Execute(df)
	s.Require().NoError(err)

	// Result should be the same as input
	s.Equal(df, result)
}

func (s *ChainTestSuite) TestSelectOp_AllColumns() {
	df := s.createTestDataFrame()
	defer df.Release()

	// Select all columns
	result, err := NewFuncChainWithAllocator(s.pool).
		Select(IDFieldName, ScoreFieldName, "age", "name").
		Execute(df)
	s.Require().NoError(err)
	defer result.Release()

	s.Equal(df.NumColumns(), result.NumColumns())
	s.Equal(df.NumRows(), result.NumRows())
}

func (s *ChainTestSuite) TestLimitOp_ZeroLimit() {
	df := s.createTestDataFrame()
	defer df.Release()

	result, err := NewFuncChainWithAllocator(s.pool).
		Limit(0).
		Execute(df)
	s.Require().NoError(err)
	defer result.Release()

	s.Equal(int64(0), result.NumRows())
}

// =============================================================================
// Validate Tests
// =============================================================================

func (s *ChainTestSuite) TestValidate_ValidChain() {
	fc := NewFuncChainWithAllocator(s.pool).
		Select(IDFieldName, "age").
		Sort("age", false).
		Limit(10)

	err := fc.Validate()
	s.NoError(err)
}

func (s *ChainTestSuite) TestValidate_NilMapFunction() {
	fc := NewFuncChainWithAllocator(s.pool).
		Map(nil, []string{}, []string{})

	err := fc.Validate()
	s.Error(err)
	s.Contains(err.Error(), "chain build error")
}

func (s *ChainTestSuite) TestValidate_BuildError() {
	fc := NewFuncChainWithAllocator(s.pool)
	// Force a build error by adding a map with nil function
	fc.Map(nil, []string{"a"}, []string{"b"})

	err := fc.Validate()
	s.Error(err)
}

// =============================================================================
// MapWithError Tests
// =============================================================================

func (s *ChainTestSuite) TestMapWithError_Success() {
	df := s.createTestDataFrame()
	defer df.Release()

	fc := NewFuncChainWithAllocator(s.pool)
	_, err := fc.MapWithError(&MockAddColumnFunction{value: 42}, []string{IDFieldName}, []string{"constant"})
	s.NoError(err)

	result, err := fc.Execute(df)
	s.Require().NoError(err)
	defer result.Release()

	s.True(result.HasColumn("constant"))
}

func (s *ChainTestSuite) TestMapWithError_NilFunction() {
	fc := NewFuncChainWithAllocator(s.pool)
	_, err := fc.MapWithError(nil, []string{IDFieldName}, []string{"out"})
	s.Error(err)
}

// =============================================================================
// ExecuteWithStage Tests
// =============================================================================

// MockStagedFunction is a function that only runs in certain stages
type MockStagedFunction struct {
	value  int64
	stages []string
}

func (f *MockStagedFunction) Name() string { return "MockStaged" }

func (f *MockStagedFunction) OutputDataTypes() []arrow.DataType {
	return []arrow.DataType{arrow.PrimitiveTypes.Int64}
}

func (f *MockStagedFunction) IsRunnable(stage string) bool {
	for _, s := range f.stages {
		if s == stage {
			return true
		}
	}
	return false
}

func (f *MockStagedFunction) Execute(ctx *FuncContext, inputs []*arrow.Chunked) ([]*arrow.Chunked, error) {
	idCol := inputs[0]
	chunks := make([]arrow.Array, len(idCol.Chunks()))
	for i, chunk := range idCol.Chunks() {
		builder := array.NewInt64Builder(ctx.Pool())
		for range chunk.Len() {
			builder.Append(f.value)
		}
		chunks[i] = builder.NewArray()
		builder.Release()
	}
	result := arrow.NewChunked(arrow.PrimitiveTypes.Int64, chunks)
	for _, chunk := range chunks {
		chunk.Release()
	}
	return []*arrow.Chunked{result}, nil
}

func (s *ChainTestSuite) TestExecuteWithStage_SkipOperator() {
	df := s.createTestDataFrame()
	defer df.Release()

	// Create a function that only runs in "L2_rerank" stage
	stagedFn := &MockStagedFunction{value: 999, stages: []string{ChainTypeL2Rerank}}

	result, err := NewFuncChainWithAllocator(s.pool).
		Map(stagedFn, []string{IDFieldName}, []string{"staged_col"}).
		ExecuteWithStage(df, ChainTypeL1Rerank) // Different stage, should skip

	s.Require().NoError(err)
	defer result.Release()

	// The staged column should NOT exist because the function was skipped
	s.False(result.HasColumn("staged_col"))
}

func (s *ChainTestSuite) TestExecuteWithStage_RunOperator() {
	df := s.createTestDataFrame()
	defer df.Release()

	// Create a function that only runs in "L2_rerank" stage
	stagedFn := &MockStagedFunction{value: 999, stages: []string{ChainTypeL2Rerank}}

	result, err := NewFuncChainWithAllocator(s.pool).
		Map(stagedFn, []string{IDFieldName}, []string{"staged_col"}).
		ExecuteWithStage(df, ChainTypeL2Rerank) // Same stage, should run

	s.Require().NoError(err)
	defer result.Release()

	// The staged column should exist
	s.True(result.HasColumn("staged_col"))
}

func (s *ChainTestSuite) TestExecuteWithStage_EmptyStage() {
	df := s.createTestDataFrame()
	defer df.Release()

	// Create a function that only runs in specific stages
	stagedFn := &MockStagedFunction{value: 999, stages: []string{ChainTypeL2Rerank}}

	// Empty stage means all operators run
	result, err := NewFuncChainWithAllocator(s.pool).
		Map(stagedFn, []string{IDFieldName}, []string{"staged_col"}).
		ExecuteWithStage(df, "")

	s.Require().NoError(err)
	defer result.Release()

	// Should run because empty stage means run all
	s.True(result.HasColumn("staged_col"))
}

// =============================================================================
// FilterOp Type Validation Tests
// =============================================================================

func (s *ChainTestSuite) TestFilterOp_NonBooleanColumn() {
	df := s.createTestDataFrame()
	defer df.Release()

	// Try to filter on a non-boolean column
	_, err := NewFuncChainWithAllocator(s.pool).
		Filter("age"). // age is Int64, not Boolean
		Execute(df)
	s.Error(err)
	s.Contains(err.Error(), "must be boolean type")
}

// =============================================================================
// FuncContext Stage Tests
// =============================================================================

func (s *ChainTestSuite) TestNewFuncContextWithStage() {
	ctx := NewFuncContextWithStage(s.pool, ChainTypeL2Rerank)
	s.Equal(s.pool, ctx.Pool())
	s.Equal(ChainTypeL2Rerank, ctx.Stage())
}

func (s *ChainTestSuite) TestFuncContextStage_Empty() {
	ctx := NewFuncContext(s.pool)
	s.Equal("", ctx.Stage())
}

// =============================================================================
// ProcessChunksParallel Tests
// =============================================================================

func (s *ChainTestSuite) TestProcessChunksParallel_Sequential() {
	processor := func(chunkIdx int) (arrow.Array, error) {
		builder := array.NewInt64Builder(s.pool)
		builder.Append(int64(chunkIdx))
		arr := builder.NewArray()
		builder.Release()
		return arr, nil
	}

	results, err := ProcessChunksParallel(3, processor, 0) // 0 means sequential
	s.Require().NoError(err)
	s.Len(results, 3)

	for i, arr := range results {
		s.Equal(int64(i), arr.(*array.Int64).Value(0))
		arr.Release()
	}
}

func (s *ChainTestSuite) TestProcessChunksParallel_Parallel() {
	processor := func(chunkIdx int) (arrow.Array, error) {
		builder := array.NewInt64Builder(s.pool)
		builder.Append(int64(chunkIdx * 10))
		arr := builder.NewArray()
		builder.Release()
		return arr, nil
	}

	results, err := ProcessChunksParallel(5, processor, 3) // 3 goroutines
	s.Require().NoError(err)
	s.Len(results, 5)

	for i, arr := range results {
		s.Equal(int64(i*10), arr.(*array.Int64).Value(0))
		arr.Release()
	}
}

func (s *ChainTestSuite) TestProcessChunksParallel_EmptyChunks() {
	processor := func(chunkIdx int) (arrow.Array, error) {
		return nil, nil
	}

	results, err := ProcessChunksParallel(0, processor, 4)
	s.Require().NoError(err)
	s.Empty(results)
}

func (s *ChainTestSuite) TestProcessChunksParallel_Error() {
	processor := func(chunkIdx int) (arrow.Array, error) {
		if chunkIdx == 2 {
			return nil, fmt.Errorf("error at chunk %d", chunkIdx)
		}
		builder := array.NewInt64Builder(s.pool)
		builder.Append(int64(chunkIdx))
		arr := builder.NewArray()
		builder.Release()
		return arr, nil
	}

	_, err := ProcessChunksParallel(5, processor, 0) // sequential to make error predictable
	s.Error(err)
	s.Contains(err.Error(), "error at chunk 2")
}

// =============================================================================
// Registry Tests
// =============================================================================

func (s *ChainTestSuite) TestFunctionRegistry() {
	registry := NewFunctionRegistry()
	s.NotNil(registry)

	// Test Register and Has
	factory := func(params map[string]interface{}) (FunctionExpr, error) {
		return &MockAddColumnFunction{value: 1}, nil
	}
	registry.Register("test_func", factory)
	s.True(registry.Has("test_func"))
	s.False(registry.Has("nonexistent"))

	// Test Get
	f, ok := registry.Get("test_func")
	s.True(ok)
	s.NotNil(f)

	_, ok = registry.Get("nonexistent")
	s.False(ok)

	// Test Create
	expr, err := registry.Create("test_func", nil)
	s.NoError(err)
	s.NotNil(expr)

	_, err = registry.Create("nonexistent", nil)
	s.Error(err)

	// Test Names
	names := registry.Names()
	s.Contains(names, "test_func")
}

func (s *ChainTestSuite) TestGlobalFunctionRegistry() {
	// Register a test function first
	RegisterFunction("test_global_func", func(params map[string]interface{}) (FunctionExpr, error) {
		return &MockAddColumnFunction{value: 42}, nil
	})

	// Test global registry functions
	s.True(HasFunction("test_global_func"))
	s.False(HasFunction("nonexistent_function"))

	// Test FunctionNames
	names := FunctionNames()
	s.Contains(names, "test_global_func")

	// Test GetFunctionFactory
	factory, ok := GetFunctionFactory("test_global_func")
	s.True(ok)
	s.NotNil(factory)

	_, ok = GetFunctionFactory("nonexistent")
	s.False(ok)

	// Test CreateFunction
	expr, err := CreateFunction("test_global_func", nil)
	s.NoError(err)
	s.NotNil(expr)

	_, err = CreateFunction("nonexistent", nil)
	s.Error(err)
}

// =============================================================================
// Operator Inputs/Outputs Tests
// =============================================================================

func (s *ChainTestSuite) TestOperatorInputsOutputs() {
	// BaseOp
	baseOp := &BaseOp{}
	s.Nil(baseOp.Inputs())
	s.Nil(baseOp.Outputs())

	// MapOp
	mapOp, _ := NewMapOp(&MockAddColumnFunction{value: 1}, []string{"a", "b"}, []string{"c"})
	s.Equal([]string{"a", "b"}, mapOp.Inputs())
	s.Equal([]string{"c"}, mapOp.Outputs())

	// FilterOp
	filterOp := &FilterOp{column: "filter_col"}
	s.Equal([]string{"filter_col"}, filterOp.Inputs())
	s.Empty(filterOp.Outputs())

	// SelectOp
	selectOp := &SelectOp{columns: []string{"a", "b", "c"}}
	s.Equal([]string{"a", "b", "c"}, selectOp.Inputs())
	s.Equal([]string{"a", "b", "c"}, selectOp.Outputs())

	// SortOp
	sortOp := &SortOp{column: "sort_col", desc: true}
	s.Equal([]string{"sort_col"}, sortOp.Inputs())
	s.Empty(sortOp.Outputs())

	// LimitOp
	limitOp := &LimitOp{limit: 10, offset: 5}
	s.Empty(limitOp.Inputs())
	s.Empty(limitOp.Outputs())
}

// =============================================================================
// compareArrayValues Tests
// =============================================================================

func (s *ChainTestSuite) TestCompareArrayValues_AllTypes() {
	// Test Int64
	int64Builder := array.NewInt64Builder(s.pool)
	int64Builder.AppendValues([]int64{10, 20, 10}, nil)
	int64Arr := int64Builder.NewArray()
	int64Builder.Release()
	defer int64Arr.Release()

	s.Equal(-1, compareArrayValues(int64Arr, 0, 1)) // 10 < 20
	s.Equal(1, compareArrayValues(int64Arr, 1, 0))  // 20 > 10
	s.Equal(0, compareArrayValues(int64Arr, 0, 2))  // 10 == 10

	// Test Float32
	float32Builder := array.NewFloat32Builder(s.pool)
	float32Builder.AppendValues([]float32{1.5, 2.5, 1.5}, nil)
	float32Arr := float32Builder.NewArray()
	float32Builder.Release()
	defer float32Arr.Release()

	s.Equal(-1, compareArrayValues(float32Arr, 0, 1))
	s.Equal(1, compareArrayValues(float32Arr, 1, 0))
	s.Equal(0, compareArrayValues(float32Arr, 0, 2))

	// Test Float64
	float64Builder := array.NewFloat64Builder(s.pool)
	float64Builder.AppendValues([]float64{1.5, 2.5}, nil)
	float64Arr := float64Builder.NewArray()
	float64Builder.Release()
	defer float64Arr.Release()

	s.Equal(-1, compareArrayValues(float64Arr, 0, 1))
	s.Equal(1, compareArrayValues(float64Arr, 1, 0))

	// Test String
	stringBuilder := array.NewStringBuilder(s.pool)
	stringBuilder.AppendValues([]string{"apple", "banana", "apple"}, nil)
	stringArr := stringBuilder.NewArray()
	stringBuilder.Release()
	defer stringArr.Release()

	s.Equal(-1, compareArrayValues(stringArr, 0, 1)) // "apple" < "banana"
	s.Equal(1, compareArrayValues(stringArr, 1, 0))
	s.Equal(0, compareArrayValues(stringArr, 0, 2))

	// Test Int8
	int8Builder := array.NewInt8Builder(s.pool)
	int8Builder.AppendValues([]int8{1, 2}, nil)
	int8Arr := int8Builder.NewArray()
	int8Builder.Release()
	defer int8Arr.Release()

	s.Equal(-1, compareArrayValues(int8Arr, 0, 1))

	// Test Int16
	int16Builder := array.NewInt16Builder(s.pool)
	int16Builder.AppendValues([]int16{100, 200}, nil)
	int16Arr := int16Builder.NewArray()
	int16Builder.Release()
	defer int16Arr.Release()

	s.Equal(-1, compareArrayValues(int16Arr, 0, 1))

	// Test Int32
	int32Builder := array.NewInt32Builder(s.pool)
	int32Builder.AppendValues([]int32{1000, 2000}, nil)
	int32Arr := int32Builder.NewArray()
	int32Builder.Release()
	defer int32Arr.Release()

	s.Equal(-1, compareArrayValues(int32Arr, 0, 1))

	// Test with nulls
	int64WithNullBuilder := array.NewInt64Builder(s.pool)
	int64WithNullBuilder.AppendNull()
	int64WithNullBuilder.Append(10)
	int64WithNullBuilder.AppendNull()
	int64WithNullArr := int64WithNullBuilder.NewArray()
	int64WithNullBuilder.Release()
	defer int64WithNullArr.Release()

	s.Equal(0, compareArrayValues(int64WithNullArr, 0, 2))  // null == null
	s.Equal(-1, compareArrayValues(int64WithNullArr, 0, 1)) // null < 10
	s.Equal(1, compareArrayValues(int64WithNullArr, 1, 0))  // 10 > null
}

// =============================================================================
// dispatchPickByIndices Tests
// =============================================================================

func (s *ChainTestSuite) TestDispatchPickByIndices_AllTypes() {
	indices := []int{2, 0, 1}

	// Test Int8
	int8Builder := array.NewInt8Builder(s.pool)
	int8Builder.AppendValues([]int8{10, 20, 30}, nil)
	int8Arr := int8Builder.NewArray()
	int8Builder.Release()
	defer int8Arr.Release()

	result, err := dispatchPickByIndices(s.pool, int8Arr, indices)
	s.Require().NoError(err)
	defer result.Release()
	s.Equal(int8(30), result.(*array.Int8).Value(0))

	// Test Int16
	int16Builder := array.NewInt16Builder(s.pool)
	int16Builder.AppendValues([]int16{100, 200, 300}, nil)
	int16Arr := int16Builder.NewArray()
	int16Builder.Release()
	defer int16Arr.Release()

	result, err = dispatchPickByIndices(s.pool, int16Arr, indices)
	s.Require().NoError(err)
	defer result.Release()
	s.Equal(int16(300), result.(*array.Int16).Value(0))

	// Test Int32
	int32Builder := array.NewInt32Builder(s.pool)
	int32Builder.AppendValues([]int32{1000, 2000, 3000}, nil)
	int32Arr := int32Builder.NewArray()
	int32Builder.Release()
	defer int32Arr.Release()

	result, err = dispatchPickByIndices(s.pool, int32Arr, indices)
	s.Require().NoError(err)
	defer result.Release()
	s.Equal(int32(3000), result.(*array.Int32).Value(0))

	// Test Float64
	float64Builder := array.NewFloat64Builder(s.pool)
	float64Builder.AppendValues([]float64{1.1, 2.2, 3.3}, nil)
	float64Arr := float64Builder.NewArray()
	float64Builder.Release()
	defer float64Arr.Release()

	result, err = dispatchPickByIndices(s.pool, float64Arr, indices)
	s.Require().NoError(err)
	defer result.Release()
	s.InDelta(3.3, result.(*array.Float64).Value(0), 0.001)

	// Test Boolean
	boolBuilder := array.NewBooleanBuilder(s.pool)
	boolBuilder.AppendValues([]bool{true, false, true}, nil)
	boolArr := boolBuilder.NewArray()
	boolBuilder.Release()
	defer boolArr.Release()

	result, err = dispatchPickByIndices(s.pool, boolArr, indices)
	s.Require().NoError(err)
	defer result.Release()
	s.True(result.(*array.Boolean).Value(0))
}

// =============================================================================
// SortOp with different types Tests
// =============================================================================

func (s *ChainTestSuite) TestSortOp_FloatColumn() {
	// Create DataFrame with float scores
	resultData := &schemapb.SearchResultData{
		NumQueries: 1,
		TopK:       3,
		Topks:      []int64{3},
		Scores:     []float32{0.5, 0.9, 0.1},
		Ids: &schemapb.IDs{
			IdField: &schemapb.IDs_IntId{
				IntId: &schemapb.LongArray{Data: []int64{1, 2, 3}},
			},
		},
	}

	df, err := FromSearchResultData(resultData)
	s.Require().NoError(err)
	defer df.Release()

	// Sort by score descending
	result, err := NewFuncChainWithAllocator(s.pool).
		Sort(ScoreFieldName, true).
		Execute(df)
	s.Require().NoError(err)
	defer result.Release()

	// Verify order: should be 0.9, 0.5, 0.1
	scoreCol, _ := result.Column(ScoreFieldName)
	chunk := scoreCol.Chunk(0).(*array.Float32)
	s.Equal(float32(0.9), chunk.Value(0))
	s.Equal(float32(0.5), chunk.Value(1))
	s.Equal(float32(0.1), chunk.Value(2))
}

func (s *ChainTestSuite) TestSortOp_StringColumn() {
	resultData := &schemapb.SearchResultData{
		NumQueries: 1,
		TopK:       3,
		Topks:      []int64{3},
		Ids: &schemapb.IDs{
			IdField: &schemapb.IDs_IntId{
				IntId: &schemapb.LongArray{Data: []int64{1, 2, 3}},
			},
		},
		FieldsData: []*schemapb.FieldData{
			{
				Type:      schemapb.DataType_VarChar,
				FieldName: "name",
				FieldId:   100,
				Field: &schemapb.FieldData_Scalars{
					Scalars: &schemapb.ScalarField{
						Data: &schemapb.ScalarField_StringData{
							StringData: &schemapb.StringArray{Data: []string{"charlie", "alice", "bob"}},
						},
					},
				},
			},
		},
	}

	df, err := FromSearchResultData(resultData)
	s.Require().NoError(err)
	defer df.Release()

	// Sort by name ascending
	result, err := NewFuncChainWithAllocator(s.pool).
		Sort("name", false).
		Execute(df)
	s.Require().NoError(err)
	defer result.Release()

	// Verify order: should be alice, bob, charlie
	nameCol, _ := result.Column("name")
	chunk := nameCol.Chunk(0).(*array.String)
	s.Equal("alice", chunk.Value(0))
	s.Equal("bob", chunk.Value(1))
	s.Equal("charlie", chunk.Value(2))
}

// =============================================================================
// MapOp Name Tests
// =============================================================================

func (s *ChainTestSuite) TestMapOp_Name() {
	mapOp, _ := NewMapOp(&MockAddColumnFunction{value: 1}, []string{"a"}, []string{"b"})
	s.Equal("Map", mapOp.Name())
}

// =============================================================================
// FuncChain String with operators
// =============================================================================

func (s *ChainTestSuite) TestFuncChain_StringWithOperators() {
	fc := NewFuncChain().
		SetName("test-chain").
		Select("a", "b").
		Filter("f").
		Sort("s", true).
		Limit(10)

	str := fc.String()
	s.Contains(str, "test-chain")
	s.Contains(str, "Select")
	s.Contains(str, "Filter")
	s.Contains(str, "Sort")
	s.Contains(str, "Limit")
}
