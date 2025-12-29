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
	"testing"

	"github.com/apache/arrow/go/v17/arrow"
	"github.com/apache/arrow/go/v17/arrow/array"
	"github.com/apache/arrow/go/v17/arrow/memory"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/suite"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
)

type DataFrameSuite struct {
	suite.Suite
	pool    *memory.CheckedAllocator
	rawPool *memory.GoAllocator
}

func (s *DataFrameSuite) SetupTest() {
	s.rawPool = memory.NewGoAllocator()
	s.pool = memory.NewCheckedAllocator(s.rawPool)
}

func (s *DataFrameSuite) TearDownTest() {
	// Check for memory leaks after each test
	s.pool.AssertSize(s.T(), 0)
}

// =============================================================================
// Type Mapping Tests
// =============================================================================

func (s *DataFrameSuite) TestToArrowType() {
	testCases := []struct {
		milvusType schemapb.DataType
		arrowType  arrow.DataType
		expectErr  bool
	}{
		{schemapb.DataType_Bool, arrow.FixedWidthTypes.Boolean, false},
		{schemapb.DataType_Int8, arrow.PrimitiveTypes.Int8, false},
		{schemapb.DataType_Int16, arrow.PrimitiveTypes.Int16, false},
		{schemapb.DataType_Int32, arrow.PrimitiveTypes.Int32, false},
		{schemapb.DataType_Int64, arrow.PrimitiveTypes.Int64, false},
		{schemapb.DataType_Float, arrow.PrimitiveTypes.Float32, false},
		{schemapb.DataType_Double, arrow.PrimitiveTypes.Float64, false},
		{schemapb.DataType_String, arrow.BinaryTypes.String, false},
		{schemapb.DataType_VarChar, arrow.BinaryTypes.String, false},
		{schemapb.DataType_Text, arrow.BinaryTypes.String, false},
		{schemapb.DataType_JSON, nil, true},        // Unsupported
		{schemapb.DataType_FloatVector, nil, true}, // Unsupported
	}

	for _, tc := range testCases {
		result, err := ToArrowType(tc.milvusType)
		if tc.expectErr {
			s.Error(err)
		} else {
			s.NoError(err)
			s.Equal(tc.arrowType.ID(), result.ID())
		}
	}
}

func (s *DataFrameSuite) TestToMilvusType() {
	testCases := []struct {
		arrowType  arrow.DataType
		milvusType schemapb.DataType
		expectErr  bool
	}{
		{arrow.FixedWidthTypes.Boolean, schemapb.DataType_Bool, false},
		{arrow.PrimitiveTypes.Int8, schemapb.DataType_Int8, false},
		{arrow.PrimitiveTypes.Int16, schemapb.DataType_Int16, false},
		{arrow.PrimitiveTypes.Int32, schemapb.DataType_Int32, false},
		{arrow.PrimitiveTypes.Int64, schemapb.DataType_Int64, false},
		{arrow.PrimitiveTypes.Float32, schemapb.DataType_Float, false},
		{arrow.PrimitiveTypes.Float64, schemapb.DataType_Double, false},
		{arrow.BinaryTypes.String, schemapb.DataType_VarChar, false},
		{arrow.BinaryTypes.Binary, schemapb.DataType_None, true}, // Unsupported
	}

	for _, tc := range testCases {
		result, err := ToMilvusType(tc.arrowType)
		if tc.expectErr {
			s.Error(err)
		} else {
			s.NoError(err)
			s.Equal(tc.milvusType, result)
		}
	}
}

// =============================================================================
// Import Tests
// =============================================================================

func (s *DataFrameSuite) TestFromSearchResultData_Basic() {
	// Create test data with NQ=2, Topks=[3, 2]
	resultData := &schemapb.SearchResultData{
		NumQueries: 2,
		TopK:       3,
		Topks:      []int64{3, 2},
		Scores:     []float32{0.9, 0.8, 0.7, 0.6, 0.5},
		Ids: &schemapb.IDs{
			IdField: &schemapb.IDs_IntId{
				IntId: &schemapb.LongArray{
					Data: []int64{1, 2, 3, 4, 5},
				},
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
							StringData: &schemapb.StringArray{
								Data: []string{"a", "b", "c", "d", "e"},
							},
						},
					},
				},
			},
			{
				Type:      schemapb.DataType_Int64,
				FieldName: "age",
				FieldId:   101,
				Field: &schemapb.FieldData_Scalars{
					Scalars: &schemapb.ScalarField{
						Data: &schemapb.ScalarField_LongData{
							LongData: &schemapb.LongArray{
								Data: []int64{20, 30, 40, 50, 60},
							},
						},
					},
				},
			},
		},
	}

	df, err := FromSearchResultData(resultData)
	s.Require().NoError(err)
	defer df.Release()

	// Verify basic properties
	s.Equal(2, df.NumChunks())
	s.Equal(int64(5), df.NumRows())
	s.Equal([]int64{3, 2}, df.ChunkSizes())

	// Verify columns exist
	s.True(df.HasColumn(IDFieldName))
	s.True(df.HasColumn(ScoreFieldName))
	s.True(df.HasColumn("name"))
	s.True(df.HasColumn("age"))

	// Verify column data using GetColumns
	cols, err := df.GetColumns([]string{IDFieldName, ScoreFieldName}, 0)
	s.Require().NoError(err)
	s.Equal(3, cols[0].Len()) // chunk 0 has 3 rows
	s.Equal(3, cols[1].Len())

	cols, err = df.GetColumns([]string{IDFieldName, ScoreFieldName}, 1)
	s.Require().NoError(err)
	s.Equal(2, cols[0].Len()) // chunk 1 has 2 rows
	s.Equal(2, cols[1].Len())
}

func (s *DataFrameSuite) TestFromSearchResultData_EmptyResult() {
	resultData := &schemapb.SearchResultData{
		NumQueries: 0,
		TopK:       0,
		Topks:      []int64{},
	}

	df, err := FromSearchResultData(resultData)
	s.Require().NoError(err)
	defer df.Release()

	s.Equal(0, df.NumChunks())
	s.Equal(int64(0), df.NumRows())
}

func (s *DataFrameSuite) TestFromSearchResultData_NilResult() {
	df, err := FromSearchResultData(nil)
	s.Error(err)
	s.Nil(df)
}

func (s *DataFrameSuite) TestFromSearchResultData_StringIDs() {
	resultData := &schemapb.SearchResultData{
		NumQueries: 1,
		TopK:       2,
		Topks:      []int64{2},
		Scores:     []float32{0.9, 0.8},
		Ids: &schemapb.IDs{
			IdField: &schemapb.IDs_StrId{
				StrId: &schemapb.StringArray{
					Data: []string{"id1", "id2"},
				},
			},
		},
	}

	df, err := FromSearchResultData(resultData)
	s.Require().NoError(err)
	defer df.Release()

	s.True(df.HasColumn(IDFieldName))
	idType, ok := df.GetFieldType(IDFieldName)
	s.True(ok)
	s.Equal(schemapb.DataType_VarChar, idType)
}

// =============================================================================
// Export Tests
// =============================================================================

func (s *DataFrameSuite) TestToSearchResultData() {
	// Create test data
	resultData := &schemapb.SearchResultData{
		NumQueries: 2,
		TopK:       3,
		Topks:      []int64{3, 2},
		Scores:     []float32{0.9, 0.8, 0.7, 0.6, 0.5},
		Ids: &schemapb.IDs{
			IdField: &schemapb.IDs_IntId{
				IntId: &schemapb.LongArray{
					Data: []int64{1, 2, 3, 4, 5},
				},
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
							StringData: &schemapb.StringArray{
								Data: []string{"a", "b", "c", "d", "e"},
							},
						},
					},
				},
			},
		},
	}

	df, err := FromSearchResultData(resultData)
	s.Require().NoError(err)
	defer df.Release()

	// Export back
	exported, err := df.ToSearchResultData()
	s.Require().NoError(err)

	// Verify exported data
	s.Equal(int64(2), exported.NumQueries)
	s.Equal(int64(3), exported.TopK)
	s.Equal([]int64{3, 2}, exported.Topks)
	s.Equal([]float32{0.9, 0.8, 0.7, 0.6, 0.5}, exported.Scores)
	s.Equal([]int64{1, 2, 3, 4, 5}, exported.Ids.GetIntId().GetData())
	s.Len(exported.FieldsData, 1)
	s.Equal("name", exported.FieldsData[0].FieldName)
	s.Equal([]string{"a", "b", "c", "d", "e"}, exported.FieldsData[0].GetScalars().GetStringData().GetData())
}

// =============================================================================
// Column Access Tests
// =============================================================================

func (s *DataFrameSuite) TestColumnAccess() {
	df := s.createTestDataFrame()
	defer df.Release()

	// Test Column
	col, err := df.Column("int_col")
	s.Require().NoError(err)
	s.NotNil(col)

	// Test Column - not found
	_, err = df.Column("nonexistent")
	s.Error(err)

	// Test HasColumn
	s.True(df.HasColumn("int_col"))
	s.False(df.HasColumn("nonexistent"))

	// Test GetFieldType
	dt, ok := df.GetFieldType("int_col")
	s.True(ok)
	s.Equal(schemapb.DataType_Int64, dt)

	// Test GetFieldID
	id, ok := df.GetFieldID("int_col")
	s.True(ok)
	s.Equal(int64(1), id)

	// Test ColumnNames
	names := df.ColumnNames()
	s.Len(names, 3) // $id, int_col, str_col
}

func (s *DataFrameSuite) TestGetColumns() {
	df := s.createTestDataFrame()
	defer df.Release()

	// Valid chunk
	cols, err := df.GetColumns([]string{"int_col", "str_col"}, 0)
	s.Require().NoError(err)
	s.Len(cols, 2)

	// Invalid chunk index
	_, err = df.GetColumns([]string{"int_col"}, 100)
	s.Error(err)

	// Invalid column name
	_, err = df.GetColumns([]string{"nonexistent"}, 0)
	s.Error(err)
}

// =============================================================================
// Column Operations Tests
// =============================================================================

func (s *DataFrameSuite) TestAddColumn() {
	df := s.createTestDataFrame()
	defer df.Release()

	initialColumns := df.NumColumns() // 3 columns: $id, int_col, str_col

	// Create new column data
	builder := array.NewFloat32Builder(s.pool)
	builder.AppendValues([]float32{1.1, 2.2, 3.3}, nil)
	arr1 := builder.NewArray()
	builder.AppendValues([]float32{4.4, 5.5}, nil)
	arr2 := builder.NewArray()
	builder.Release()
	defer arr1.Release()
	defer arr2.Release()

	newDF, err := df.AddColumn("float_col", []arrow.Array{arr1, arr2}, arrow.PrimitiveTypes.Float32)
	s.Require().NoError(err)
	defer newDF.Release()

	s.True(newDF.HasColumn("float_col"))
	s.Equal(initialColumns+1, newDF.NumColumns())

	// Original df should be unchanged
	s.False(df.HasColumn("float_col"))
	s.Equal(initialColumns, df.NumColumns())

	// Adding existing column should fail
	_, err = newDF.AddColumn("float_col", []arrow.Array{arr1, arr2}, arrow.PrimitiveTypes.Float32)
	s.Error(err)
}

func (s *DataFrameSuite) TestRemoveColumn() {
	df := s.createTestDataFrame()
	defer df.Release()

	initialColumns := df.NumColumns() // 3 columns: $id, int_col, str_col

	newDF, err := df.RemoveColumn("str_col")
	s.Require().NoError(err)
	defer newDF.Release()

	s.Equal(initialColumns-1, newDF.NumColumns())
	s.False(newDF.HasColumn("str_col"))
	s.True(newDF.HasColumn("int_col"))
	s.True(newDF.HasColumn(IDFieldName))

	// Original df should be unchanged
	s.True(df.HasColumn("str_col"))
	s.Equal(initialColumns, df.NumColumns())

	// Removing non-existent column should fail
	_, err = df.RemoveColumn("nonexistent")
	s.Error(err)
}

// =============================================================================
// Lifecycle Tests
// =============================================================================

func (s *DataFrameSuite) TestRetain() {
	df := s.createTestDataFrame()

	// Retain should not panic
	df.Retain()

	// Release twice (once for retain, once for original)
	df.Release()
	df.Release()
}

func (s *DataFrameSuite) TestClone() {
	df := s.createTestDataFrame()
	defer df.Release()

	// Clone the DataFrame
	cloned := df.Clone()
	defer cloned.Release()

	// Verify cloned DataFrame has same structure
	s.Equal(df.NumColumns(), cloned.NumColumns())
	s.Equal(df.NumRows(), cloned.NumRows())
	s.Equal(df.NumChunks(), cloned.NumChunks())
	s.Equal(df.ChunkSizes(), cloned.ChunkSizes())

	// Verify columns exist
	for _, name := range df.ColumnNames() {
		s.True(cloned.HasColumn(name))
	}

	// Verify data is the same
	col1, _ := df.Column("int_col")
	col2, _ := cloned.Column("int_col")
	s.Equal(col1.Len(), col2.Len())
}

func (s *DataFrameSuite) TestClone_MemoryLeak() {
	// Multiple clone cycles
	for range 10 {
		df := s.createTestDataFrame()
		cloned := df.Clone()

		// Both should be releasable independently
		cloned.Release()
		df.Release()
	}
	// Memory leak check happens in TearDownTest
}

// =============================================================================
// Memory Leak Tests
// =============================================================================

func (s *DataFrameSuite) TestMemoryLeak_ImportExport() {
	// This test verifies no memory leak during import/export cycle
	resultData := &schemapb.SearchResultData{
		NumQueries: 2,
		TopK:       3,
		Topks:      []int64{3, 2},
		Scores:     []float32{0.9, 0.8, 0.7, 0.6, 0.5},
		Ids: &schemapb.IDs{
			IdField: &schemapb.IDs_IntId{
				IntId: &schemapb.LongArray{
					Data: []int64{1, 2, 3, 4, 5},
				},
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
							StringData: &schemapb.StringArray{
								Data: []string{"a", "b", "c", "d", "e"},
							},
						},
					},
				},
			},
		},
	}

	// Multiple import/export cycles
	for range 10 {
		df, err := FromSearchResultData(resultData)
		s.Require().NoError(err)

		_, err = df.ToSearchResultData()
		s.Require().NoError(err)

		df.Release()
	}
	// Memory leak check happens in TearDownTest
}

func (s *DataFrameSuite) TestMemoryLeak_AddRemoveColumn() {
	df := s.createTestDataFrame()
	defer df.Release()

	// Multiple add/remove cycles
	for range 10 {
		builder := array.NewFloat32Builder(s.pool)
		builder.AppendValues([]float32{1.1, 2.2, 3.3}, nil)
		arr1 := builder.NewArray()
		builder.AppendValues([]float32{4.4, 5.5}, nil)
		arr2 := builder.NewArray()
		builder.Release()

		newDF, err := df.AddColumn("temp_col", []arrow.Array{arr1, arr2}, arrow.PrimitiveTypes.Float32)
		s.Require().NoError(err)

		arr1.Release()
		arr2.Release()

		removedDF, err := newDF.RemoveColumn("temp_col")
		s.Require().NoError(err)

		newDF.Release()
		removedDF.Release()
	}
	// Memory leak check happens in TearDownTest
}

func (s *DataFrameSuite) TestMemoryLeak_MultipleRetainRelease() {
	df := s.createTestDataFrame()

	// Multiple retain/release cycles
	for range 10 {
		df.Retain()
	}

	for range 10 {
		df.Release()
	}

	// Final release
	df.Release()
	// Memory leak check happens in TearDownTest
}

// =============================================================================
// Helper Functions
// =============================================================================

func (s *DataFrameSuite) createTestDataFrame() *DataFrame {
	resultData := &schemapb.SearchResultData{
		NumQueries: 2,
		TopK:       3,
		Topks:      []int64{3, 2},
		Ids: &schemapb.IDs{
			IdField: &schemapb.IDs_IntId{
				IntId: &schemapb.LongArray{
					Data: []int64{1, 2, 3, 4, 5},
				},
			},
		},
		FieldsData: []*schemapb.FieldData{
			{
				Type:      schemapb.DataType_Int64,
				FieldName: "int_col",
				FieldId:   1,
				Field: &schemapb.FieldData_Scalars{
					Scalars: &schemapb.ScalarField{
						Data: &schemapb.ScalarField_LongData{
							LongData: &schemapb.LongArray{
								Data: []int64{1, 2, 3, 4, 5},
							},
						},
					},
				},
			},
			{
				Type:      schemapb.DataType_VarChar,
				FieldName: "str_col",
				FieldId:   2,
				Field: &schemapb.FieldData_Scalars{
					Scalars: &schemapb.ScalarField{
						Data: &schemapb.ScalarField_StringData{
							StringData: &schemapb.StringArray{
								Data: []string{"a", "b", "c", "d", "e"},
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

func TestDataFrameSuite(t *testing.T) {
	suite.Run(t, new(DataFrameSuite))
}

// =============================================================================
// Additional Tests (non-suite based) with memory leak detection
// =============================================================================

func TestNewDataFrame(t *testing.T) {
	df := NewDataFrame()
	assert.NotNil(t, df)
	assert.Equal(t, 0, df.NumChunks())
	assert.Equal(t, int64(0), df.NumRows())
	assert.Equal(t, 0, df.NumColumns())
	df.Release()
}

func TestDataFrameRelease(t *testing.T) {
	df := NewDataFrame()
	// Should not panic
	df.Release()
	assert.Nil(t, df.columns)
	assert.Nil(t, df.schema)
}

func TestSchema(t *testing.T) {
	pool := memory.NewCheckedAllocator(memory.NewGoAllocator())
	defer pool.AssertSize(t, 0)

	resultData := &schemapb.SearchResultData{
		NumQueries: 1,
		TopK:       2,
		Topks:      []int64{2},
		Scores:     []float32{0.9, 0.8},
		Ids: &schemapb.IDs{
			IdField: &schemapb.IDs_IntId{
				IntId: &schemapb.LongArray{
					Data: []int64{1, 2},
				},
			},
		},
	}

	df, err := FromSearchResultData(resultData)
	assert.NoError(t, err)
	defer df.Release()

	schema := df.Schema()
	assert.NotNil(t, schema)
	assert.Equal(t, 2, schema.NumFields()) // $id and $score
}

func TestSetAndGetFieldType(t *testing.T) {
	df := NewDataFrame()
	defer df.Release()

	df.SetFieldType("test_field", schemapb.DataType_Int64)

	dt, ok := df.GetFieldType("test_field")
	assert.True(t, ok)
	assert.Equal(t, schemapb.DataType_Int64, dt)

	_, ok = df.GetFieldType("nonexistent")
	assert.False(t, ok)
}

func TestSetAndGetFieldID(t *testing.T) {
	df := NewDataFrame()
	defer df.Release()

	df.SetFieldID("test_field", 123)

	id, ok := df.GetFieldID("test_field")
	assert.True(t, ok)
	assert.Equal(t, int64(123), id)

	_, ok = df.GetFieldID("nonexistent")
	assert.False(t, ok)
}

// =============================================================================
// Stress Test for Memory Leaks
// =============================================================================

func TestMemoryLeakStress(t *testing.T) {
	pool := memory.NewCheckedAllocator(memory.NewGoAllocator())
	defer pool.AssertSize(t, 0)

	resultData := &schemapb.SearchResultData{
		NumQueries: 5,
		TopK:       10,
		Topks:      []int64{10, 10, 10, 10, 10},
		Scores:     make([]float32, 50),
		Ids: &schemapb.IDs{
			IdField: &schemapb.IDs_IntId{
				IntId: &schemapb.LongArray{
					Data: make([]int64, 50),
				},
			},
		},
		FieldsData: []*schemapb.FieldData{
			{
				Type:      schemapb.DataType_Int64,
				FieldName: "col1",
				FieldId:   1,
				Field: &schemapb.FieldData_Scalars{
					Scalars: &schemapb.ScalarField{
						Data: &schemapb.ScalarField_LongData{
							LongData: &schemapb.LongArray{
								Data: make([]int64, 50),
							},
						},
					},
				},
			},
			{
				Type:      schemapb.DataType_Double,
				FieldName: "col2",
				FieldId:   2,
				Field: &schemapb.FieldData_Scalars{
					Scalars: &schemapb.ScalarField{
						Data: &schemapb.ScalarField_DoubleData{
							DoubleData: &schemapb.DoubleArray{
								Data: make([]float64, 50),
							},
						},
					},
				},
			},
			{
				Type:      schemapb.DataType_VarChar,
				FieldName: "col3",
				FieldId:   3,
				Field: &schemapb.FieldData_Scalars{
					Scalars: &schemapb.ScalarField{
						Data: &schemapb.ScalarField_StringData{
							StringData: &schemapb.StringArray{
								Data: make([]string, 50),
							},
						},
					},
				},
			},
		},
	}

	// Fill in some data
	for i := range resultData.Scores {
		resultData.Scores[i] = float32(i) * 0.1
	}
	for i := range resultData.Ids.GetIntId().Data {
		resultData.Ids.GetIntId().Data[i] = int64(i)
	}

	// Stress test: many import/export cycles
	for range 100 {
		df, err := FromSearchResultData(resultData)
		assert.NoError(t, err)

		// Access columns
		_, _ = df.Column(IDFieldName)
		_, _ = df.Column(ScoreFieldName)
		_, _ = df.GetColumns([]string{"col1", "col2"}, 0)

		// Export
		_, err = df.ToSearchResultData()
		assert.NoError(t, err)

		df.Release()
	}
}

// =============================================================================
// Additional Tests for Coverage
// =============================================================================

// TestImportExport_AllDataTypes tests import/export for all supported data types
func (s *DataFrameSuite) TestImportExport_AllDataTypes() {
	resultData := &schemapb.SearchResultData{
		NumQueries: 1,
		TopK:       3,
		Topks:      []int64{3},
		Scores:     []float32{0.9, 0.8, 0.7},
		Ids: &schemapb.IDs{
			IdField: &schemapb.IDs_IntId{
				IntId: &schemapb.LongArray{Data: []int64{1, 2, 3}},
			},
		},
		FieldsData: []*schemapb.FieldData{
			// Bool
			{
				Type:      schemapb.DataType_Bool,
				FieldName: "bool_col",
				FieldId:   1,
				Field: &schemapb.FieldData_Scalars{
					Scalars: &schemapb.ScalarField{
						Data: &schemapb.ScalarField_BoolData{
							BoolData: &schemapb.BoolArray{Data: []bool{true, false, true}},
						},
					},
				},
			},
			// Int8
			{
				Type:      schemapb.DataType_Int8,
				FieldName: "int8_col",
				FieldId:   2,
				Field: &schemapb.FieldData_Scalars{
					Scalars: &schemapb.ScalarField{
						Data: &schemapb.ScalarField_IntData{
							IntData: &schemapb.IntArray{Data: []int32{1, 2, 3}},
						},
					},
				},
			},
			// Int16
			{
				Type:      schemapb.DataType_Int16,
				FieldName: "int16_col",
				FieldId:   3,
				Field: &schemapb.FieldData_Scalars{
					Scalars: &schemapb.ScalarField{
						Data: &schemapb.ScalarField_IntData{
							IntData: &schemapb.IntArray{Data: []int32{100, 200, 300}},
						},
					},
				},
			},
			// Int32
			{
				Type:      schemapb.DataType_Int32,
				FieldName: "int32_col",
				FieldId:   4,
				Field: &schemapb.FieldData_Scalars{
					Scalars: &schemapb.ScalarField{
						Data: &schemapb.ScalarField_IntData{
							IntData: &schemapb.IntArray{Data: []int32{1000, 2000, 3000}},
						},
					},
				},
			},
			// Int64
			{
				Type:      schemapb.DataType_Int64,
				FieldName: "int64_col",
				FieldId:   5,
				Field: &schemapb.FieldData_Scalars{
					Scalars: &schemapb.ScalarField{
						Data: &schemapb.ScalarField_LongData{
							LongData: &schemapb.LongArray{Data: []int64{10000, 20000, 30000}},
						},
					},
				},
			},
			// Float
			{
				Type:      schemapb.DataType_Float,
				FieldName: "float_col",
				FieldId:   6,
				Field: &schemapb.FieldData_Scalars{
					Scalars: &schemapb.ScalarField{
						Data: &schemapb.ScalarField_FloatData{
							FloatData: &schemapb.FloatArray{Data: []float32{1.1, 2.2, 3.3}},
						},
					},
				},
			},
			// Double
			{
				Type:      schemapb.DataType_Double,
				FieldName: "double_col",
				FieldId:   7,
				Field: &schemapb.FieldData_Scalars{
					Scalars: &schemapb.ScalarField{
						Data: &schemapb.ScalarField_DoubleData{
							DoubleData: &schemapb.DoubleArray{Data: []float64{1.11, 2.22, 3.33}},
						},
					},
				},
			},
			// VarChar
			{
				Type:      schemapb.DataType_VarChar,
				FieldName: "varchar_col",
				FieldId:   8,
				Field: &schemapb.FieldData_Scalars{
					Scalars: &schemapb.ScalarField{
						Data: &schemapb.ScalarField_StringData{
							StringData: &schemapb.StringArray{Data: []string{"a", "b", "c"}},
						},
					},
				},
			},
		},
	}

	// Import
	df, err := FromSearchResultData(resultData)
	s.Require().NoError(err)
	defer df.Release()

	// Verify all columns exist
	s.True(df.HasColumn("bool_col"))
	s.True(df.HasColumn("int8_col"))
	s.True(df.HasColumn("int16_col"))
	s.True(df.HasColumn("int32_col"))
	s.True(df.HasColumn("int64_col"))
	s.True(df.HasColumn("float_col"))
	s.True(df.HasColumn("double_col"))
	s.True(df.HasColumn("varchar_col"))

	// Export
	exported, err := df.ToSearchResultData()
	s.Require().NoError(err)

	// Verify exported field count
	s.Len(exported.FieldsData, 8)

	// Verify some data
	s.Equal([]float32{0.9, 0.8, 0.7}, exported.Scores)
	s.Equal([]int64{1, 2, 3}, exported.Ids.GetIntId().GetData())
}

// TestImportExport_StringIDs tests import/export with string IDs
func (s *DataFrameSuite) TestImportExport_StringIDs() {
	resultData := &schemapb.SearchResultData{
		NumQueries: 1,
		TopK:       2,
		Topks:      []int64{2},
		Scores:     []float32{0.9, 0.8},
		Ids: &schemapb.IDs{
			IdField: &schemapb.IDs_StrId{
				StrId: &schemapb.StringArray{Data: []string{"id1", "id2"}},
			},
		},
	}

	df, err := FromSearchResultData(resultData)
	s.Require().NoError(err)
	defer df.Release()

	exported, err := df.ToSearchResultData()
	s.Require().NoError(err)

	s.Equal([]string{"id1", "id2"}, exported.Ids.GetStrId().GetData())
}

// TestColumnNames_NilSchema tests ColumnNames with nil schema
func (s *DataFrameSuite) TestColumnNames_NilSchema() {
	df := NewDataFrame()
	defer df.Release()

	names := df.ColumnNames()
	s.Nil(names)
}

// TestGetArrayValue tests the getArrayValue helper function
func TestGetArrayValue(t *testing.T) {
	pool := memory.NewCheckedAllocator(memory.NewGoAllocator())
	defer pool.AssertSize(t, 0)

	// Test Int64
	int64Builder := array.NewInt64Builder(pool)
	int64Builder.AppendValues([]int64{100, 200}, nil)
	int64Arr := int64Builder.NewArray()
	int64Builder.Release()
	defer int64Arr.Release()

	val := getArrayValue(int64Arr, 0)
	assert.Equal(t, int64(100), val)

	// Test Float32
	float32Builder := array.NewFloat32Builder(pool)
	float32Builder.AppendValues([]float32{1.5, 2.5}, nil)
	float32Arr := float32Builder.NewArray()
	float32Builder.Release()
	defer float32Arr.Release()

	val = getArrayValue(float32Arr, 1)
	assert.Equal(t, float32(2.5), val)

	// Test Float64
	float64Builder := array.NewFloat64Builder(pool)
	float64Builder.AppendValues([]float64{10.5, 20.5}, nil)
	float64Arr := float64Builder.NewArray()
	float64Builder.Release()
	defer float64Arr.Release()

	val = getArrayValue(float64Arr, 0)
	assert.Equal(t, float64(10.5), val)

	// Test String
	stringBuilder := array.NewStringBuilder(pool)
	stringBuilder.AppendValues([]string{"hello", "world"}, nil)
	stringArr := stringBuilder.NewArray()
	stringBuilder.Release()
	defer stringArr.Release()

	val = getArrayValue(stringArr, 1)
	assert.Equal(t, "world", val)

	// Test Boolean
	boolBuilder := array.NewBooleanBuilder(pool)
	boolBuilder.AppendValues([]bool{true, false}, nil)
	boolArr := boolBuilder.NewArray()
	boolBuilder.Release()
	defer boolArr.Release()

	val = getArrayValue(boolArr, 0)
	assert.Equal(t, true, val)

	// Test Int8
	int8Builder := array.NewInt8Builder(pool)
	int8Builder.AppendValues([]int8{10, 20}, nil)
	int8Arr := int8Builder.NewArray()
	int8Builder.Release()
	defer int8Arr.Release()

	val = getArrayValue(int8Arr, 1)
	assert.Equal(t, int8(20), val)

	// Test Int16
	int16Builder := array.NewInt16Builder(pool)
	int16Builder.AppendValues([]int16{100, 200}, nil)
	int16Arr := int16Builder.NewArray()
	int16Builder.Release()
	defer int16Arr.Release()

	val = getArrayValue(int16Arr, 0)
	assert.Equal(t, int16(100), val)

	// Test Int32
	int32Builder := array.NewInt32Builder(pool)
	int32Builder.AppendValues([]int32{1000, 2000}, nil)
	int32Arr := int32Builder.NewArray()
	int32Builder.Release()
	defer int32Arr.Release()

	val = getArrayValue(int32Arr, 1)
	assert.Equal(t, int32(2000), val)

	// Test null value
	int64BuilderWithNull := array.NewInt64Builder(pool)
	int64BuilderWithNull.AppendNull()
	int64ArrWithNull := int64BuilderWithNull.NewArray()
	int64BuilderWithNull.Release()
	defer int64ArrWithNull.Release()

	val = getArrayValue(int64ArrWithNull, 0)
	assert.Nil(t, val)
}
