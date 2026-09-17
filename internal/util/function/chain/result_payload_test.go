// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package chain

import (
	"context"
	"testing"

	"github.com/apache/arrow/go/v17/arrow/memory"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/util/function/chain/types"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

func completePayloadResult() *schemapb.SearchResultData {
	fields := []*schemapb.FieldData{
		jsonProjectorTestInt64Field(101, "value", []int64{10, 11, 12, 13, 14}),
		jsonProjectorTestJSONField(102, "$meta", []string{`{"v":0 }`, `{"v":1 }`, `{"v":2 }`, `{"v":3 }`, `{"v":4 }`}, nil),
		{FieldId: 103, FieldName: "float_vector", Type: schemapb.DataType_FloatVector, Field: &schemapb.FieldData_Vectors{Vectors: &schemapb.VectorField{Dim: 2, Data: &schemapb.VectorField_FloatVector{FloatVector: &schemapb.FloatArray{Data: []float32{1, 2, 3, 4, 5, 6}}}}}},
		{FieldId: 104, FieldName: "binary_vector", Type: schemapb.DataType_BinaryVector, Field: &schemapb.FieldData_Vectors{Vectors: &schemapb.VectorField{Dim: 8, Data: &schemapb.VectorField_BinaryVector{BinaryVector: []byte{1, 3, 5}}}}},
		{FieldId: 105, FieldName: "fp16", Type: schemapb.DataType_Float16Vector, Field: &schemapb.FieldData_Vectors{Vectors: &schemapb.VectorField{Dim: 2, Data: &schemapb.VectorField_Float16Vector{Float16Vector: []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}}}}},
		{FieldId: 106, FieldName: "bf16", Type: schemapb.DataType_BFloat16Vector, Field: &schemapb.FieldData_Vectors{Vectors: &schemapb.VectorField{Dim: 2, Data: &schemapb.VectorField_Bfloat16Vector{Bfloat16Vector: []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}}}}},
		{FieldId: 107, FieldName: "int8_vector", Type: schemapb.DataType_Int8Vector, Field: &schemapb.FieldData_Vectors{Vectors: &schemapb.VectorField{Dim: 2, Data: &schemapb.VectorField_Int8Vector{Int8Vector: []byte{1, 2, 3, 4, 5, 6}}}}},
		{FieldId: 108, FieldName: "sparse", Type: schemapb.DataType_SparseFloatVector, Field: &schemapb.FieldData_Vectors{Vectors: &schemapb.VectorField{Dim: 1, Data: &schemapb.VectorField_SparseFloatVector{SparseFloatVector: &schemapb.SparseFloatArray{Dim: 1, Contents: [][]byte{
			typeutil.CreateSparseFloatRow([]uint32{0}, []float32{1}), typeutil.CreateSparseFloatRow([]uint32{0}, []float32{3}), typeutil.CreateSparseFloatRow([]uint32{0}, []float32{5}),
		}}}}}},
		{FieldId: 109, FieldName: "array", Type: schemapb.DataType_Array, Field: &schemapb.FieldData_Scalars{Scalars: &schemapb.ScalarField{Data: &schemapb.ScalarField_ArrayData{ArrayData: &schemapb.ArrayArray{ElementType: schemapb.DataType_Int64}}}}},
		{FieldId: 110, FieldName: "vectors", Type: schemapb.DataType_ArrayOfVector, Field: &schemapb.FieldData_Vectors{Vectors: &schemapb.VectorField{Dim: 2, Data: &schemapb.VectorField_VectorArray{VectorArray: &schemapb.VectorArray{Dim: 2, ElementType: schemapb.DataType_FloatVector}}}}},
	}
	fields[1].IsDynamic = true
	for _, field := range fields[2:8] {
		typeutil.SetFieldDataValidData(field, []bool{true, false, true, false, true})
	}
	for i := 0; i < 5; i++ {
		fields[8].GetScalars().GetArrayData().Data = append(fields[8].GetScalars().GetArrayData().Data, &schemapb.ScalarField{Data: &schemapb.ScalarField_LongData{LongData: &schemapb.LongArray{Data: []int64{int64(i), int64(i + 1)}}}})
		fields[9].GetVectors().GetVectorArray().Data = append(fields[9].GetVectors().GetVectorArray().Data, &schemapb.VectorField{Dim: 2, Data: &schemapb.VectorField_FloatVector{FloatVector: &schemapb.FloatArray{Data: []float32{float32(i), float32(i + 1)}}}})
	}
	data := jsonProjectorTestResult([]int64{3, 2}, fields...)
	data.Ids.GetIntId().Data = []int64{7, 7, 9, 7, 8}
	data.Scores = []float32{.1, .9, .5, .7, .8}
	data.Distances = []float32{1, 9, 5, 7, 8}
	data.Recalls = []float32{.75, .5}
	return data
}

func TestResultPayloadSortLimit(t *testing.T) {
	pool := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer pool.AssertSize(t, 0)
	original := completePayloadResult()
	// Independent protobuf row selection is the expected result, not the
	// implementation: production uses Arrow arrays through the whole chain.
	want := typeutil.PrepareResultFieldData(original.FieldsData, 0)
	physical := make([][]int64, 5)
	computer := typeutil.NewFieldDataIdxComputer(original.FieldsData)
	for i := range physical {
		physical[i] = append([]int64(nil), computer.Compute(int64(i))...)
	}
	for _, row := range []int64{1, 2, 4, 3} {
		typeutil.AppendFieldData(want, original.FieldsData, row, physical[row]...)
	}
	input, err := FromSearchResultDataWithPayload(original, pool, nil)
	require.NoError(t, err)
	defer input.Release()
	// Prove result export no longer reads any external response rows.
	original.FieldsData[0].GetScalars().GetLongData().Data[1] = -999
	original.FieldsData = nil
	original.Ids = nil
	original.Scores = nil
	result, err := NewFuncChainWithAllocator(pool).SetStage(types.StagePostProcess).
		Sort(types.ScoreFieldName, true, types.IDFieldName).Limit(2).
		ExecuteWithOptions(context.Background(), ExecuteOptions{EnableColumnPruning: true}, input)
	require.NoError(t, err)
	defer result.Release()
	output, err := ToSearchResultDataWithPayload(result)
	require.NoError(t, err)
	assert.Equal(t, []int64{2, 2}, output.Topks)
	assert.Equal(t, []int64{7, 9, 8, 7}, output.Ids.GetIntId().Data)
	assert.Equal(t, []float32{9, 5, 8, 7}, output.Distances)
	assert.Equal(t, []float32{.75, .5}, output.Recalls)
	require.Len(t, output.FieldsData, len(want))
	for i, field := range output.FieldsData {
		assert.True(t, proto.Equal(want[i], field), "field %s: expected %s got %s", field.FieldName, want[i], field)
	}
}

func TestResultPayloadEmptyAndMissing(t *testing.T) {
	pool := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer pool.AssertSize(t, 0)
	data := completePayloadResult()
	input, err := FromSearchResultDataWithPayload(data, pool, nil)
	require.NoError(t, err)
	defer input.Release()
	output, err := NewFuncChainWithAllocator(pool).SetStage(types.StagePostProcess).LimitWithOffset(1, 100).Execute(input)
	require.NoError(t, err)
	defer output.Release()
	result, err := ToSearchResultDataWithPayload(output)
	require.NoError(t, err)
	assert.Equal(t, []int64{0, 0}, result.Topks)
	for _, field := range result.FieldsData {
		require.NoError(t, ValidateResultField(field, 0))
	}
	for name := range input.resultFields {
		delete(input.resultFields, name)
		break
	}
	_, err = ToSearchResultDataWithPayload(input)
	require.ErrorIs(t, err, merr.ErrServiceInternal)
}
