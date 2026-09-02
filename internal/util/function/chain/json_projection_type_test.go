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

package chain

import (
	"math"
	"testing"

	"github.com/apache/arrow/go/v17/arrow"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
)

func TestJSONTypeAccumulator(t *testing.T) {
	accumulator := &JSONTypeAccumulator{}
	require.NoError(t, accumulator.Add(ExtractedJSONValue{Kind: JSONValueNull}))
	require.NoError(t, accumulator.Add(ExtractedJSONValue{Kind: JSONValueInt64, I64: 10}))
	require.NoError(t, accumulator.Add(ExtractedJSONValue{Kind: JSONValueFloat64, F64: 10.5}))
	dataType, err := accumulator.DataType()
	require.NoError(t, err)
	assert.Equal(t, schemapb.DataType_Double, dataType)

	allNull := &JSONTypeAccumulator{}
	require.NoError(t, allNull.Add(ExtractedJSONValue{Kind: JSONValueNull}))
	_, err = allNull.DataType()
	assert.ErrorContains(t, err, "all values are null or missing")

	tooLarge := &JSONTypeAccumulator{}
	err = tooLarge.Add(ExtractedJSONValue{Kind: JSONValueUint64, U64: math.MaxUint64})
	assert.ErrorContains(t, err, "beyond int64 range")

	complexValue := &JSONTypeAccumulator{}
	err = complexValue.Add(ExtractedJSONValue{Kind: JSONValueObject})
	assert.ErrorContains(t, err, "specify JSON")
}

func TestMergeJSONProjectionDataTypes(t *testing.T) {
	tests := []struct {
		name        string
		left        schemapb.DataType
		right       schemapb.DataType
		expected    schemapb.DataType
		expectError bool
	}{
		{name: "unknown", left: schemapb.DataType_None, right: schemapb.DataType_Int64, expected: schemapb.DataType_Int64},
		{name: "same", left: schemapb.DataType_Bool, right: schemapb.DataType_Bool, expected: schemapb.DataType_Bool},
		{name: "numeric promotion", left: schemapb.DataType_Int64, right: schemapb.DataType_Double, expected: schemapb.DataType_Double},
		{name: "string normalization", left: schemapb.DataType_String, right: schemapb.DataType_Text, expected: schemapb.DataType_VarChar},
		{name: "incompatible", left: schemapb.DataType_Int64, right: schemapb.DataType_VarChar, expectError: true},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual, err := MergeJSONProjectionDataTypes(test.left, test.right)
			if test.expectError {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, test.expected, actual)
		})
	}
}

func TestValidateJSONProjectionValue(t *testing.T) {
	require.NoError(t, ValidateJSONProjectionValue(
		ExtractedJSONValue{Kind: JSONValueInt64, I64: 10}, schemapb.DataType_Double))
	require.NoError(t, ValidateJSONProjectionValue(
		ExtractedJSONValue{Kind: JSONValueObject}, schemapb.DataType_JSON))
	require.NoError(t, ValidateJSONProjectionValue(
		ExtractedJSONValue{Kind: JSONValueNull}, schemapb.DataType_VarChar))

	err := ValidateJSONProjectionValue(
		ExtractedJSONValue{Kind: JSONValueInt64, I64: math.MaxInt16}, schemapb.DataType_Int8)
	assert.ErrorContains(t, err, "incompatible")

	err = ValidateJSONProjectionValue(
		ExtractedJSONValue{Kind: JSONValueString}, schemapb.DataType_Double)
	assert.ErrorContains(t, err, "String")

	err = ValidateJSONProjectionValue(
		ExtractedJSONValue{Kind: JSONValueFloat64, F64: math.MaxFloat64}, schemapb.DataType_Float)
	assert.ErrorContains(t, err, "incompatible")
}

func TestJSONProjectionArrowType(t *testing.T) {
	dataType, err := JSONProjectionArrowType(schemapb.DataType_None)
	require.NoError(t, err)
	assert.True(t, arrow.TypeEqual(arrow.Null, dataType))

	dataType, err = JSONProjectionArrowType(schemapb.DataType_JSON)
	require.NoError(t, err)
	assert.True(t, arrow.TypeEqual(arrow.BinaryTypes.Binary, dataType))

	dataType, err = JSONProjectionArrowType(schemapb.DataType_Double)
	require.NoError(t, err)
	assert.True(t, arrow.TypeEqual(arrow.PrimitiveTypes.Float64, dataType))
}
