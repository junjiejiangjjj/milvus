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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/util/function/chain/types"
	"github.com/milvus-io/milvus/pkg/v3/common"
)

func functionChainInputPlanTestSchema(dynamic bool) *schemapb.CollectionSchema {
	fields := []*schemapb.FieldSchema{
		{FieldID: 100, Name: "id", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
		{FieldID: 101, Name: "price", DataType: schemapb.DataType_Double},
		{FieldID: 102, Name: "title", DataType: schemapb.DataType_VarChar},
		{FieldID: 103, Name: "metadata", DataType: schemapb.DataType_JSON},
	}
	if dynamic {
		fields = append(fields, &schemapb.FieldSchema{
			FieldID: 104, Name: common.MetaFieldName, DataType: schemapb.DataType_JSON, IsDynamic: true,
		})
	}
	return &schemapb.CollectionSchema{
		Name:               "function_chain_input_plan",
		EnableDynamicField: dynamic,
		Fields:             fields,
	}
}

func TestCompileDataFrameInputPlan(t *testing.T) {
	repr := &ChainRepr{Operators: []OperatorRepr{
		{
			Type: types.OpTypeMap,
			Inputs: []string{
				types.ScoreFieldName,
				"price",
				`metadata["price"]`,
				`metadata["price"]`,
			},
			InputDataTypes: []schemapb.DataType{
				schemapb.DataType_None,
				schemapb.DataType_None,
				schemapb.DataType_None,
				schemapb.DataType_Double,
			},
			Outputs: []string{"temporary"},
		},
		{
			Type:           types.OpTypeMap,
			Inputs:         []string{"temporary", `$meta["content"]`},
			InputDataTypes: []schemapb.DataType{schemapb.DataType_None, schemapb.DataType_VarChar},
			Outputs:        []string{types.ScoreFieldName},
		},
	}}

	plan, err := CompileDataFrameInputPlan(repr, functionChainInputPlanTestSchema(true))
	require.NoError(t, err)
	require.Len(t, plan.Inputs, 3)
	assert.Equal(t, "price", plan.Inputs[0].LogicalName)
	assert.Equal(t, schemapb.DataType_Double, plan.Inputs[0].DataType)
	assert.Equal(t, `metadata["price"]`, plan.Inputs[1].LogicalName)
	assert.Equal(t, []string{"price"}, plan.Inputs[1].NestedPath)
	assert.Equal(t, schemapb.DataType_Double, plan.Inputs[1].DataTypeHint)
	assert.Equal(t, `$meta["content"]`, plan.Inputs[2].LogicalName)
	assert.Equal(t, []string{"content"}, plan.Inputs[2].NestedPath)
	assert.Equal(t, schemapb.DataType_VarChar, plan.Inputs[2].DataTypeHint)
	assert.ElementsMatch(t, []int64{101, 103, 104}, plan.PhysicalFieldIDs())
	assert.ElementsMatch(t, []string{"price", "metadata", common.MetaFieldName}, plan.PhysicalFieldNames())
}

func TestCompileDataFrameInputPlanRejectsBareDynamicInput(t *testing.T) {
	repr := &ChainRepr{Operators: []OperatorRepr{{Type: types.OpTypeMap, Inputs: []string{"content"}}}}
	_, err := CompileDataFrameInputPlan(repr, functionChainInputPlanTestSchema(true))
	require.Error(t, err)
	assert.ErrorContains(t, err, "must use explicit $meta[...] syntax")
}

func TestCompileDataFrameInputPlanRejectsInvalidInputs(t *testing.T) {
	tests := []struct {
		name   string
		input  string
		hint   schemapb.DataType
		schema *schemapb.CollectionSchema
		match  string
	}{
		{
			name: "dynamic field disabled", input: `$meta["content"]`,
			schema: functionChainInputPlanTestSchema(false), match: "cannot parse identifier",
		},
		{
			name: "nested scalar", input: `price["value"]`,
			schema: functionChainInputPlanTestSchema(true), match: "not supported accessed with",
		},
		{
			name: "unsupported JSON hint", input: `metadata["price"]`, hint: schemapb.DataType_FloatVector,
			schema: functionChainInputPlanTestSchema(true), match: "unsupported JSON path data type hint",
		},
		{
			name: "scalar hint mismatch", input: "price", hint: schemapb.DataType_VarChar,
			schema: functionChainInputPlanTestSchema(true), match: "incompatible with schema field type",
		},
		{
			name: "JSON root scalar hint", input: "metadata", hint: schemapb.DataType_Double,
			schema: functionChainInputPlanTestSchema(true), match: "JSON root input only accepts JSON",
		},
		{
			name: "system hint", input: types.ScoreFieldName, hint: schemapb.DataType_Float,
			schema: functionChainInputPlanTestSchema(true), match: "system input does not accept",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			repr := &ChainRepr{Operators: []OperatorRepr{{
				Type:           types.OpTypeMap,
				Inputs:         []string{test.input},
				InputDataTypes: []schemapb.DataType{test.hint},
			}}}
			_, err := CompileDataFrameInputPlan(repr, test.schema)
			require.Error(t, err)
			assert.ErrorContains(t, err, test.match)
		})
	}
}

func TestCompileDataFrameInputPlanRejectsConflictingHints(t *testing.T) {
	repr := &ChainRepr{Operators: []OperatorRepr{{
		Type:           types.OpTypeMap,
		Inputs:         []string{`metadata["price"]`, `metadata["price"]`},
		InputDataTypes: []schemapb.DataType{schemapb.DataType_Int64, schemapb.DataType_Double},
	}}}
	_, err := CompileDataFrameInputPlan(repr, functionChainInputPlanTestSchema(true))
	require.Error(t, err)
	assert.ErrorContains(t, err, "conflicting data type hints")
}

func TestCompileDataFrameInputPlanNormalizesStringHints(t *testing.T) {
	repr := &ChainRepr{Operators: []OperatorRepr{{
		Type:           types.OpTypeMap,
		Inputs:         []string{`metadata["category"]`, `metadata["category"]`},
		InputDataTypes: []schemapb.DataType{schemapb.DataType_String, schemapb.DataType_Text},
	}}}
	plan, err := CompileDataFrameInputPlan(repr, functionChainInputPlanTestSchema(true))
	require.NoError(t, err)
	require.Len(t, plan.Inputs, 1)
	assert.Equal(t, schemapb.DataType_VarChar, plan.Inputs[0].DataTypeHint)
}

func TestCompileDataFrameInputPlanJSONRoot(t *testing.T) {
	repr := &ChainRepr{Operators: []OperatorRepr{{Type: types.OpTypeMap, Inputs: []string{"metadata", common.MetaFieldName}}}}
	plan, err := CompileDataFrameInputPlan(repr, functionChainInputPlanTestSchema(true))
	require.NoError(t, err)
	require.Len(t, plan.Inputs, 2)
	assert.Empty(t, plan.Inputs[0].NestedPath)
	assert.Equal(t, schemapb.DataType_JSON, plan.Inputs[0].DataTypeHint)
	assert.Empty(t, plan.Inputs[1].NestedPath)
	assert.Equal(t, schemapb.DataType_JSON, plan.Inputs[1].DataTypeHint)
}
