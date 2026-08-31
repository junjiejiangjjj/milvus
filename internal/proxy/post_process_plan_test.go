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
package proxy

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/util/function/chain/types"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

func TestBuildPostProcessPlan(t *testing.T) {
	schema := newFunctionChainTestSchema()
	mapChain := postProcessFunctionChain(postProcessRoundDecimalMapOp("display_score", types.ScoreFieldName))

	t.Run("no explicit post process", func(t *testing.T) {
		plan, err := buildPostProcessPlan(nil, schema)
		require.NoError(t, err)
		assert.Nil(t, plan)
	})

	t.Run("explicit post process", func(t *testing.T) {
		plan, err := buildPostProcessPlan(mapChain, schema)
		require.NoError(t, err)
		require.NotNil(t, plan)
		assert.Same(t, mapChain, plan.Chain)
		assert.NotNil(t, plan.ChainRepr)
	})

	t.Run("explicit sort is accepted by builder", func(t *testing.T) {
		plan, err := buildPostProcessPlan(postProcessFunctionChain(
			&schemapb.FunctionChainOp{Op: types.OpTypeSort, Inputs: []string{types.ScoreFieldName}},
		), schema)
		require.NoError(t, err)
		require.NotNil(t, plan)
		assert.Equal(t, types.OpTypeSort, plan.ChainRepr.Operators[0].Type)
	})

	t.Run("plans scalar schema dependencies and excludes temporary outputs", func(t *testing.T) {
		plan, err := buildPostProcessPlan(postProcessFunctionChain(
			postProcessRoundDecimalMapOp("temporary1", "ts"),
			postProcessRoundDecimalMapOp("temporary2", "ts"),
			&schemapb.FunctionChainOp{Op: types.OpTypeSort, Inputs: []string{"temporary1", "temporary2"}},
		), schema)
		require.NoError(t, err)
		require.NotNil(t, plan)
		assert.Equal(t, []string{"ts"}, plan.GetInputFieldNames())
		assert.Equal(t, []int64{101}, plan.GetInputFieldIDs())
	})

	t.Run("rejects dynamic input", func(t *testing.T) {
		plan, err := buildPostProcessPlan(postProcessFunctionChain(
			postProcessRoundDecimalMapOp("temporary", `$meta["age"]`),
		), schema)
		require.Error(t, err)
		assert.Nil(t, plan)
		assert.ErrorIs(t, err, merr.ErrParameterInvalid)
		assert.Contains(t, err.Error(), "dynamic field input")
	})

	t.Run("rejects dynamic output", func(t *testing.T) {
		plan, err := buildPostProcessPlan(postProcessFunctionChain(
			postProcessRoundDecimalMapOp(`$meta["age"]`, types.ScoreFieldName),
		), schema)
		require.Error(t, err)
		assert.Nil(t, plan)
		assert.ErrorIs(t, err, merr.ErrParameterInvalid)
		assert.Contains(t, err.Error(), "dynamic field output")
	})

	t.Run("rejects highlight output", func(t *testing.T) {
		plan, err := buildPostProcessPlan(postProcessFunctionChain(
			postProcessRoundDecimalMapOp(types.HighlightFieldName, types.ScoreFieldName),
		), schema)
		require.Error(t, err)
		assert.Nil(t, plan)
		assert.ErrorIs(t, err, merr.ErrParameterInvalid)
		assert.Contains(t, err.Error(), `output "$highlight" is not supported yet`)
	})

	t.Run("rejects schema field overwrite", func(t *testing.T) {
		plan, err := buildPostProcessPlan(postProcessFunctionChain(
			postProcessRoundDecimalMapOp("ts", types.ScoreFieldName),
		), schema)
		require.Error(t, err)
		assert.Nil(t, plan)
		assert.ErrorIs(t, err, merr.ErrParameterInvalid)
		assert.Contains(t, err.Error(), `cannot overwrite schema field "ts"`)
	})

	jsonSchema := mustNewSchemaInfo(&schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{
		{FieldID: 100, Name: "pk", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
		{FieldID: 101, Name: "metadata", DataType: schemapb.DataType_JSON},
		{FieldID: 102, Name: "content", DataType: schemapb.DataType_Text},
	}})
	for _, tc := range []struct {
		input       string
		errContains string
	}{
		{input: "metadata", errContains: "unsupported field type JSON"},
		{input: `metadata["user"]["score"]`, errContains: "JSON path input"},
	} {
		t.Run("rejects JSON input "+tc.input, func(t *testing.T) {
			plan, err := buildPostProcessPlan(postProcessFunctionChain(
				postProcessRoundDecimalMapOp("temporary", tc.input),
			), jsonSchema)
			require.Error(t, err)
			assert.Nil(t, plan)
			assert.ErrorIs(t, err, merr.ErrParameterInvalid)
			assert.Contains(t, err.Error(), tc.errContains)
		})
	}

	t.Run("accepts Text input supported by chain converter", func(t *testing.T) {
		plan, err := buildPostProcessPlan(postProcessFunctionChain(
			&schemapb.FunctionChainOp{Op: types.OpTypeSort, Inputs: []string{"content"}},
		), jsonSchema)
		require.NoError(t, err)
		require.NotNil(t, plan)
		assert.Equal(t, []string{"content"}, plan.GetInputFieldNames())
		assert.Equal(t, []int64{102}, plan.GetInputFieldIDs())
	})

	t.Run("rejects unknown map function", func(t *testing.T) {
		plan, err := buildPostProcessPlan(postProcessFunctionChain(
			mapOp("temporary", "unknown_post_process_function", columnArg(types.ScoreFieldName)),
		), schema)
		require.Error(t, err)
		assert.Nil(t, plan)
		assert.ErrorIs(t, err, merr.ErrParameterInvalid)
		assert.Contains(t, err.Error(), "unknown function")
	})

	t.Run("rejects function unavailable at post-process stage", func(t *testing.T) {
		op := mapOp("temporary", "xgboost", columnArg(types.ScoreFieldName))
		op.Expr.Params["model_resource"] = chainStringParam("test-model")
		plan, err := buildPostProcessPlan(postProcessFunctionChain(op), schema)
		require.Error(t, err)
		assert.Nil(t, plan)
		assert.ErrorIs(t, err, merr.ErrParameterInvalid)
		assert.Contains(t, err.Error(), `does not support stage "post_process"`)
	})

	t.Run("validates operator parameters", func(t *testing.T) {
		plan, err := buildPostProcessPlan(postProcessFunctionChain(
			&schemapb.FunctionChainOp{Op: types.OpTypeLimit},
		), schema)
		require.Error(t, err)
		assert.Nil(t, plan)
		assert.ErrorIs(t, err, merr.ErrParameterInvalid)
		assert.Contains(t, err.Error(), `limit_op: missing required parameter "limit"`)
	})

	for _, tc := range []struct {
		name        string
		input       string
		errContains string
	}{
		{name: "unknown field", input: "unknown", errContains: "neither a previous output nor a collection field"},
		{name: "vector field", input: "vec", errContains: "unsupported field type FloatVector"},
		{name: "unsupported system input", input: "$timestamp", errContains: "system input \"$timestamp\" is not supported"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			plan, err := buildPostProcessPlan(postProcessFunctionChain(
				postProcessRoundDecimalMapOp("temporary", tc.input),
			), schema)
			require.Error(t, err)
			assert.Nil(t, plan)
			assert.Contains(t, err.Error(), tc.errContains)
		})
	}
}
