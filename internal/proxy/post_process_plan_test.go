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

	"github.com/apache/arrow/go/v17/arrow/array"
	"github.com/apache/arrow/go/v17/arrow/memory"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/util/function/chain"
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

	t.Run("rejects dynamic input without enabled dynamic field", func(t *testing.T) {
		plan, err := buildPostProcessPlan(postProcessFunctionChain(
			postProcessRoundDecimalMapOp("temporary", `$meta["age"]`),
		), schema)
		require.Error(t, err)
		assert.Nil(t, plan)
		assert.ErrorIs(t, err, merr.ErrParameterInvalid)
		assert.Contains(t, err.Error(), "cannot parse identifier")
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
		{input: "metadata", errContains: "complete JSON root input is not supported"},
		{input: `metadata["user"]["score"]`, errContains: "requires an explicit data_type"},
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
		{name: "unknown field", input: "unknown", errContains: "cannot parse identifier"},
		{name: "vector field", input: "vec", errContains: "unsupported field type FloatVector"},
		{name: "unsupported system input", input: "$timestamp", errContains: "unsupported function chain system input"},
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

func TestPostProcessSharedInputPlan(t *testing.T) {
	schema := newFunctionChainJSONTestSchema()
	for _, tc := range []struct {
		name    string
		input   string
		hint    schemapb.DataType
		wantErr string
	}{
		{name: "dynamic", input: `$meta["profile"]["rank"]`, hint: schemapb.DataType_Double},
		{name: "JSON array", input: `metadata["items"][0]["rank"]`, hint: schemapb.DataType_Int64},
		{name: "missing hint", input: `$meta["rank"]`, wantErr: "requires an explicit data_type"},
		{name: "unsupported hint", input: `$meta["rank"]`, hint: schemapb.DataType_FloatVector, wantErr: "unsupported JSON path data type hint"},
		{name: "bare dynamic name", input: "rank", hint: schemapb.DataType_Double, wantErr: "must use explicit"},
		{name: "complete root", input: "$meta", wantErr: "complete JSON root input"},
		{name: "scalar hint mismatch", input: "pk", hint: schemapb.DataType_VarChar, wantErr: "incompatible with schema field type"},
		{name: "system hint", input: types.ScoreFieldName, hint: schemapb.DataType_Float, wantErr: "system input does not accept"},
		{name: "unknown system input", input: "$timestamp", wantErr: "unsupported function chain system input"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			op := postProcessRoundDecimalMapOp("temporary", tc.input)
			op.Params = map[string]*schemapb.FunctionParamValue{
				types.InputDataTypesParam: chainDataTypesParam(tc.hint),
			}
			post, postErr := buildPostProcessPlan(postProcessFunctionChain(op), schema)
			rerank, rerankErr := newFunctionChainRerankMeta([]*schemapb.FunctionChain{l2FunctionChain(op)}, schema)
			if tc.wantErr != "" {
				require.ErrorIs(t, postErr, merr.ErrParameterInvalid)
				require.ErrorIs(t, rerankErr, merr.ErrParameterInvalid)
				assert.ErrorContains(t, postErr, tc.wantErr)
				assert.EqualError(t, postErr, rerankErr.Error())
				assert.Nil(t, post)
				return
			}
			require.NoError(t, postErr)
			require.NoError(t, rerankErr)
			assert.Equal(t, rerank.GetInputPlan(), post.GetInputPlan())
			assert.Equal(t, rerank.GetInputFieldIDs(), post.GetInputFieldIDs())
			assert.Equal(t, rerank.GetInputFieldNames(), post.GetInputFieldNames())
			require.Len(t, post.GetInputPlan().Inputs, 1)
			assert.Equal(t, tc.input, post.GetInputPlan().Inputs[0].LogicalName)
			assert.Equal(t, tc.hint, post.GetInputPlan().Inputs[0].DataTypeHint)
		})
	}

	t.Run("deduplicates physical roots and excludes temporary columns", func(t *testing.T) {
		first := postProcessRoundDecimalMapOp("first", `$meta["profile"]["rank"]`)
		second := postProcessRoundDecimalMapOp("second", `$meta["profile"]["bonus"]`)
		for _, op := range []*schemapb.FunctionChainOp{first, second} {
			op.Params = map[string]*schemapb.FunctionParamValue{
				types.InputDataTypesParam: chainDataTypesParam(schemapb.DataType_Double),
			}
		}
		plan, err := buildPostProcessPlan(postProcessFunctionChain(first, second,
			postProcessRoundDecimalMapOp("third", "first")), schema)
		require.NoError(t, err)
		assert.Equal(t, []string{"$meta"}, plan.GetInputFieldNames())
		assert.Equal(t, []int64{103}, plan.GetInputFieldIDs())
		require.Len(t, plan.GetInputPlan().Inputs, 2)
		assert.Equal(t, []string{"profile", "rank"}, plan.GetInputPlan().Inputs[0].NestedPath)
		assert.Equal(t, []string{"profile", "bonus"}, plan.GetInputPlan().Inputs[1].NestedPath)

		second.Expr.Args = first.Expr.Args
		second.Params[types.InputDataTypesParam] = chainDataTypesParam(schemapb.DataType_Int64)
		_, err = buildPostProcessPlan(postProcessFunctionChain(first, second), schema)
		require.ErrorIs(t, err, merr.ErrParameterInvalid)
		assert.ErrorContains(t, err, "conflicting data type hints")
	})
}

func TestPostProcessInputPlanMaterialization(t *testing.T) {
	path := `$meta["profile"]["rank"]`
	op := postProcessRoundDecimalMapOp("display_rank", path)
	op.Params = map[string]*schemapb.FunctionParamValue{
		types.InputDataTypesParam: chainDataTypesParam(schemapb.DataType_Double),
	}
	plan, err := buildPostProcessPlan(postProcessFunctionChain(op), newFunctionChainJSONTestSchema())
	require.NoError(t, err)

	pool := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer pool.AssertSize(t, 0)
	jsonData := &schemapb.JSONArray{Data: [][]byte{
		[]byte(`{"profile":{"rank":12.5}}`), []byte(`{}`), []byte(`{"profile":{"rank":"invalid"}}`),
	}}
	result := &schemapb.SearchResultData{
		NumQueries: 1, Topks: []int64{3}, Scores: []float32{3, 2, 1},
		Ids: &schemapb.IDs{IdField: &schemapb.IDs_IntId{IntId: &schemapb.LongArray{Data: []int64{1, 2, 3}}}},
		FieldsData: []*schemapb.FieldData{{
			FieldId: 103, FieldName: "$meta", Type: schemapb.DataType_JSON, IsDynamic: true,
			Field: &schemapb.FieldData_Scalars{Scalars: &schemapb.ScalarField{
				Data: &schemapb.ScalarField_JsonData{JsonData: jsonData},
			}},
		}},
	}
	df, err := chain.FromSearchResultData(result, pool, plan.GetInputPlan())
	require.NoError(t, err)
	defer df.Release()
	require.NoError(t, chain.ValidateMaterializedInput(df, plan.GetInputPlan().Inputs[0]))
	values := df.Column(path).Chunk(0).(*array.Float64)
	assert.Equal(t, 12.5, values.Value(0))
	assert.True(t, values.IsNull(1))
	assert.True(t, values.IsNull(2))

	jsonData.Data[0] = []byte(`{broken`)
	_, err = chain.FromSearchResultData(result, pool, plan.GetInputPlan())
	require.ErrorIs(t, err, merr.ErrDataIntegrity)

	result.FieldsData = nil
	_, err = chain.FromSearchResultData(result, pool, plan.GetInputPlan())
	require.ErrorIs(t, err, merr.ErrServiceInternal)
}
