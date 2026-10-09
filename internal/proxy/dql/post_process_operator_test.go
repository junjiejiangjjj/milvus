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
package dql

import (
	"context"
	"math"
	"testing"

	"github.com/apache/arrow/go/v17/arrow"
	"github.com/apache/arrow/go/v17/arrow/memory"
	"github.com/bytedance/mockey"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	chainexpr "github.com/milvus-io/milvus/internal/util/function/chain/expr"
	"github.com/milvus-io/milvus/internal/util/function/chain/types"
	"github.com/milvus-io/milvus/internal/util/segcore"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

func postProcessTestSort(input string, hint schemapb.DataType) *schemapb.FunctionChainOp {
	return &schemapb.FunctionChainOp{Op: types.OpTypeSort, Inputs: []string{input}, Params: map[string]*schemapb.FunctionParamValue{
		"orders":                  chainStringArrayParam("asc"),
		types.InputDataTypesParam: chainDataTypesParam(hint),
	}}
}

func postProcessTestLimit(limit, offset int64) *schemapb.FunctionChainOp {
	return &schemapb.FunctionChainOp{Op: types.OpTypeLimit, Params: map[string]*schemapb.FunctionParamValue{
		"limit": chainIntParam(limit), "offset": chainIntParam(offset),
	}}
}

func postProcessTestResult() *milvuspb.SearchResults {
	return &milvuspb.SearchResults{Status: merr.Success(), CollectionName: "test", SessionTs: 123, Results: &schemapb.SearchResultData{
		NumQueries: 3, Topks: []int64{3, 0, 2}, TopK: 3, AllSearchCount: 99,
		Ids:    &schemapb.IDs{IdField: &schemapb.IDs_IntId{IntId: &schemapb.LongArray{Data: []int64{7, 7, 9, 7, 8}}}},
		Scores: []float32{.9, .8, .7, .6, .5}, Distances: []float32{9, 8, 7, 6, 5}, Recalls: []float32{.5, 0, 1},
		FieldsData: []*schemapb.FieldData{
			{FieldId: 103, FieldName: "$meta", Type: schemapb.DataType_JSON, IsDynamic: true, Field: &schemapb.FieldData_Scalars{Scalars: &schemapb.ScalarField{Data: &schemapb.ScalarField_JsonData{JsonData: &schemapb.JSONArray{Data: [][]byte{
				[]byte(`{"rank":3,"title":"a"}`), []byte(`{"rank":1,"title":"b"}`), []byte(`{"rank":2,"title":"c"}`), []byte(`{"rank":2,"title":"d"}`), []byte(`{"rank":1,"title":"e"}`),
			}}}}}},
			{FieldId: 105, FieldName: "vec", Type: schemapb.DataType_FloatVector, Field: &schemapb.FieldData_Vectors{Vectors: &schemapb.VectorField{Dim: 2, Data: &schemapb.VectorField_FloatVector{FloatVector: &schemapb.FloatArray{Data: []float32{10, 11, 30, 31, 40, 41}}}}}},
			{FieldId: 106, FieldName: "text", Type: schemapb.DataType_Text, Field: &schemapb.FieldData_Scalars{Scalars: &schemapb.ScalarField{Data: &schemapb.ScalarField_StringData{StringData: &schemapb.StringArray{Data: []string{"a", "b", "c", "d", "e"}}}}}},
		},
	}}
}

func TestPostProcessOperatorRows(t *testing.T) {
	for _, stringPK := range []bool{false, true} {
		t.Run(map[bool]string{false: "int PK", true: "string PK"}[stringPK], func(t *testing.T) {
			original := postProcessTestResult()
			typeutil.SetFieldDataValidData(original.Results.FieldsData[1], []bool{true, false, true, true, false})
			if stringPK {
				original.Results.Ids = &schemapb.IDs{IdField: &schemapb.IDs_StrId{StrId: &schemapb.StringArray{Data: []string{"x", "x", "z", "x", "y"}}}}
			}
			before := proto.Clone(original)
			plan, err := buildPostProcessPlan(postProcessFunctionChain(
				postProcessRoundDecimalMapOp("title", types.ScoreFieldName),
				postProcessTestSort(`$meta["rank"]`, schemapb.DataType_Int64), postProcessTestLimit(2, 0),
			), newFunctionChainJSONTestSchema())
			require.NoError(t, err)
			pool := memory.NewCheckedAllocator(memory.DefaultAllocator)
			defer pool.AssertSize(t, 0)
			op := &postProcessOperator{plan: plan, alloc: pool, dynamicFields: []string{"title"}}
			outputs, err := op.run(context.Background(), trace.SpanFromContext(context.Background()), original)
			require.NoError(t, err)
			result := outputs[0].(*milvuspb.SearchResults)
			assert.True(t, proto.Equal(before, original), "input must remain unchanged")
			assert.Equal(t, []int64{2, 0, 2}, result.Results.Topks)
			assert.Equal(t, int64(2), result.Results.TopK)
			assert.Equal(t, int64(99), result.Results.AllSearchCount)
			assert.Equal(t, uint64(123), result.SessionTs)
			if stringPK {
				assert.Equal(t, []string{"x", "z", "y", "x"}, result.Results.Ids.GetStrId().Data)
			} else {
				assert.Equal(t, []int64{7, 9, 8, 7}, result.Results.Ids.GetIntId().Data)
			}
			assert.Equal(t, []float32{.8, .7, .5, .6}, result.Results.Scores)
			assert.Equal(t, []float32{8, 7, 5, 6}, result.Results.Distances)
			assert.Equal(t, []float32{.5, 0, 1}, result.Results.Recalls)
			require.Len(t, result.Results.FieldsData, 3)
			meta := result.Results.FieldsData[0]
			assert.JSONEq(t, `{"title":"b"}`, string(meta.GetScalars().GetJsonData().Data[0]))
			assert.JSONEq(t, `{"title":"e"}`, string(meta.GetScalars().GetJsonData().Data[2]))
			assert.True(t, meta.IsDynamic)
			vec := result.Results.FieldsData[1]
			assert.Equal(t, []bool{false, true, false, true}, typeutil.GetFieldDataValidData(vec))
			assert.Equal(t, []float32{30, 31, 40, 41}, vec.GetVectors().GetFloatVector().Data)
			assert.Equal(t, []string{"b", "c", "e", "d"}, result.Results.FieldsData[2].GetScalars().GetStringData().Data)
		})
	}
}

func TestPostProcessOperatorEmptyAndFailures(t *testing.T) {
	plan, err := buildPostProcessPlan(postProcessFunctionChain(postProcessTestSort(`$meta["rank"]`, schemapb.DataType_Int64)), newFunctionChainJSONTestSchema())
	require.NoError(t, err)
	for _, tc := range []struct {
		name   string
		mutate func(*milvuspb.SearchResults)
		target error
	}{
		{"malformed JSON", func(r *milvuspb.SearchResults) {
			r.Results.FieldsData[0].GetScalars().GetJsonData().Data[0] = []byte(`{broken`)
		}, merr.ErrDataIntegrity},
		{"missing root", func(r *milvuspb.SearchResults) { r.Results.FieldsData = r.Results.FieldsData[1:] }, merr.ErrServiceInternal},
		{"bad Topks", func(r *milvuspb.SearchResults) { r.Results.Topks[0] = math.MaxInt64 }, merr.ErrServiceInternal},
		{"negative Topks", func(r *milvuspb.SearchResults) { r.Results.Topks[0] = -1 }, merr.ErrServiceInternal},
		{"short IDs", func(r *milvuspb.SearchResults) { r.Results.Ids.GetIntId().Data = nil }, merr.ErrServiceInternal},
		{"short field", func(r *milvuspb.SearchResults) { r.Results.FieldsData[2].GetScalars().GetStringData().Data = nil }, merr.ErrServiceInternal},
		{"bad validity", func(r *milvuspb.SearchResults) { typeutil.SetFieldDataValidData(r.Results.FieldsData[2], []bool{true}) }, merr.ErrServiceInternal},
		{"nil payload", func(r *milvuspb.SearchResults) {
			r.Results.FieldsData[2].GetScalars().Data = &schemapb.ScalarField_StringData{}
		}, merr.ErrServiceInternal},
		{"mismatched payload type", func(r *milvuspb.SearchResults) { r.Results.FieldsData[1].Type = schemapb.DataType_Int64 }, merr.ErrServiceInternal},
		{"short compact vector", func(r *milvuspb.SearchResults) { r.Results.FieldsData[1].GetVectors().GetFloatVector().Data = nil }, merr.ErrServiceInternal},
		{"unsupported group result", func(r *milvuspb.SearchResults) { r.Results.GroupByFieldValue = r.Results.FieldsData[2] }, merr.ErrServiceInternal},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pool := memory.NewCheckedAllocator(memory.DefaultAllocator)
			defer pool.AssertSize(t, 0)
			original := postProcessTestResult()
			typeutil.SetFieldDataValidData(original.Results.FieldsData[1], []bool{true, false, true, true, false})
			tc.mutate(original)
			before := proto.Clone(original)
			outputs, err := (&postProcessOperator{plan: plan, alloc: pool}).run(context.Background(), nil, original)
			require.ErrorIs(t, err, tc.target)
			assert.Nil(t, outputs)
			assert.True(t, proto.Equal(before, original))
			assert.Equal(t, merr.SystemError, merr.GetErrorType(merr.Error(merr.Status(err))))
		})
	}
	t.Run("cancel", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		outputs, err := (&postProcessOperator{plan: plan, alloc: memory.DefaultAllocator}).run(ctx, nil, postProcessTestResult())
		require.ErrorIs(t, err, context.Canceled)
		assert.Nil(t, outputs)
	})
	for _, trim := range []bool{false, true} {
		t.Run(map[bool]string{false: "empty input", true: "limit removes all rows"}[trim], func(t *testing.T) {
			pool := memory.NewCheckedAllocator(memory.DefaultAllocator)
			defer pool.AssertSize(t, 0)
			original := postProcessTestResult()
			typeutil.SetFieldDataValidData(original.Results.FieldsData[1], []bool{true, false, true, true, false})
			if !trim {
				original.Results = &schemapb.SearchResultData{NumQueries: 3, Topks: []int64{0, 0, 0}}
			}
			plan, err := buildPostProcessPlan(postProcessFunctionChain(postProcessTestLimit(1, math.MaxInt64)), newFunctionChainJSONTestSchema())
			require.NoError(t, err)
			outputs, err := (&postProcessOperator{plan: plan, alloc: pool}).run(context.Background(), nil, original)
			require.NoError(t, err)
			result := outputs[0].(*milvuspb.SearchResults).Results
			assert.Equal(t, []int64{0, 0, 0}, result.Topks)
			assert.Zero(t, result.TopK)
			assert.Empty(t, result.Scores)
			for _, field := range result.FieldsData {
				assert.Empty(t, typeutil.GetFieldDataValidData(field))
			}
		})
	}
}

func TestPostProcessSearchPipelines(t *testing.T) {
	paramtable.Init()
	for _, withRerank := range []bool{false, true} {
		for _, withRequery := range []bool{false, true} {
			name := map[bool]string{false: "plain", true: "rerank"}[withRerank] + "/" + map[bool]string{false: "direct", true: "requery"}[withRequery]
			t.Run(name, func(t *testing.T) {
				schema := mustNewSchemaInfo(&schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{
					{FieldID: 100, Name: "pk", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
					{FieldID: 101, Name: "value", DataType: schemapb.DataType_Int64},
				}})
				plan, err := buildPostProcessPlan(postProcessFunctionChain(postProcessRoundDecimalMapOp("temporary", types.ScoreFieldName), postProcessTestSort("value", schemapb.DataType_None), postProcessTestLimit(1, 0)), schema)
				require.NoError(t, err)
				task := &SearchTask{ctx: context.Background(), SearchRequest: &internalpb.SearchRequest{Base: &commonpb.MsgBase{}, Nq: 1, Topk: 3, Offset: 1, MetricType: "COSINE"}, request: &milvuspb.SearchRequest{}, schema: schema, needRequery: withRequery, postProcessPlan: plan, queryInfos: []*planpb.QueryInfo{{RoundDecimal: -1}}, translatedOutputFields: []string{"value"}}
				if withRerank {
					task.rerankMeta, err = newFunctionChainRerankMeta([]*schemapb.FunctionChain{l2FunctionChain(postProcessRoundDecimalMapOp(types.ScoreFieldName, types.ScoreFieldName))}, schema)
					require.NoError(t, err)
				}
				value := &schemapb.FieldData{FieldId: 101, FieldName: "value", Type: schemapb.DataType_Int64, Field: &schemapb.FieldData_Scalars{Scalars: &schemapb.ScalarField{Data: &schemapb.ScalarField_LongData{LongData: &schemapb.LongArray{Data: []int64{10, 30, 20}}}}}}
				pk := &schemapb.FieldData{FieldId: 100, FieldName: "pk", Type: schemapb.DataType_Int64, Field: &schemapb.FieldData_Scalars{Scalars: &schemapb.ScalarField{Data: &schemapb.ScalarField_LongData{LongData: &schemapb.LongArray{Data: []int64{1, 2, 3}}}}}}
				if withRequery {
					mock := mockey.Mock((*requeryOperator).requery).Return(&milvuspb.QueryResults{FieldsData: []*schemapb.FieldData{value, pk}}, segcore.StorageCost{}, nil).Build()
					defer mock.UnPatch()
				}
				source := &schemapb.SearchResultData{NumQueries: 1, TopK: 3, Topks: []int64{3}, Scores: []float32{.9, .8, .7}, Ids: &schemapb.IDs{IdField: &schemapb.IDs_IntId{IntId: &schemapb.LongArray{Data: []int64{1, 2, 3}}}}, FieldsData: []*schemapb.FieldData{value, pk}, AllSearchCount: 33}
				blob, err := proto.Marshal(source)
				require.NoError(t, err)
				p, err := newSearchPipeline(task)
				require.NoError(t, err)
				require.Equal(t, postProcessOp, p.nodes[len(p.nodes)-2].opName)
				output, _, err := p.Run(context.Background(), trace.SpanFromContext(context.Background()), []*internalpb.SearchResults{{Status: merr.Success(), NumQueries: 1, TopK: 3, MetricType: "COSINE", SlicedBlob: blob}}, segcore.StorageCost{})
				require.NoError(t, err)
				assert.Equal(t, []int64{3}, output.Results.Ids.GetIntId().Data)
				assert.Equal(t, []int64{1}, output.Results.Topks)
				require.Len(t, output.Results.FieldsData, 1)
				assert.Equal(t, []int64{20}, output.Results.FieldsData[0].GetScalars().GetLongData().Data)
				assert.Equal(t, int64(33), output.Results.AllSearchCount)
			})
		}
	}
}

func TestPostProcessFunctionFailurePreservesError(t *testing.T) {
	plan, err := buildPostProcessPlan(postProcessFunctionChain(postProcessRoundDecimalMapOp("temporary", types.ScoreFieldName)), newFunctionChainJSONTestSchema())
	require.NoError(t, err)
	for _, target := range []error{merr.ErrServiceUnavailable, context.Canceled, context.DeadlineExceeded} {
		t.Run(target.Error(), func(t *testing.T) {
			pool := memory.NewCheckedAllocator(memory.DefaultAllocator)
			defer pool.AssertSize(t, 0)
			original := postProcessTestResult()
			typeutil.SetFieldDataValidData(original.Results.FieldsData[1], []bool{true, false, true, true, false})
			before := proto.Clone(original)
			patch := mockey.Mock((*chainexpr.RoundDecimalExpr).Execute).Return([]*arrow.Chunked(nil), target).Build()
			defer patch.UnPatch()
			outputs, err := (&postProcessOperator{plan: plan, alloc: pool}).run(context.Background(), nil, original)
			require.ErrorIs(t, err, target)
			assert.Nil(t, outputs)
			assert.True(t, proto.Equal(before, original))
			assert.Equal(t, merr.Code(target), merr.Code(merr.Error(merr.Status(err))))
		})
	}
}

func TestPostProcessRequeryGeometryPayload(t *testing.T) {
	field := &schemapb.FieldData{FieldId: 104, FieldName: "location", Type: schemapb.DataType_Geometry,
		Field: &schemapb.FieldData_Scalars{Scalars: &schemapb.ScalarField{
			Data: &schemapb.ScalarField_GeometryWktData{GeometryWktData: &schemapb.GeometryWktArray{Data: []string{"POINT (1 2)", "POINT (3 4)"}}},
		}},
	}
	input := &milvuspb.SearchResults{Results: &schemapb.SearchResultData{
		NumQueries: 1, Topks: []int64{2}, Scores: []float32{.1, .2},
		Ids:        &schemapb.IDs{IdField: &schemapb.IDs_IntId{IntId: &schemapb.LongArray{Data: []int64{2, 1}}}},
		FieldsData: []*schemapb.FieldData{field},
	}}
	plan, err := buildPostProcessPlan(postProcessFunctionChain(postProcessTestSort(types.IDFieldName, schemapb.DataType_None)), newFunctionChainJSONTestSchema())
	require.NoError(t, err)
	pool := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer pool.AssertSize(t, 0)
	output, err := (&postProcessOperator{plan: plan, alloc: pool}).run(context.Background(), nil, input)
	require.NoError(t, err)
	result := output[0].(*milvuspb.SearchResults)
	assert.Equal(t, []string{"POINT (3 4)", "POINT (1 2)"}, result.Results.FieldsData[0].GetScalars().GetGeometryWktData().Data)
}
