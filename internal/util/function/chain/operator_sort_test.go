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
	"context"
	"strings"
	"testing"

	"github.com/apache/arrow/go/v17/arrow"
	"github.com/apache/arrow/go/v17/arrow/array"
	"github.com/apache/arrow/go/v17/arrow/memory"
	"github.com/stretchr/testify/suite"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/util/function/chain/types"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

type SortOpTestSuite struct {
	suite.Suite
	pool *memory.CheckedAllocator
}

func (s *SortOpTestSuite) SetupTest() {
	s.pool = memory.NewCheckedAllocator(memory.NewGoAllocator())
}

func (s *SortOpTestSuite) TearDownTest() {
	s.pool.AssertSize(s.T(), 0)
}

func TestSortOpTestSuite(t *testing.T) {
	suite.Run(t, new(SortOpTestSuite))
}

// createSortTestDF creates a simple DataFrame with $id, $score columns and the given chunk sizes.
func (s *SortOpTestSuite) createSortTestDF(ids []int64, scores []float64, chunkSizes []int64) *DataFrame {
	builder := NewDataFrameBuilder()

	builder.SetChunkSizes(chunkSizes)

	offset := 0
	idChunks := make([]arrow.Array, len(chunkSizes))
	scoreChunks := make([]arrow.Array, len(chunkSizes))
	for i, size := range chunkSizes {
		idBuilder := array.NewInt64Builder(s.pool)
		scoreBuilder := array.NewFloat64Builder(s.pool)
		for j := 0; j < int(size); j++ {
			idBuilder.Append(ids[offset+j])
			scoreBuilder.Append(scores[offset+j])
		}
		idChunks[i] = idBuilder.NewArray()
		idBuilder.Release()
		scoreChunks[i] = scoreBuilder.NewArray()
		scoreBuilder.Release()
		offset += int(size)
	}

	err := builder.AddColumnFromChunks(types.IDFieldName, idChunks)
	s.Require().NoError(err)
	err = builder.AddColumnFromChunks(types.ScoreFieldName, scoreChunks)
	s.Require().NoError(err)

	return builder.Build()
}

func (s *SortOpTestSuite) TestSortDescending() {
	df := s.createSortTestDF(
		[]int64{1, 2, 3, 4},
		[]float64{0.1, 0.9, 0.5, 0.3},
		[]int64{4},
	)
	defer df.Release()

	op := newSortOp(types.ScoreFieldName, true, types.IDFieldName)
	ctx := types.NewFuncContextFull(context.TODO(), s.pool, "rerank")
	result, err := op.Execute(ctx, df)
	s.Require().NoError(err)
	defer result.Release()

	s.Equal(int64(4), result.NumRows())
	scoreCol := result.Column(types.ScoreFieldName)
	scores := scoreCol.Chunk(0).(*array.Float64)
	// Should be sorted descending: 0.9, 0.5, 0.3, 0.1
	s.InDelta(0.9, scores.Value(0), 1e-9)
	s.InDelta(0.5, scores.Value(1), 1e-9)
	s.InDelta(0.3, scores.Value(2), 1e-9)
	s.InDelta(0.1, scores.Value(3), 1e-9)
}

func (s *SortOpTestSuite) TestSortAscending() {
	df := s.createSortTestDF(
		[]int64{1, 2, 3, 4},
		[]float64{0.9, 0.1, 0.5, 0.3},
		[]int64{4},
	)
	defer df.Release()

	op := newSortOp(types.ScoreFieldName, false, types.IDFieldName)
	ctx := types.NewFuncContextFull(context.TODO(), s.pool, "rerank")
	result, err := op.Execute(ctx, df)
	s.Require().NoError(err)
	defer result.Release()

	scoreCol := result.Column(types.ScoreFieldName)
	scores := scoreCol.Chunk(0).(*array.Float64)
	// Should be sorted ascending: 0.1, 0.3, 0.5, 0.9
	s.InDelta(0.1, scores.Value(0), 1e-9)
	s.InDelta(0.3, scores.Value(1), 1e-9)
	s.InDelta(0.5, scores.Value(2), 1e-9)
	s.InDelta(0.9, scores.Value(3), 1e-9)
}

func (s *SortOpTestSuite) TestSortWithAllNullScores() {
	builder := NewDataFrameBuilder()
	builder.SetChunkSizes([]int64{3})

	idBuilder := array.NewInt64Builder(s.pool)
	idBuilder.AppendValues([]int64{1, 2, 3}, nil)
	idChunk := idBuilder.NewArray()
	idBuilder.Release()

	scoreBuilder := array.NewFloat64Builder(s.pool)
	scoreBuilder.AppendNull()
	scoreBuilder.AppendNull()
	scoreBuilder.AppendNull()
	scoreChunk := scoreBuilder.NewArray()
	scoreBuilder.Release()

	err := builder.AddColumnFromChunks(types.IDFieldName, []arrow.Array{idChunk})
	s.Require().NoError(err)
	err = builder.AddColumnFromChunks(types.ScoreFieldName, []arrow.Array{scoreChunk})
	s.Require().NoError(err)

	df := builder.Build()
	defer df.Release()

	op := newSortOp(types.ScoreFieldName, true, types.IDFieldName)
	ctx := types.NewFuncContextFull(context.TODO(), s.pool, "rerank")
	result, err := op.Execute(ctx, df)
	s.Require().NoError(err)
	defer result.Release()

	// All nulls - should still have 3 rows and not panic
	s.Equal(int64(3), result.NumRows())
	scoreCol := result.Column(types.ScoreFieldName)
	scores := scoreCol.Chunk(0)
	for i := 0; i < 3; i++ {
		s.True(scores.IsNull(i))
	}
}

func (s *SortOpTestSuite) TestSortNullsLastDescending() {
	// Scores: null, 0.9, null, 0.1, 0.5  IDs: 1, 2, 3, 4, 5
	// DESC: non-null descending first, then nulls at the end
	// Expected: 0.9(2), 0.5(5), 0.1(4), null(1), null(3)
	builder := NewDataFrameBuilder()
	builder.SetChunkSizes([]int64{5})

	idBuilder := array.NewInt64Builder(s.pool)
	idBuilder.AppendValues([]int64{1, 2, 3, 4, 5}, nil)
	idChunk := idBuilder.NewArray()
	idBuilder.Release()

	scoreBuilder := array.NewFloat64Builder(s.pool)
	scoreBuilder.AppendNull()
	scoreBuilder.Append(0.9)
	scoreBuilder.AppendNull()
	scoreBuilder.Append(0.1)
	scoreBuilder.Append(0.5)
	scoreChunk := scoreBuilder.NewArray()
	scoreBuilder.Release()

	err := builder.AddColumnFromChunks(types.IDFieldName, []arrow.Array{idChunk})
	s.Require().NoError(err)
	err = builder.AddColumnFromChunks(types.ScoreFieldName, []arrow.Array{scoreChunk})
	s.Require().NoError(err)

	df := builder.Build()
	defer df.Release()

	op := newSortOp(types.ScoreFieldName, true, types.IDFieldName)
	ctx := types.NewFuncContextFull(context.TODO(), s.pool, "rerank")
	result, err := op.Execute(ctx, df)
	s.Require().NoError(err)
	defer result.Release()

	scores := result.Column(types.ScoreFieldName).Chunk(0).(*array.Float64)
	s.InDelta(0.9, scores.Value(0), 1e-9)
	s.InDelta(0.5, scores.Value(1), 1e-9)
	s.InDelta(0.1, scores.Value(2), 1e-9)
	s.True(scores.IsNull(3))
	s.True(scores.IsNull(4))
}

func (s *SortOpTestSuite) TestSortNullsLastAscending() {
	// Scores: null, 0.9, null, 0.1, 0.5  IDs: 1, 2, 3, 4, 5
	// ASC: non-null ascending first, then nulls at the end
	// Expected: 0.1(4), 0.5(5), 0.9(2), null(1), null(3)
	builder := NewDataFrameBuilder()
	builder.SetChunkSizes([]int64{5})

	idBuilder := array.NewInt64Builder(s.pool)
	idBuilder.AppendValues([]int64{1, 2, 3, 4, 5}, nil)
	idChunk := idBuilder.NewArray()
	idBuilder.Release()

	scoreBuilder := array.NewFloat64Builder(s.pool)
	scoreBuilder.AppendNull()
	scoreBuilder.Append(0.9)
	scoreBuilder.AppendNull()
	scoreBuilder.Append(0.1)
	scoreBuilder.Append(0.5)
	scoreChunk := scoreBuilder.NewArray()
	scoreBuilder.Release()

	err := builder.AddColumnFromChunks(types.IDFieldName, []arrow.Array{idChunk})
	s.Require().NoError(err)
	err = builder.AddColumnFromChunks(types.ScoreFieldName, []arrow.Array{scoreChunk})
	s.Require().NoError(err)

	df := builder.Build()
	defer df.Release()

	op := newSortOp(types.ScoreFieldName, false, types.IDFieldName)
	ctx := types.NewFuncContextFull(context.TODO(), s.pool, "rerank")
	result, err := op.Execute(ctx, df)
	s.Require().NoError(err)
	defer result.Release()

	scores := result.Column(types.ScoreFieldName).Chunk(0).(*array.Float64)
	s.InDelta(0.1, scores.Value(0), 1e-9)
	s.InDelta(0.5, scores.Value(1), 1e-9)
	s.InDelta(0.9, scores.Value(2), 1e-9)
	s.True(scores.IsNull(3))
	s.True(scores.IsNull(4))
}

func (s *SortOpTestSuite) TestSortMultiChunkIndependent() {
	// Two chunks (two queries), each sorted independently
	df := s.createSortTestDF(
		[]int64{1, 2, 3, 4, 5, 6},
		[]float64{0.1, 0.9, 0.5, 0.8, 0.2, 0.6},
		[]int64{3, 3},
	)
	defer df.Release()

	op := newSortOp(types.ScoreFieldName, true, types.IDFieldName)
	ctx := types.NewFuncContextFull(context.TODO(), s.pool, "rerank")
	result, err := op.Execute(ctx, df)
	s.Require().NoError(err)
	defer result.Release()

	// Chunk 0 sorted desc: 0.9, 0.5, 0.1
	scores0 := result.Column(types.ScoreFieldName).Chunk(0).(*array.Float64)
	s.InDelta(0.9, scores0.Value(0), 1e-9)
	s.InDelta(0.5, scores0.Value(1), 1e-9)
	s.InDelta(0.1, scores0.Value(2), 1e-9)

	// Chunk 1 sorted desc: 0.8, 0.6, 0.2
	scores1 := result.Column(types.ScoreFieldName).Chunk(1).(*array.Float64)
	s.InDelta(0.8, scores1.Value(0), 1e-9)
	s.InDelta(0.6, scores1.Value(1), 1e-9)
	s.InDelta(0.2, scores1.Value(2), 1e-9)
}

func (s *SortOpTestSuite) TestSortColumnNotFound() {
	df := s.createSortTestDF([]int64{1}, []float64{0.5}, []int64{1})
	defer df.Release()

	op := newSortOp("nonexistent", true, types.IDFieldName)
	ctx := types.NewFuncContextFull(context.TODO(), s.pool, "rerank")
	_, err := op.Execute(ctx, df)
	s.Error(err)
	s.Contains(err.Error(), "not found")
}

func (s *SortOpTestSuite) TestSortEmptyChunk() {
	builder := NewDataFrameBuilder()
	builder.SetChunkSizes([]int64{0})

	idBuilder := array.NewInt64Builder(s.pool)
	idChunk := idBuilder.NewArray()
	idBuilder.Release()

	scoreBuilder := array.NewFloat64Builder(s.pool)
	scoreChunk := scoreBuilder.NewArray()
	scoreBuilder.Release()

	err := builder.AddColumnFromChunks(types.IDFieldName, []arrow.Array{idChunk})
	s.Require().NoError(err)
	err = builder.AddColumnFromChunks(types.ScoreFieldName, []arrow.Array{scoreChunk})
	s.Require().NoError(err)

	df := builder.Build()
	defer df.Release()

	op := newSortOp(types.ScoreFieldName, true, types.IDFieldName)
	ctx := types.NewFuncContextFull(context.TODO(), s.pool, "rerank")
	result, err := op.Execute(ctx, df)
	s.Require().NoError(err)
	defer result.Release()

	s.Equal(int64(0), result.NumRows())
}

func (s *SortOpTestSuite) TestSortInputsIncludeTieBreak() {
	op := newSortOp(types.ScoreFieldName, true, types.IDFieldName)
	s.Equal([]string{types.ScoreFieldName, types.IDFieldName}, op.Inputs())
	s.Empty(op.Outputs())
}

func (s *SortOpTestSuite) TestSortFromReprUsesInputsAndDefaultsTieBreak() {
	op, err := NewSortOpFromRepr(&OperatorRepr{
		Inputs: []string{types.ScoreFieldName},
		Params: map[string]*schemapb.FunctionParamValue{
			"desc": {Value: &schemapb.FunctionParamValue_BoolValue{BoolValue: true}},
		},
	})
	s.Require().NoError(err)
	s.Equal([]string{types.ScoreFieldName, types.IDFieldName}, op.Inputs())
	s.Equal("Sort($score DESC, $id ASC)", op.String())

	op, err = NewSortOpFromRepr(&OperatorRepr{
		Inputs: []string{types.ScoreFieldName, "tie"},
	})
	s.Require().NoError(err)
	s.Equal([]string{types.ScoreFieldName, "tie"}, op.Inputs())
	s.Equal("Sort($score ASC, tie ASC)", op.String())
}

func (s *SortOpTestSuite) TestSortFromReprAcceptsExactPyMilvusLegacyWire() {
	opRaw, err := NewSortOpFromRepr(&OperatorRepr{
		Inputs: []string{types.ScoreFieldName, types.IDFieldName},
		Params: map[string]*schemapb.FunctionParamValue{
			"column":        stringParam(types.ScoreFieldName),
			"desc":          boolParam(true),
			"tie_break_col": stringParam(types.IDFieldName),
		},
	})
	s.Require().NoError(err)
	op := opRaw.(*SortOp)
	s.Equal([]SortKey{
		{Column: types.ScoreFieldName, Descending: true, NullsFirst: false},
		{Column: types.IDFieldName, Descending: false, NullsFirst: false},
	}, op.Keys())
	s.True(op.Stable())
	s.Equal("Sort($score DESC, $id ASC)", op.String())
}

func (s *SortOpTestSuite) TestSortFromReprLegacyInputsAreCanonical() {
	opRaw, err := NewSortOpFromRepr(&OperatorRepr{
		Inputs: []string{types.ScoreFieldName, types.IDFieldName},
		Params: map[string]*schemapb.FunctionParamValue{
			// PyMilvus duplicates these values in params. The declarative inputs
			// remain authoritative for backward compatibility.
			"column":        stringParam("ignored_column"),
			"desc":          boolParam(false),
			"tie_break_col": stringParam("ignored_tie_break"),
			"stable":        boolParam(false),
		},
	})
	s.Require().NoError(err)
	op := opRaw.(*SortOp)
	s.Equal([]SortKey{
		{Column: types.ScoreFieldName, Descending: false, NullsFirst: false},
		{Column: types.IDFieldName, Descending: false, NullsFirst: false},
	}, op.Keys())
	s.False(op.Stable())
}

func (s *SortOpTestSuite) TestSortFromReprRejectsInvalidInputs() {
	_, err := NewSortOp(nil, true)
	s.Require().Error(err)
	s.Contains(err.Error(), "at least one sort key")

	_, err = NewSortOp([]SortKey{{Column: "  "}}, true)
	s.Require().Error(err)
	s.Contains(err.Error(), "column is empty")

	_, err = NewSortOpFromRepr(&OperatorRepr{})
	s.Require().Error(err)
	s.Contains(err.Error(), "column is required")

	_, err = NewSortOpFromRepr(&OperatorRepr{Inputs: []string{"score", "tie", "extra"}})
	s.Require().Error(err)
	s.Contains(err.Error(), "legacy format expects at most 2 input columns")
}

func (s *SortOpTestSuite) TestSortRejectsNonComparableKey() {
	builder := NewDataFrameBuilder()
	builder.SetChunkSizes([]int64{2})

	idBuilder := array.NewInt64Builder(s.pool)
	idBuilder.AppendValues([]int64{1, 2}, nil)
	idChunk := idBuilder.NewArray()
	idBuilder.Release()

	listBuilder := array.NewListBuilder(s.pool, arrow.PrimitiveTypes.Int64)
	valueBuilder := listBuilder.ValueBuilder().(*array.Int64Builder)
	listBuilder.Append(true)
	valueBuilder.Append(1)
	listBuilder.Append(true)
	valueBuilder.Append(2)
	listChunk := listBuilder.NewArray()
	listBuilder.Release()

	s.Require().NoError(builder.AddColumnFromChunks(types.IDFieldName, []arrow.Array{idChunk}))
	s.Require().NoError(builder.AddColumnFromChunks("items", []arrow.Array{listChunk}))
	df := builder.Build()
	defer df.Release()

	op, err := NewSortOp([]SortKey{{Column: "items"}}, true)
	s.Require().NoError(err)
	ctx := types.NewFuncContextFull(context.TODO(), s.pool, types.StagePostProcess)
	_, err = op.Execute(ctx, df)
	s.Require().Error(err)
	s.Contains(err.Error(), "non-comparable type")
}

func (s *SortOpTestSuite) TestSortFromReprMultiKey() {
	opRaw, err := NewSortOpFromRepr(&OperatorRepr{
		Inputs: []string{"price", "rating", "active"},
		Params: map[string]*schemapb.FunctionParamValue{
			"orders":      sortStringArrayParam("asc", " DESC ", "asc"),
			"null_orders": sortStringArrayParam("nulls_last", "nulls_first", "nulls_last"),
			"stable":      boolParam(false),
		},
	})
	s.Require().NoError(err)
	op := opRaw.(*SortOp)
	s.Equal([]string{"price", "rating", "active"}, op.Inputs())
	s.Equal([]SortKey{
		{Column: "price", Descending: false, NullsFirst: false},
		{Column: "rating", Descending: true, NullsFirst: true},
		{Column: "active", Descending: false, NullsFirst: false},
	}, op.Keys())
	s.False(op.Stable())
	s.Equal("Sort(price ASC, rating DESC, active ASC)", op.String())
}

func (s *SortOpTestSuite) TestSortFromReprDefaultNullOrdering() {
	opRaw, err := NewSortOpFromRepr(&OperatorRepr{
		Inputs: []string{"price", "rating"},
		Params: map[string]*schemapb.FunctionParamValue{
			"orders": sortStringArrayParam("asc", "desc"),
		},
	})
	s.Require().NoError(err)
	op := opRaw.(*SortOp)
	s.Equal([]SortKey{
		{Column: "price", Descending: false, NullsFirst: false},
		{Column: "rating", Descending: true, NullsFirst: true},
	}, op.Keys())
	s.True(op.Stable())
}

func (s *SortOpTestSuite) TestSortFromReprRejectsInvalidMultiKeyParams() {
	tests := []struct {
		name        string
		repr        *OperatorRepr
		errContains string
	}{
		{name: "nil repr", repr: nil, errContains: "representation is nil"},
		{
			name: "legacy desc type",
			repr: &OperatorRepr{
				Inputs: []string{"price"},
				Params: map[string]*schemapb.FunctionParamValue{"desc": stringParam("true")},
			},
			errContains: "parameter \"desc\" must be a bool",
		},
		{
			name: "legacy desc unset",
			repr: &OperatorRepr{
				Inputs: []string{"price"},
				Params: map[string]*schemapb.FunctionParamValue{"desc": nil},
			},
			errContains: "parameter \"desc\" is unset",
		},
		{
			name: "legacy stable type",
			repr: &OperatorRepr{
				Inputs: []string{"price"},
				Params: map[string]*schemapb.FunctionParamValue{"stable": stringParam("true")},
			},
			errContains: "parameter \"stable\" must be a bool",
		},
		{
			name: "orders and desc",
			repr: &OperatorRepr{
				Inputs: []string{"price"},
				Params: map[string]*schemapb.FunctionParamValue{
					"orders": sortStringArrayParam("asc"),
					"desc":   boolParam(true),
				},
			},
			errContains: "cannot be used together",
		},
		{
			name: "orders unset",
			repr: &OperatorRepr{
				Inputs: []string{"price"},
				Params: map[string]*schemapb.FunctionParamValue{"orders": nil},
			},
			errContains: "parameter \"orders\" is unset",
		},
		{
			name: "orders scalar type",
			repr: &OperatorRepr{
				Inputs: []string{"price"},
				Params: map[string]*schemapb.FunctionParamValue{"orders": stringParam("asc")},
			},
			errContains: "parameter \"orders\" must be a string array",
		},
		{
			name: "orders item type",
			repr: &OperatorRepr{
				Inputs: []string{"price", "rating"},
				Params: map[string]*schemapb.FunctionParamValue{
					"orders": sortParamArray(stringParam("asc"), boolParam(true)),
				},
			},
			errContains: "parameter \"orders\"[1] must be a string",
		},
		{
			name: "orders item unset",
			repr: &OperatorRepr{
				Inputs: []string{"price"},
				Params: map[string]*schemapb.FunctionParamValue{
					"orders": sortParamArray(nil),
				},
			},
			errContains: "parameter \"orders\"[0] must be a string",
		},
		{
			name: "orders count",
			repr: &OperatorRepr{
				Inputs: []string{"price", "rating"},
				Params: map[string]*schemapb.FunctionParamValue{"orders": sortStringArrayParam("asc")},
			},
			errContains: "orders count (1) must match inputs count (2)",
		},
		{
			name: "invalid order",
			repr: &OperatorRepr{
				Inputs: []string{"price"},
				Params: map[string]*schemapb.FunctionParamValue{"orders": sortStringArrayParam("sideways")},
			},
			errContains: "orders[0] must be",
		},
		{
			name: "null orders require orders",
			repr: &OperatorRepr{
				Inputs: []string{"price"},
				Params: map[string]*schemapb.FunctionParamValue{"null_orders": sortStringArrayParam("nulls_last")},
			},
			errContains: "requires \"orders\"",
		},
		{
			name: "null orders unset",
			repr: &OperatorRepr{
				Inputs: []string{"price"},
				Params: map[string]*schemapb.FunctionParamValue{
					"orders":      sortStringArrayParam("asc"),
					"null_orders": {},
				},
			},
			errContains: "parameter \"null_orders\" is unset",
		},
		{
			name: "null orders scalar type",
			repr: &OperatorRepr{
				Inputs: []string{"price"},
				Params: map[string]*schemapb.FunctionParamValue{
					"orders":      sortStringArrayParam("asc"),
					"null_orders": stringParam("nulls_last"),
				},
			},
			errContains: "parameter \"null_orders\" must be a string array",
		},
		{
			name: "null orders item unset",
			repr: &OperatorRepr{
				Inputs: []string{"price"},
				Params: map[string]*schemapb.FunctionParamValue{
					"orders":      sortStringArrayParam("asc"),
					"null_orders": sortParamArray(nil),
				},
			},
			errContains: "parameter \"null_orders\"[0] must be a string",
		},
		{
			name: "null orders count",
			repr: &OperatorRepr{
				Inputs: []string{"price", "rating"},
				Params: map[string]*schemapb.FunctionParamValue{
					"orders":      sortStringArrayParam("asc", "desc"),
					"null_orders": sortStringArrayParam("nulls_last"),
				},
			},
			errContains: "null_orders count (1) must match inputs count (2)",
		},
		{
			name: "invalid null order",
			repr: &OperatorRepr{
				Inputs: []string{"price"},
				Params: map[string]*schemapb.FunctionParamValue{
					"orders":      sortStringArrayParam("asc"),
					"null_orders": sortStringArrayParam("middle"),
				},
			},
			errContains: "null_orders[0] must be",
		},
		{
			name: "stable type",
			repr: &OperatorRepr{
				Inputs: []string{"price"},
				Params: map[string]*schemapb.FunctionParamValue{
					"orders": sortStringArrayParam("asc"),
					"stable": stringParam("true"),
				},
			},
			errContains: "parameter \"stable\" must be a bool",
		},
	}

	for _, test := range tests {
		s.Run(test.name, func() {
			_, err := NewSortOpFromRepr(test.repr)
			s.Require().Error(err)
			s.ErrorIs(err, merr.ErrParameterInvalid)
			s.Contains(err.Error(), test.errContains)
		})
	}
}

func (s *SortOpTestSuite) TestSortMultiKeyMixedDirectionNullAndBool() {
	builder := NewDataFrameBuilder()
	builder.SetChunkSizes([]int64{6})
	builder.SetMetricType("COSINE")

	idBuilder := array.NewInt64Builder(s.pool)
	idBuilder.AppendValues([]int64{1, 2, 3, 4, 5, 6}, nil)
	idChunk := idBuilder.NewArray()
	idBuilder.Release()

	priceBuilder := array.NewInt64Builder(s.pool)
	priceBuilder.Append(10)
	priceBuilder.Append(10)
	priceBuilder.Append(10)
	priceBuilder.AppendNull()
	priceBuilder.Append(5)
	priceBuilder.Append(5)
	priceChunk := priceBuilder.NewArray()
	priceBuilder.Release()

	ratingBuilder := array.NewFloat64Builder(s.pool)
	ratingBuilder.Append(1)
	ratingBuilder.Append(3)
	ratingBuilder.AppendNull()
	ratingBuilder.Append(9)
	ratingBuilder.Append(2)
	ratingBuilder.Append(2)
	ratingChunk := ratingBuilder.NewArray()
	ratingBuilder.Release()

	activeBuilder := array.NewBooleanBuilder(s.pool)
	activeBuilder.AppendValues([]bool{true, false, true, false, true, false}, nil)
	activeChunk := activeBuilder.NewArray()
	activeBuilder.Release()

	s.Require().NoError(builder.AddColumnFromChunks(types.IDFieldName, []arrow.Array{idChunk}))
	s.Require().NoError(builder.AddColumnFromChunks("price", []arrow.Array{priceChunk}))
	s.Require().NoError(builder.AddColumnFromChunks("rating", []arrow.Array{ratingChunk}))
	s.Require().NoError(builder.AddColumnFromChunks("active", []arrow.Array{activeChunk}))
	df := builder.Build()
	defer df.Release()

	op, err := NewSortOp([]SortKey{
		{Column: "price", NullsFirst: false},
		{Column: "rating", Descending: true, NullsFirst: true},
		{Column: "active", NullsFirst: false},
	}, true)
	s.Require().NoError(err)
	ctx := types.NewFuncContextFull(context.TODO(), s.pool, types.StagePostProcess)
	result, err := op.Execute(ctx, df)
	s.Require().NoError(err)
	defer result.Release()

	ids := result.Column(types.IDFieldName).Chunk(0).(*array.Int64)
	s.Equal([]int64{6, 5, 3, 2, 1, 4}, ids.Int64Values())
	metricType, ok := result.MetricType()
	s.True(ok)
	s.Equal("COSINE", metricType)
}

func (s *SortOpTestSuite) TestSortReordersNestedPayloadColumn() {
	highlightType := testHighlightArrowType()
	highlights, _, err := array.FromJSON(s.pool, highlightType, strings.NewReader(
		`[[{"field_name":"a","fragments":["a1"],"scores":[]}], null, [{"field_name":"c","fragments":["c1"],"scores":[0.9]}]]`,
	))
	s.Require().NoError(err)

	builder := NewDataFrameBuilder().SetChunkSizes([]int64{3})
	idBuilder := array.NewInt64Builder(s.pool)
	idBuilder.AppendValues([]int64{1, 2, 3}, nil)
	ids := idBuilder.NewArray()
	idBuilder.Release()
	keyBuilder := array.NewInt64Builder(s.pool)
	keyBuilder.AppendValues([]int64{30, 10, 20}, nil)
	keys := keyBuilder.NewArray()
	keyBuilder.Release()
	s.Require().NoError(builder.AddColumnFromChunks(types.IDFieldName, []arrow.Array{ids}))
	s.Require().NoError(builder.AddColumnFromChunks("key", []arrow.Array{keys}))
	s.Require().NoError(builder.AddColumnFromChunks("highlight", []arrow.Array{highlights}))
	df := builder.Build()
	defer df.Release()

	op, err := NewSortOp([]SortKey{{Column: "key"}}, true)
	s.Require().NoError(err)
	ctx := types.NewFuncContextFull(context.TODO(), s.pool, types.StagePostProcess)
	result, err := op.Execute(ctx, df)
	s.Require().NoError(err)
	defer result.Release()

	resultIDs := result.Column(types.IDFieldName).Chunk(0).(*array.Int64)
	s.Equal([]int64{2, 3, 1}, resultIDs.Int64Values())
	expected, _, err := array.FromJSON(s.pool, highlightType, strings.NewReader(
		`[null, [{"field_name":"c","fragments":["c1"],"scores":[0.9]}], [{"field_name":"a","fragments":["a1"],"scores":[]}]]`,
	))
	s.Require().NoError(err)
	defer expected.Release()
	s.True(array.Equal(expected, result.Column("highlight").Chunk(0)))
}

func (s *SortOpTestSuite) TestSortReordersNestedPayloadAcrossChunks() {
	highlightType := testHighlightArrowType()
	highlight0, _, err := array.FromJSON(s.pool, highlightType, strings.NewReader(
		`[[{"field_name":"a","fragments":["a1"],"scores":[]}], [{"field_name":"b","fragments":["b1"],"scores":[]}]]`,
	))
	s.Require().NoError(err)
	highlight1, _, err := array.FromJSON(s.pool, highlightType, strings.NewReader(
		`[[{"field_name":"c","fragments":["c1"],"scores":[]}], [{"field_name":"d","fragments":["d1"],"scores":[]}]]`,
	))
	s.Require().NoError(err)

	builder := NewDataFrameBuilder().SetChunkSizes([]int64{2, 2})
	id0 := array.NewInt64Builder(s.pool)
	id0.AppendValues([]int64{1, 2}, nil)
	id1 := array.NewInt64Builder(s.pool)
	id1.AppendValues([]int64{3, 4}, nil)
	key0 := array.NewInt64Builder(s.pool)
	key0.AppendValues([]int64{20, 10}, nil)
	key1 := array.NewInt64Builder(s.pool)
	key1.AppendValues([]int64{40, 30}, nil)
	s.Require().NoError(builder.AddColumnFromChunks(types.IDFieldName, []arrow.Array{id0.NewArray(), id1.NewArray()}))
	s.Require().NoError(builder.AddColumnFromChunks("key", []arrow.Array{key0.NewArray(), key1.NewArray()}))
	id0.Release()
	id1.Release()
	key0.Release()
	key1.Release()
	s.Require().NoError(builder.AddColumnFromChunks("highlight", []arrow.Array{highlight0, highlight1}))
	df := builder.Build()
	defer df.Release()

	op, err := NewSortOp([]SortKey{{Column: "key"}}, true)
	s.Require().NoError(err)
	ctx := types.NewFuncContextFull(context.TODO(), s.pool, types.StagePostProcess)
	result, err := op.Execute(ctx, df)
	s.Require().NoError(err)
	defer result.Release()

	s.Equal([]int64{2, 1}, result.Column(types.IDFieldName).Chunk(0).(*array.Int64).Int64Values())
	s.Equal([]int64{4, 3}, result.Column(types.IDFieldName).Chunk(1).(*array.Int64).Int64Values())
	for chunkIdx, expectedJSON := range []string{
		`[[{"field_name":"b","fragments":["b1"],"scores":[]}], [{"field_name":"a","fragments":["a1"],"scores":[]}]]`,
		`[[{"field_name":"d","fragments":["d1"],"scores":[]}], [{"field_name":"c","fragments":["c1"],"scores":[]}]]`,
	} {
		expected, _, err := array.FromJSON(s.pool, highlightType, strings.NewReader(expectedJSON))
		s.Require().NoError(err)
		s.True(array.Equal(expected, result.Column("highlight").Chunk(chunkIdx)))
		expected.Release()
	}
}

func (s *SortOpTestSuite) TestSortExplicitKeyIsStableWithoutImplicitID() {
	df := s.createSortTestDF(
		[]int64{3, 1, 2},
		[]float64{0.5, 0.5, 0.5},
		[]int64{3},
	)
	defer df.Release()

	op, err := NewSortOp([]SortKey{{Column: types.ScoreFieldName}}, true)
	s.Require().NoError(err)
	s.Equal([]string{types.ScoreFieldName}, op.Inputs())
	ctx := types.NewFuncContextFull(context.TODO(), s.pool, types.StagePostProcess)
	result, err := op.Execute(ctx, df)
	s.Require().NoError(err)
	defer result.Release()

	ids := result.Column(types.IDFieldName).Chunk(0).(*array.Int64)
	s.Equal([]int64{3, 1, 2}, ids.Int64Values())
}

func (s *SortOpTestSuite) TestSortTieBreakByID() {
	// All scores are equal (0.5), IDs are 5, 3, 1, 4, 2
	// After sort descending by score with tie-break by $id ascending:
	// expected order by ID: 1, 2, 3, 4, 5
	df := s.createSortTestDF(
		[]int64{5, 3, 1, 4, 2},
		[]float64{0.5, 0.5, 0.5, 0.5, 0.5},
		[]int64{5},
	)
	defer df.Release()

	op := newSortOp(types.ScoreFieldName, true, types.IDFieldName)
	ctx := types.NewFuncContextFull(context.TODO(), s.pool, "rerank")
	result, err := op.Execute(ctx, df)
	s.Require().NoError(err)
	defer result.Release()

	idCol := result.Column(types.IDFieldName)
	ids := idCol.Chunk(0).(*array.Int64)
	s.Equal(int64(1), ids.Value(0))
	s.Equal(int64(2), ids.Value(1))
	s.Equal(int64(3), ids.Value(2))
	s.Equal(int64(4), ids.Value(3))
	s.Equal(int64(5), ids.Value(4))
}

func (s *SortOpTestSuite) TestSortTieBreakPartialTies() {
	// Scores: 0.9, 0.5, 0.5, 0.5, 0.1 with IDs: 10, 30, 20, 40, 50
	// Expected: ID 10 (0.9), then 20, 30, 40 (0.5 sorted by ID asc), then 50 (0.1)
	df := s.createSortTestDF(
		[]int64{10, 30, 20, 40, 50},
		[]float64{0.9, 0.5, 0.5, 0.5, 0.1},
		[]int64{5},
	)
	defer df.Release()

	op := newSortOp(types.ScoreFieldName, true, types.IDFieldName)
	ctx := types.NewFuncContextFull(context.TODO(), s.pool, "rerank")
	result, err := op.Execute(ctx, df)
	s.Require().NoError(err)
	defer result.Release()

	idCol := result.Column(types.IDFieldName)
	ids := idCol.Chunk(0).(*array.Int64)
	s.Equal(int64(10), ids.Value(0)) // score 0.9
	s.Equal(int64(20), ids.Value(1)) // score 0.5, smallest ID
	s.Equal(int64(30), ids.Value(2)) // score 0.5
	s.Equal(int64(40), ids.Value(3)) // score 0.5
	s.Equal(int64(50), ids.Value(4)) // score 0.1
}

func (s *SortOpTestSuite) TestSortFastPathFloat32DescWithInt64TieBreak() {
	builder := NewDataFrameBuilder()
	builder.SetChunkSizes([]int64{4})

	idBuilder := array.NewInt64Builder(s.pool)
	idBuilder.AppendValues([]int64{3, 2, 1, 4}, nil)
	idChunk := idBuilder.NewArray()
	idBuilder.Release()

	scoreBuilder := array.NewFloat32Builder(s.pool)
	scoreBuilder.AppendValues([]float32{0.5, 0.9, 0.5, 0.7}, nil)
	scoreChunk := scoreBuilder.NewArray()
	scoreBuilder.Release()

	s.Require().NoError(builder.AddColumnFromChunks(types.IDFieldName, []arrow.Array{idChunk}))
	s.Require().NoError(builder.AddColumnFromChunks(types.ScoreFieldName, []arrow.Array{scoreChunk}))
	df := builder.Build()
	defer df.Release()

	op := newSortOp(types.ScoreFieldName, true, types.IDFieldName)
	ctx := types.NewFuncContextFull(context.TODO(), s.pool, "rerank")
	result, err := op.Execute(ctx, df)
	s.Require().NoError(err)
	defer result.Release()

	ids := result.Column(types.IDFieldName).Chunk(0).(*array.Int64)
	s.Equal(int64(2), ids.Value(0))
	s.Equal(int64(4), ids.Value(1))
	s.Equal(int64(1), ids.Value(2))
	s.Equal(int64(3), ids.Value(3))
}

func (s *SortOpTestSuite) TestSortBooleanTieBreakAndStringHelpers() {
	builder := NewDataFrameBuilder()
	builder.SetChunkSizes([]int64{3})

	idBuilder := array.NewInt64Builder(s.pool)
	idBuilder.AppendValues([]int64{3, 2, 1}, nil)
	idChunk := idBuilder.NewArray()
	idBuilder.Release()

	scoreBuilder := array.NewFloat64Builder(s.pool)
	scoreBuilder.AppendValues([]float64{0.5, 0.5, 0.5}, nil)
	scoreChunk := scoreBuilder.NewArray()
	scoreBuilder.Release()

	boolBuilder := array.NewBooleanBuilder(s.pool)
	boolBuilder.AppendValues([]bool{true, false, true}, nil)
	boolChunk := boolBuilder.NewArray()
	boolBuilder.Release()

	s.Require().NoError(builder.AddColumnFromChunks(types.IDFieldName, []arrow.Array{idChunk}))
	s.Require().NoError(builder.AddColumnFromChunks(types.ScoreFieldName, []arrow.Array{scoreChunk}))
	s.Require().NoError(builder.AddColumnFromChunks("bool_tie", []arrow.Array{boolChunk}))
	df := builder.Build()
	defer df.Release()

	op := newSortOp(types.ScoreFieldName, true, "bool_tie")
	s.Equal("Sort($score DESC, bool_tie ASC)", op.String())
	ctx := types.NewFuncContextFull(context.TODO(), s.pool, "rerank")
	result, err := op.Execute(ctx, df)
	s.Require().NoError(err)
	defer result.Release()

	ids := result.Column(types.IDFieldName).Chunk(0).(*array.Int64)
	s.Equal(int64(2), ids.Value(0))
	s.Equal(int64(3), ids.Value(1))
	s.Equal(int64(1), ids.Value(2))

	emptyOp := &SortOp{}
	s.Empty(emptyOp.Column())
}

func sortStringArrayParam(values ...string) *schemapb.FunctionParamValue {
	items := make([]*schemapb.FunctionParamValue, len(values))
	for i, value := range values {
		items[i] = stringParam(value)
	}
	return sortParamArray(items...)
}

func sortParamArray(items ...*schemapb.FunctionParamValue) *schemapb.FunctionParamValue {
	return &schemapb.FunctionParamValue{
		Value: &schemapb.FunctionParamValue_ArrayValue{
			ArrayValue: &schemapb.FunctionParamArray{Values: items},
		},
	}
}
