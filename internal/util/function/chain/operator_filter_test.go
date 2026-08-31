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

	"github.com/milvus-io/milvus/internal/util/function/chain/types"
)

type FilterOpTestSuite struct {
	suite.Suite
	pool *memory.CheckedAllocator
}

func (s *FilterOpTestSuite) SetupTest() {
	s.pool = memory.NewCheckedAllocator(memory.NewGoAllocator())
}

func (s *FilterOpTestSuite) TearDownTest() {
	s.pool.AssertSize(s.T(), 0)
}

func TestFilterOpTestSuite(t *testing.T) {
	suite.Run(t, new(FilterOpTestSuite))
}

// alwaysFalseExpr is a FunctionExpr that returns all-false boolean column.
type alwaysFalseExpr struct{}

func (e *alwaysFalseExpr) Name() string { return "always_false" }
func (e *alwaysFalseExpr) Execute(ctx *types.FuncContext, inputs []*arrow.Chunked) ([]*arrow.Chunked, error) {
	input := inputs[0]
	chunks := make([]arrow.Array, len(input.Chunks()))
	for i, chunk := range input.Chunks() {
		b := array.NewBooleanBuilder(ctx.Pool())
		for j := 0; j < chunk.Len(); j++ {
			b.Append(false)
		}
		chunks[i] = b.NewArray()
		b.Release()
	}
	result := arrow.NewChunked(arrow.FixedWidthTypes.Boolean, chunks)
	for _, c := range chunks {
		c.Release()
	}
	return []*arrow.Chunked{result}, nil
}

func (e *alwaysFalseExpr) OutputDataTypes() []arrow.DataType {
	return []arrow.DataType{arrow.FixedWidthTypes.Boolean}
}
func (e *alwaysFalseExpr) IsRunnable(stage string) bool { return true }
func (e *alwaysFalseExpr) Stages() []string             { return []string{"rerank"} }

// alwaysTrueExpr is a FunctionExpr that returns all-true boolean column.
type alwaysTrueExpr struct{}

func (e *alwaysTrueExpr) Name() string { return "always_true" }
func (e *alwaysTrueExpr) Execute(ctx *types.FuncContext, inputs []*arrow.Chunked) ([]*arrow.Chunked, error) {
	input := inputs[0]
	chunks := make([]arrow.Array, len(input.Chunks()))
	for i, chunk := range input.Chunks() {
		b := array.NewBooleanBuilder(ctx.Pool())
		for j := 0; j < chunk.Len(); j++ {
			b.Append(true)
		}
		chunks[i] = b.NewArray()
		b.Release()
	}
	result := arrow.NewChunked(arrow.FixedWidthTypes.Boolean, chunks)
	for _, c := range chunks {
		c.Release()
	}
	return []*arrow.Chunked{result}, nil
}

func (e *alwaysTrueExpr) OutputDataTypes() []arrow.DataType {
	return []arrow.DataType{arrow.FixedWidthTypes.Boolean}
}
func (e *alwaysTrueExpr) IsRunnable(stage string) bool { return true }
func (e *alwaysTrueExpr) Stages() []string             { return []string{"rerank"} }

type malformedFilterExpr struct {
	kind string
}

func (e *malformedFilterExpr) Name() string { return "malformed_filter" }

func (e *malformedFilterExpr) Execute(ctx *types.FuncContext, _ []*arrow.Chunked) ([]*arrow.Chunked, error) {
	switch e.kind {
	case "nil":
		return []*arrow.Chunked{nil}, nil
	case "chunk_count":
		return []*arrow.Chunked{arrow.NewChunked(arrow.FixedWidthTypes.Boolean, nil)}, nil
	case "chunk_length":
		builder := array.NewBooleanBuilder(ctx.Pool())
		builder.Append(true)
		chunk := builder.NewArray()
		builder.Release()
		result := arrow.NewChunked(arrow.FixedWidthTypes.Boolean, []arrow.Array{chunk})
		chunk.Release()
		return []*arrow.Chunked{result}, nil
	default:
		return nil, nil
	}
}

func (e *malformedFilterExpr) OutputDataTypes() []arrow.DataType {
	return []arrow.DataType{arrow.FixedWidthTypes.Boolean}
}

func (e *malformedFilterExpr) IsRunnable(stage string) bool { return true }
func (e *malformedFilterExpr) Stages() []string             { return []string{"rerank"} }

type evenIDFilterExpr struct{}

func (e *evenIDFilterExpr) Name() string { return "even_id" }

func (e *evenIDFilterExpr) Execute(ctx *types.FuncContext, inputs []*arrow.Chunked) ([]*arrow.Chunked, error) {
	chunks := make([]arrow.Array, len(inputs[0].Chunks()))
	for chunkIdx, chunk := range inputs[0].Chunks() {
		ids := chunk.(*array.Int64)
		builder := array.NewBooleanBuilder(ctx.Pool())
		for rowIdx := 0; rowIdx < ids.Len(); rowIdx++ {
			builder.Append(ids.Value(rowIdx)%2 == 0)
		}
		chunks[chunkIdx] = builder.NewArray()
		builder.Release()
	}
	result := arrow.NewChunked(arrow.FixedWidthTypes.Boolean, chunks)
	for _, chunk := range chunks {
		chunk.Release()
	}
	return []*arrow.Chunked{result}, nil
}

func (e *evenIDFilterExpr) OutputDataTypes() []arrow.DataType {
	return []arrow.DataType{arrow.FixedWidthTypes.Boolean}
}

func (e *evenIDFilterExpr) IsRunnable(stage string) bool { return true }
func (e *evenIDFilterExpr) Stages() []string             { return []string{"rerank"} }

func (s *FilterOpTestSuite) createFilterTestDF(ids []int64, scores []float64) *DataFrame {
	builder := NewDataFrameBuilder()
	builder.SetChunkSizes([]int64{int64(len(ids))})

	idBuilder := array.NewInt64Builder(s.pool)
	idBuilder.AppendValues(ids, nil)
	idChunk := idBuilder.NewArray()
	idBuilder.Release()

	scoreBuilder := array.NewFloat64Builder(s.pool)
	scoreBuilder.AppendValues(scores, nil)
	scoreChunk := scoreBuilder.NewArray()
	scoreBuilder.Release()

	err := builder.AddColumnFromChunks(types.IDFieldName, []arrow.Array{idChunk})
	s.Require().NoError(err)
	err = builder.AddColumnFromChunks(types.ScoreFieldName, []arrow.Array{scoreChunk})
	s.Require().NoError(err)

	return builder.Build()
}

func (s *FilterOpTestSuite) TestFilterAllFalse() {
	df := s.createFilterTestDF([]int64{1, 2, 3}, []float64{0.9, 0.8, 0.7})
	defer df.Release()

	op, err := NewFilterOp(&alwaysFalseExpr{}, []string{types.IDFieldName})
	s.Require().NoError(err)

	ctx := types.NewFuncContextFull(context.TODO(), s.pool, "rerank")
	result, err := op.Execute(ctx, df)
	s.Require().NoError(err)
	defer result.Release()

	// All filtered out
	s.Equal(int64(0), result.NumRows())
}

func (s *FilterOpTestSuite) TestFilterAllTrue() {
	df := s.createFilterTestDF([]int64{1, 2, 3}, []float64{0.9, 0.8, 0.7})
	defer df.Release()

	op, err := NewFilterOp(&alwaysTrueExpr{}, []string{types.IDFieldName})
	s.Require().NoError(err)

	ctx := types.NewFuncContextFull(context.TODO(), s.pool, "rerank")
	result, err := op.Execute(ctx, df)
	s.Require().NoError(err)
	defer result.Release()

	// Nothing filtered
	s.Equal(int64(3), result.NumRows())
}

func (s *FilterOpTestSuite) TestFilterPreservesGlobalMetadata() {
	df := s.createFilterTestDF([]int64{1, 2}, []float64{0.9, 0.8})
	defer df.Release()
	df.metadata[types.MetadataKeyMetricType] = "COSINE"
	df.metadata["custom"] = "value"

	op, err := NewFilterOp(&alwaysTrueExpr{}, []string{types.IDFieldName})
	s.Require().NoError(err)
	result, err := op.Execute(types.NewFuncContextFull(context.TODO(), s.pool, "rerank"), df)
	s.Require().NoError(err)
	defer result.Release()
	metricType, ok := result.MetricType()
	s.True(ok)
	s.Equal("COSINE", metricType)
	custom, ok := result.Metadata("custom")
	s.True(ok)
	s.Equal("value", custom)
}

func (s *FilterOpTestSuite) TestFilterNilFunction() {
	_, err := NewFilterOp(nil, []string{types.IDFieldName})
	s.Error(err)
	s.Contains(err.Error(), "function is nil")
}

func (s *FilterOpTestSuite) TestFilterColumnNotFound() {
	df := s.createFilterTestDF([]int64{1}, []float64{0.5})
	defer df.Release()

	op, err := NewFilterOp(&alwaysTrueExpr{}, []string{"nonexistent"})
	s.Require().NoError(err)

	ctx := types.NewFuncContextFull(context.TODO(), s.pool, "rerank")
	_, err = op.Execute(ctx, df)
	s.Error(err)
	s.Contains(err.Error(), "not found")
}

func (s *FilterOpTestSuite) TestFilterRejectsMalformedFunctionOutputShape() {
	for _, test := range []struct {
		name        string
		kind        string
		errContains string
	}{
		{name: "nil output", kind: "nil", errContains: "returned nil output"},
		{name: "chunk count", kind: "chunk_count", errContains: "output chunk count 0 does not match input chunk count 1"},
		{name: "chunk length", kind: "chunk_length", errContains: "output chunk[0] length 1 does not match input chunk length 3"},
	} {
		s.Run(test.name, func() {
			df := s.createFilterTestDF([]int64{1, 2, 3}, []float64{0.9, 0.8, 0.7})
			defer df.Release()

			op, err := NewFilterOp(&malformedFilterExpr{kind: test.kind}, []string{types.IDFieldName})
			s.Require().NoError(err)
			ctx := types.NewFuncContextFull(context.TODO(), s.pool, "rerank")
			_, err = op.Execute(ctx, df)
			s.Require().Error(err)
			s.Contains(err.Error(), test.errContains)
		})
	}
}

func (s *FilterOpTestSuite) TestFilterPreservesNestedPayloadAlignment() {
	highlightType := testHighlightArrowType()
	highlights, _, err := array.FromJSON(s.pool, highlightType, strings.NewReader(
		`[[{"field_name":"a","fragments":["a1"],"scores":[]}], [{"field_name":"b","fragments":["b1"],"scores":[]}], null, [{"field_name":"d","fragments":["d1"],"scores":[]}]]`,
	))
	s.Require().NoError(err)

	builder := NewDataFrameBuilder().SetChunkSizes([]int64{4})
	idBuilder := array.NewInt64Builder(s.pool)
	idBuilder.AppendValues([]int64{1, 2, 3, 4}, nil)
	ids := idBuilder.NewArray()
	idBuilder.Release()
	s.Require().NoError(builder.AddColumnFromChunks(types.IDFieldName, []arrow.Array{ids}))
	s.Require().NoError(builder.AddColumnFromChunks("highlight", []arrow.Array{highlights}))
	df := builder.Build()
	defer df.Release()

	op, err := NewFilterOp(&evenIDFilterExpr{}, []string{types.IDFieldName})
	s.Require().NoError(err)
	ctx := types.NewFuncContextFull(context.TODO(), s.pool, types.StagePostProcess)
	result, err := op.Execute(ctx, df)
	s.Require().NoError(err)
	defer result.Release()

	s.Equal([]int64{2, 4}, result.Column(types.IDFieldName).Chunk(0).(*array.Int64).Int64Values())
	expected, _, err := array.FromJSON(s.pool, highlightType, strings.NewReader(
		`[[{"field_name":"b","fragments":["b1"],"scores":[]}], [{"field_name":"d","fragments":["d1"],"scores":[]}]]`,
	))
	s.Require().NoError(err)
	defer expected.Release()
	s.True(array.Equal(expected, result.Column("highlight").Chunk(0)))
}
