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

package tasks

import (
	"context"
	"testing"

	"github.com/apache/arrow/go/v17/arrow"
	"github.com/apache/arrow/go/v17/arrow/array"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus/internal/querynodev2/segments"
	"github.com/milvus-io/milvus/internal/util/function/chain"
	"github.com/milvus-io/milvus/internal/util/function/chain/expr"
	"github.com/milvus-io/milvus/internal/util/function/chain/types"
	"github.com/milvus-io/milvus/internal/util/segcore"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/util/metric"
)

func TestBoostScoreColumn(t *testing.T) {
	require.Equal(t, "$boost_score_0", boostScoreColumn(0))
	require.Equal(t, "$boost_score_3", boostScoreColumn(3))
}

func makeBoostScoreTestDF(t *testing.T, ids []int64, scores []float32, offsets []int64, chunkSizes []int64) *chain.DataFrame {
	builder := chain.NewDataFrameBuilder()
	builder.SetChunkSizes(chunkSizes)

	idChunks := make([]arrow.Array, len(chunkSizes))
	scoreChunks := make([]arrow.Array, len(chunkSizes))
	offsetChunks := make([]arrow.Array, len(chunkSizes))

	pos := 0
	for chunkIdx, size := range chunkSizes {
		idBuilder := array.NewInt64Builder(defaultAllocator)
		scoreBuilder := array.NewFloat32Builder(defaultAllocator)
		offsetBuilder := array.NewInt64Builder(defaultAllocator)
		for rowIdx := int64(0); rowIdx < size; rowIdx++ {
			idBuilder.Append(ids[pos])
			scoreBuilder.Append(scores[pos])
			offsetBuilder.Append(offsets[pos])
			pos++
		}
		idChunks[chunkIdx] = idBuilder.NewArray()
		scoreChunks[chunkIdx] = scoreBuilder.NewArray()
		offsetChunks[chunkIdx] = offsetBuilder.NewArray()
		idBuilder.Release()
		scoreBuilder.Release()
		offsetBuilder.Release()
	}

	require.NoError(t, builder.AddColumnFromChunks(types.IDFieldName, idChunks))
	require.NoError(t, builder.AddColumnFromChunks(types.ScoreFieldName, scoreChunks))
	require.NoError(t, builder.AddColumnFromChunks(types.SegOffsetFieldName, offsetChunks))
	return builder.Build()
}

func makeBoostScoreTestTask(t *testing.T, plan *planpb.PlanNode) *SearchTask {
	if plan.GetVectorAnns().GetQueryInfo().GetMetricType() == "" {
		plan.Node = &planpb.PlanNode_VectorAnns{
			VectorAnns: &planpb.VectorANNS{
				QueryInfo: &planpb.QueryInfo{MetricType: metric.COSINE},
			},
		}
	}

	blob, err := proto.Marshal(plan)
	require.NoError(t, err)
	return &SearchTask{
		ctx: context.Background(),
		req: &querypb.SearchRequest{
			Req: &internalpb.SearchRequest{
				SerializedExprPlan: blob,
			},
		},
	}
}

type boostScoreOutput struct {
	scores   []float32
	hasScore []bool
}

func mockBoostScoreRunnerFactory(outputs ...boostScoreOutput) func(segments.Segment, *segcore.SearchRequest, *planpb.ScoreFunction) expr.BoostScoreRunner {
	call := 0
	return func(segments.Segment, *segcore.SearchRequest, *planpb.ScoreFunction) expr.BoostScoreRunner {
		idx := call
		call++
		return func(ctx context.Context, offsets *arrow.Chunked) (*arrow.Chunked, error) {
			chunks := make([]arrow.Array, 0, len(offsets.Chunks()))
			pos := 0
			for _, offsetChunk := range offsets.Chunks() {
				builder := array.NewFloat32Builder(defaultAllocator)
				for rowIdx := 0; rowIdx < offsetChunk.Len(); rowIdx++ {
					if outputs[idx].hasScore[pos] {
						builder.Append(outputs[idx].scores[pos])
					} else {
						builder.AppendNull()
					}
					pos++
				}
				chunk := builder.NewArray()
				builder.Release()
				chunks = append(chunks, chunk)
			}
			result := arrow.NewChunked(arrow.PrimitiveTypes.Float32, chunks)
			for _, chunk := range chunks {
				chunk.Release()
			}
			return result, nil
		}
	}
}

func TestApplyBoostScoresSingleScorerCombinesAndSorts(t *testing.T) {
	oldFactory := boostScoreRunnerFactory
	boostScoreRunnerFactory = mockBoostScoreRunnerFactory(boostScoreOutput{
		scores:   []float32{1.0, 10.0, 2.0},
		hasScore: []bool{true, true, false},
	})
	defer func() { boostScoreRunnerFactory = oldFactory }()

	df := makeBoostScoreTestDF(t,
		[]int64{1, 2, 3},
		[]float32{0.5, 0.2, 0.9},
		[]int64{10, 20, 30},
		[]int64{3},
	)
	segDFs := []*chain.DataFrame{df}

	task := makeBoostScoreTestTask(t, &planpb.PlanNode{
		Scorers: []*planpb.ScoreFunction{{Weight: 1}},
		ScoreOption: &planpb.ScoreOption{
			BoostMode: planpb.BoostMode_BoostModeMultiply,
		},
	})

	require.NoError(t, task.applyBoostScores(segDFs, []segments.Segment{nil}, nil))
	defer segDFs[0].Release()

	result := segDFs[0]
	ids := result.Column(types.IDFieldName).Chunk(0).(*array.Int64)
	scores := result.Column(types.ScoreFieldName).Chunk(0).(*array.Float32)
	require.Equal(t, int64(2), ids.Value(0))
	require.InDelta(t, 2.0, scores.Value(0), 1e-6)
	require.Equal(t, int64(3), ids.Value(1))
	require.InDelta(t, 0.9, scores.Value(1), 1e-6)
	require.Equal(t, int64(1), ids.Value(2))
	require.InDelta(t, 0.5, scores.Value(2), 1e-6)
}

func TestApplyBoostScoresMultipleScorersCombinesFunctionScoreAndSorts(t *testing.T) {
	oldFactory := boostScoreRunnerFactory
	boostScoreRunnerFactory = mockBoostScoreRunnerFactory(
		boostScoreOutput{scores: []float32{2.0, 0.0, 3.0}, hasScore: []bool{true, false, true}},
		boostScoreOutput{scores: []float32{4.0, 5.0, 0.0}, hasScore: []bool{true, true, false}},
	)
	defer func() { boostScoreRunnerFactory = oldFactory }()

	df := makeBoostScoreTestDF(t,
		[]int64{1, 2, 3},
		[]float32{0.5, 0.2, 0.9},
		[]int64{10, 20, 30},
		[]int64{3},
	)
	segDFs := []*chain.DataFrame{df}

	task := makeBoostScoreTestTask(t, &planpb.PlanNode{
		Scorers: []*planpb.ScoreFunction{{Weight: 1}, {Weight: 2}},
		ScoreOption: &planpb.ScoreOption{
			FunctionMode: planpb.FunctionMode_FunctionModeSum,
			BoostMode:    planpb.BoostMode_BoostModeSum,
		},
	})

	require.NoError(t, task.applyBoostScores(segDFs, []segments.Segment{nil}, nil))
	defer segDFs[0].Release()

	result := segDFs[0]
	ids := result.Column(types.IDFieldName).Chunk(0).(*array.Int64)
	scores := result.Column(types.ScoreFieldName).Chunk(0).(*array.Float32)
	functionScores := result.Column(functionScoreColumn).Chunk(0).(*array.Float32)
	require.Equal(t, int64(1), ids.Value(0))
	require.InDelta(t, 6.5, scores.Value(0), 1e-6)
	require.InDelta(t, 6.0, functionScores.Value(0), 1e-6)
	require.Equal(t, int64(2), ids.Value(1))
	require.InDelta(t, 5.2, scores.Value(1), 1e-6)
	require.InDelta(t, 5.0, functionScores.Value(1), 1e-6)
	require.Equal(t, int64(3), ids.Value(2))
	require.InDelta(t, 3.9, scores.Value(2), 1e-6)
	require.InDelta(t, 3.0, functionScores.Value(2), 1e-6)
}

func TestApplyBoostScoresNoScorersNoop(t *testing.T) {
	df := makeBoostScoreTestDF(t,
		[]int64{1},
		[]float32{0.5},
		[]int64{10},
		[]int64{1},
	)
	defer df.Release()
	segDFs := []*chain.DataFrame{df}
	task := makeBoostScoreTestTask(t, &planpb.PlanNode{})

	require.NoError(t, task.applyBoostScores(segDFs, []segments.Segment{nil}, nil))
	require.Same(t, df, segDFs[0])
}

func TestApplyBoostScoresDistanceMetricKeepsInternalScoreDescending(t *testing.T) {
	oldFactory := boostScoreRunnerFactory
	boostScoreRunnerFactory = mockBoostScoreRunnerFactory(boostScoreOutput{
		scores:   []float32{1.0, 1.0, 0.25},
		hasScore: []bool{true, true, true},
	})
	defer func() { boostScoreRunnerFactory = oldFactory }()

	df := makeBoostScoreTestDF(t,
		[]int64{1, 2, 3},
		[]float32{0.0, -0.4, -0.8},
		[]int64{10, 20, 30},
		[]int64{3},
	)
	segDFs := []*chain.DataFrame{df}

	task := makeBoostScoreTestTask(t, &planpb.PlanNode{
		Node: &planpb.PlanNode_VectorAnns{
			VectorAnns: &planpb.VectorANNS{
				QueryInfo: &planpb.QueryInfo{MetricType: metric.L2},
			},
		},
		Scorers: []*planpb.ScoreFunction{{Weight: 1}},
		ScoreOption: &planpb.ScoreOption{
			BoostMode: planpb.BoostMode_BoostModeMultiply,
		},
	})

	require.NoError(t, task.applyBoostScores(segDFs, []segments.Segment{nil}, nil))
	defer segDFs[0].Release()

	result := segDFs[0]
	ids := result.Column(types.IDFieldName).Chunk(0).(*array.Int64)
	scores := result.Column(types.ScoreFieldName).Chunk(0).(*array.Float32)
	require.Equal(t, int64(1), ids.Value(0))
	require.InDelta(t, 0.0, scores.Value(0), 1e-6)
	require.Equal(t, int64(3), ids.Value(1))
	require.InDelta(t, -0.2, scores.Value(1), 1e-6)
	require.Equal(t, int64(2), ids.Value(2))
	require.InDelta(t, -0.4, scores.Value(2), 1e-6)
}

func TestApplyBoostScoresDistanceMetricSumKeepsInternalScoreDescending(t *testing.T) {
	oldFactory := boostScoreRunnerFactory
	boostScoreRunnerFactory = mockBoostScoreRunnerFactory(boostScoreOutput{
		scores:   []float32{0.0, 0.0, 0.7},
		hasScore: []bool{false, false, true},
	})
	defer func() { boostScoreRunnerFactory = oldFactory }()

	df := makeBoostScoreTestDF(t,
		[]int64{1, 2, 3},
		[]float32{0.0, -0.4, -0.8},
		[]int64{10, 20, 30},
		[]int64{3},
	)
	segDFs := []*chain.DataFrame{df}

	task := makeBoostScoreTestTask(t, &planpb.PlanNode{
		Node: &planpb.PlanNode_VectorAnns{
			VectorAnns: &planpb.VectorANNS{
				QueryInfo: &planpb.QueryInfo{MetricType: metric.L2},
			},
		},
		Scorers: []*planpb.ScoreFunction{{Weight: 1}},
		ScoreOption: &planpb.ScoreOption{
			BoostMode: planpb.BoostMode_BoostModeSum,
		},
	})

	require.NoError(t, task.applyBoostScores(segDFs, []segments.Segment{nil}, nil))
	defer segDFs[0].Release()

	result := segDFs[0]
	ids := result.Column(types.IDFieldName).Chunk(0).(*array.Int64)
	scores := result.Column(types.ScoreFieldName).Chunk(0).(*array.Float32)
	require.Equal(t, int64(1), ids.Value(0))
	require.InDelta(t, 0.0, scores.Value(0), 1e-6)
	require.Equal(t, int64(3), ids.Value(1))
	require.InDelta(t, -0.1, scores.Value(1), 1e-6)
	require.Equal(t, int64(2), ids.Value(2))
	require.InDelta(t, -0.4, scores.Value(2), 1e-6)
}

func TestApplyBoostScoresMultipleScorersAllMissKeepOriginalScores(t *testing.T) {
	oldFactory := boostScoreRunnerFactory
	boostScoreRunnerFactory = mockBoostScoreRunnerFactory(
		boostScoreOutput{scores: []float32{0.0, 0.0}, hasScore: []bool{false, false}},
		boostScoreOutput{scores: []float32{0.0, 0.0}, hasScore: []bool{false, false}},
	)
	defer func() { boostScoreRunnerFactory = oldFactory }()

	df := makeBoostScoreTestDF(t,
		[]int64{1, 2},
		[]float32{0.3, 0.7},
		[]int64{10, 20},
		[]int64{2},
	)
	segDFs := []*chain.DataFrame{df}

	task := makeBoostScoreTestTask(t, &planpb.PlanNode{
		Scorers: []*planpb.ScoreFunction{{Weight: 1}, {Weight: 2}},
		ScoreOption: &planpb.ScoreOption{
			FunctionMode: planpb.FunctionMode_FunctionModeSum,
			BoostMode:    planpb.BoostMode_BoostModeSum,
		},
	})

	require.NoError(t, task.applyBoostScores(segDFs, []segments.Segment{nil}, nil))
	defer segDFs[0].Release()

	result := segDFs[0]
	ids := result.Column(types.IDFieldName).Chunk(0).(*array.Int64)
	scores := result.Column(types.ScoreFieldName).Chunk(0).(*array.Float32)
	functionScores := result.Column(functionScoreColumn).Chunk(0).(*array.Float32)
	require.Equal(t, int64(2), ids.Value(0))
	require.InDelta(t, 0.7, scores.Value(0), 1e-6)
	require.True(t, functionScores.IsNull(0))
	require.Equal(t, int64(1), ids.Value(1))
	require.InDelta(t, 0.3, scores.Value(1), 1e-6)
	require.True(t, functionScores.IsNull(1))
}

func TestExtractPlanScorers(t *testing.T) {
	t.Run("empty plan", func(t *testing.T) {
		scorers, err := extractPlanScorers(nil)
		require.NoError(t, err)
		require.Empty(t, scorers)
	})

	t.Run("no scorers", func(t *testing.T) {
		blob, err := proto.Marshal(&planpb.PlanNode{})
		require.NoError(t, err)

		scorers, err := extractPlanScorers(blob)
		require.NoError(t, err)
		require.Empty(t, scorers)
	})

	t.Run("one scorer", func(t *testing.T) {
		plan := &planpb.PlanNode{
			Scorers: []*planpb.ScoreFunction{{Weight: 2.5}},
		}
		blob, err := proto.Marshal(plan)
		require.NoError(t, err)

		scorers, err := extractPlanScorers(blob)
		require.NoError(t, err)
		require.Len(t, scorers, 1)
		require.Equal(t, float32(2.5), scorers[0].GetWeight())
	})

	t.Run("multiple scorers", func(t *testing.T) {
		plan := &planpb.PlanNode{
			Scorers: []*planpb.ScoreFunction{{Weight: 1}, {Weight: 3}},
		}
		blob, err := proto.Marshal(plan)
		require.NoError(t, err)

		scorers, err := extractPlanScorers(blob)
		require.NoError(t, err)
		require.Len(t, scorers, 2)
		require.Equal(t, float32(1), scorers[0].GetWeight())
		require.Equal(t, float32(3), scorers[1].GetWeight())
	})

	t.Run("invalid plan", func(t *testing.T) {
		_, err := extractPlanScorers([]byte{0xff, 0x01})
		require.Error(t, err)
	})
}
