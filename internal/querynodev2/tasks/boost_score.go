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
	"fmt"

	"github.com/apache/arrow/go/v17/arrow"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus/internal/querynodev2/segments"
	"github.com/milvus-io/milvus/internal/util/function/chain"
	"github.com/milvus-io/milvus/internal/util/function/chain/expr"
	"github.com/milvus-io/milvus/internal/util/function/chain/types"
	"github.com/milvus-io/milvus/internal/util/segcore"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

const (
	boostScoreColumnPrefix = "$boost_score_"
	functionScoreColumn    = "$function_score"
)

func boostScoreColumn(index int) string {
	return fmt.Sprintf("%s%d", boostScoreColumnPrefix, index)
}

func extractPlanScorers(serializedPlan []byte) ([]*planpb.ScoreFunction, error) {
	plan, err := extractPlanWithScorers(serializedPlan)
	if err != nil || plan == nil {
		return nil, err
	}
	return plan.GetScorers(), nil
}

func extractPlanWithScorers(serializedPlan []byte) (*planpb.PlanNode, error) {
	if len(serializedPlan) == 0 {
		return nil, nil
	}

	plan := &planpb.PlanNode{}
	if err := proto.Unmarshal(serializedPlan, plan); err != nil {
		return nil, err
	}
	return plan, nil
}

func functionModeToScoreCombineMode(mode planpb.FunctionMode) (string, error) {
	switch mode {
	case planpb.FunctionMode_FunctionModeMultiply:
		return expr.ModeMultiply, nil
	case planpb.FunctionMode_FunctionModeSum:
		return expr.ModeSum, nil
	default:
		return "", merr.WrapErrServiceInternal(fmt.Sprintf("boost_score: unknown function mode %s", mode.String()))
	}
}

func boostModeToScoreCombineMode(mode planpb.BoostMode) (string, error) {
	switch mode {
	case planpb.BoostMode_BoostModeMultiply:
		return expr.ModeMultiply, nil
	case planpb.BoostMode_BoostModeSum:
		return expr.ModeSum, nil
	default:
		return "", merr.WrapErrServiceInternal(fmt.Sprintf("boost_score: unknown boost mode %s", mode.String()))
	}
}

var boostScoreRunnerFactory = newSegmentBoostScoreRunner

func newSegmentBoostScoreRunner(segment segments.Segment, searchReq *segcore.SearchRequest, scorer *planpb.ScoreFunction) expr.BoostScoreRunner {
	return func(ctx context.Context, offsets *arrow.Chunked) (*arrow.Chunked, error) {
		return segments.ComputeScorerScoresOnChunkedOffsets(ctx, segment, searchReq, scorer, offsets)
	}
}

func (t *SearchTask) applyBoostScores(segDFs []*chain.DataFrame, searchedSegments []segments.Segment, searchReq *segcore.SearchRequest) error {
	if len(segDFs) != len(searchedSegments) {
		return merr.WrapErrServiceInternal(fmt.Sprintf("boost_score: DataFrame count %d does not match segment count %d", len(segDFs), len(searchedSegments)))
	}

	plan, err := extractPlanWithScorers(t.req.GetReq().GetSerializedExprPlan())
	if err != nil {
		return merr.WrapErrServiceInternal(fmt.Sprintf("boost_score: failed to parse search plan scorers: %v", err))
	}
	if plan == nil || len(plan.GetScorers()) == 0 {
		return nil
	}

	scorers := plan.GetScorers()
	functionMode, err := functionModeToScoreCombineMode(plan.GetScoreOption().GetFunctionMode())
	if err != nil {
		return err
	}
	boostMode, err := boostModeToScoreCombineMode(plan.GetScoreOption().GetBoostMode())
	if err != nil {
		return err
	}

	boostedDFs := make([]*chain.DataFrame, len(segDFs))
	for i, df := range segDFs {
		segment := searchedSegments[i]
		if df == nil {
			return merr.WrapErrServiceInternal(fmt.Sprintf("boost_score: DataFrame %d is nil", i))
		}

		boostChain := chain.NewFuncChainWithAllocator(defaultAllocator).
			SetName("l0-rerank").
			SetStage(types.StageL0Rerank)
		boostScoreColumns := make([]string, 0, len(scorers))
		for scorerIdx, scorer := range scorers {
			outputCol := boostScoreColumn(scorerIdx)
			boostExpr, err := expr.NewBoostScoreExpr(boostScoreRunnerFactory(segment, searchReq, scorer))
			if err != nil {
				return err
			}
			boostChain.Map(boostExpr, []string{types.SegOffsetFieldName}, []string{outputCol})
			boostScoreColumns = append(boostScoreColumns, outputCol)
		}

		finalFunctionScoreColumn := boostScoreColumns[0]
		if len(boostScoreColumns) > 1 {
			functionCombineExpr, err := expr.NewScoreCombineExpr(functionMode, nil)
			if err != nil {
				return err
			}
			boostChain.Map(functionCombineExpr, boostScoreColumns, []string{functionScoreColumn})
			finalFunctionScoreColumn = functionScoreColumn
		}

		finalCombineExpr, err := expr.NewScoreCombineExpr(boostMode, nil)
		if err != nil {
			return err
		}
		boostChain.Map(finalCombineExpr,
			[]string{types.ScoreFieldName, finalFunctionScoreColumn},
			[]string{types.ScoreFieldName})
		boostChain.Sort(types.ScoreFieldName, true)

		boosted, err := boostChain.ExecuteWithContext(t.ctx, df)
		if err != nil {
			return err
		}

		boostedDFs[i] = boosted
		df.Release()
	}
	copy(segDFs, boostedDFs)

	return nil
}
