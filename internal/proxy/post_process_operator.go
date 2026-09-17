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
	"context"

	"github.com/apache/arrow/go/v17/arrow/memory"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	json "github.com/milvus-io/milvus/internal/json"
	"github.com/milvus-io/milvus/internal/util/function/chain"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

type postProcessOperator struct {
	plan          *PostProcessPlan
	alloc         memory.Allocator
	dynamicFields []string
}

func newPostProcessOperator(t *searchTask, _ map[string]any) (operator, error) {
	if t.postProcessPlan == nil {
		return nil, merr.WrapErrServiceInternal("post-process plan is missing")
	}
	return &postProcessOperator{plan: t.postProcessPlan, alloc: memory.DefaultAllocator, dynamicFields: append([]string(nil), t.userDynamicFields...)}, nil
}

func (op *postProcessOperator) run(ctx context.Context, _ trace.Span, inputs ...any) ([]any, error) {
	if len(inputs) != 1 {
		return nil, merr.WrapErrServiceInternal("post-process expects one search result")
	}
	result, ok := inputs[0].(*milvuspb.SearchResults)
	if !ok || result == nil || result.GetResults() == nil {
		return nil, merr.WrapErrServiceInternal("post-process search result is missing")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if op.plan == nil || op.plan.ChainRepr == nil || op.plan.GetInputPlan() == nil {
		return nil, merr.WrapErrServiceInternal("post-process input plan is missing")
	}
	if err := validatePostProcessResult(result.Results); err != nil {
		return nil, err
	}
	fc, err := chain.FuncChainFromRepr(op.plan.ChainRepr, op.alloc)
	if err != nil {
		return nil, err
	}
	df, err := chain.FromSearchResultDataWithPayload(result.Results, op.alloc, op.plan.GetInputPlan())
	if err != nil {
		return nil, err
	}
	defer df.Release()
	// Keep complete response columns through every operator.
	processed, err := fc.ExecuteWithOptions(ctx, chain.ExecuteOptions{}, df)
	if err != nil {
		return nil, err
	}
	if processed != df {
		defer processed.Release()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	data, err := chain.ToSearchResultDataWithPayload(processed)
	if err != nil {
		return nil, err
	}
	selected := proto.Clone(result).(*milvuspb.SearchResults)
	// Preserve non-row metadata while all row data comes directly from the DataFrame.
	selected.Results.Ids = data.Ids
	selected.Results.Scores = data.Scores
	selected.Results.FieldsData = data.FieldsData
	selected.Results.Topks = data.Topks
	selected.Results.TopK = data.TopK
	selected.Results.Distances = data.Distances
	selected.Results.Recalls = data.Recalls
	if err := projectPostProcessDynamicFields(selected.Results.FieldsData, op.dynamicFields); err != nil {
		return nil, err
	}
	return []any{selected}, nil
}

func validatePostProcessResult(data *schemapb.SearchResultData) error {
	if data.NumQueries <= 0 || int64(len(data.Topks)) != data.NumQueries {
		return merr.WrapErrServiceInternal("post-process query count does not match Topks")
	}
	// These modes need their own result contracts before enabling PostProcess.
	if data.GetGroupByFieldValue() != nil || len(data.GetGroupByFieldValues()) > 0 ||
		data.GetElementIndices() != nil || data.GetSearchIteratorV2Results() != nil ||
		len(data.GetHighlightResults()) > 0 || len(data.GetAggBuckets()) > 0 || len(data.GetAggTopks()) > 0 {
		return merr.WrapErrServiceInternal("post-process received an unsupported result mode")
	}
	rows := len(data.Scores)
	remaining := int64(rows)
	for _, size := range data.Topks {
		if size < 0 || size > remaining {
			return merr.WrapErrServiceInternal("post-process Topks do not match score count")
		}
		remaining -= size
	}
	if remaining != 0 || typeutil.GetSizeOfIDs(data.Ids) != rows {
		return merr.WrapErrServiceInternal("post-process IDs, scores and Topks are not aligned")
	}
	if (len(data.Distances) != 0 && len(data.Distances) != rows) ||
		(len(data.Recalls) != 0 && int64(len(data.Recalls)) != data.NumQueries) {
		return merr.WrapErrServiceInternal("post-process auxiliary scores are not aligned")
	}
	for _, field := range data.FieldsData {
		if err := chain.ValidateResultField(field, rows); err != nil {
			return err
		}
	}
	return nil
}

// A complete dynamic root may have been fetched for hidden chain dependencies.
// Project it back to the user's original top-level keys before returning it.
func projectPostProcessDynamicFields(fields []*schemapb.FieldData, names []string) error {
	if len(names) == 0 {
		return nil
	}
	for _, field := range fields {
		if field.GetFieldName() != common.MetaFieldName || !field.GetIsDynamic() {
			continue
		}
		data := field.GetScalars().GetJsonData()
		if data == nil {
			return merr.WrapErrServiceInternal("post-process dynamic root is not JSON")
		}
		for i, raw := range data.Data {
			if valid := typeutil.GetFieldDataValidData(field); len(valid) > 0 && !valid[i] {
				continue
			}
			var source map[string]json.RawMessage
			if err := json.Unmarshal(raw, &source); err != nil {
				return merr.WrapErrDataIntegrity(err, "invalid post-process dynamic result")
			}
			projected := make(map[string]json.RawMessage, len(names))
			for _, name := range names {
				if value, ok := source[name]; ok {
					projected[name] = value
				}
			}
			encoded, err := json.Marshal(projected)
			if err != nil {
				return merr.WrapErrServiceInternalErr(err, "encode post-process dynamic result")
			}
			data.Data[i] = encoded
		}
	}
	return nil
}
