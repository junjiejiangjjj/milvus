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
	"strings"

	"github.com/apache/arrow/go/v17/arrow/memory"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/util/function/chain"
	chaintypes "github.com/milvus-io/milvus/internal/util/function/chain/types"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

type functionChainRerankMeta struct {
	inputFieldNames []string
	inputFieldIDs   []int64
	inputPlan       *chain.DataFrameInputPlan
	chainPB         *schemapb.FunctionChain
	repr            *chain.ChainRepr
}

func (m *functionChainRerankMeta) GetInputFieldNames() []string { return m.inputFieldNames }
func (m *functionChainRerankMeta) GetInputFieldIDs() []int64    { return m.inputFieldIDs }
func (m *functionChainRerankMeta) GetInputPlan() *chain.DataFrameInputPlan {
	return m.inputPlan
}

func hasFunctionRerank(request *milvuspb.SearchRequest) bool {
	return request.GetFunctionScore() != nil || hasFunctionChainRerankStage(request.GetFunctionChains())
}

func validateFunctionChainSearchRequest(request *milvuspb.SearchRequest, _ bool) error {
	if request.GetFunctionScore() != nil && hasFunctionChainRerankStage(request.GetFunctionChains()) {
		return merr.WrapErrParameterInvalidMsg("function_score and function_chains cannot be used together")
	}
	return nil
}

func validatePostProcessCompatibility(
	postProcessChains []*schemapb.FunctionChain,
	hasOrderBy bool,
	hasHighlighter bool,
	isSearchAggregation bool,
) error {
	if len(postProcessChains) > 1 {
		return merr.WrapErrParameterInvalidMsg("function chain stage %s appears more than once",
			schemapb.FunctionChainStage_FunctionChainStagePostProcess.String())
	}

	hasPostProcess := len(postProcessChains) == 1
	if hasPostProcess && hasOrderBy {
		return merr.WrapErrParameterInvalidMsg("explicit post-process function chain and order_by_fields cannot be used together")
	}
	if hasPostProcess && hasHighlighter {
		return merr.WrapErrParameterInvalidMsg("explicit post-process function chain and highlighter cannot be used together")
	}
	if hasPostProcess && isSearchAggregation {
		return merr.WrapErrParameterInvalidMsg("post process is not supported with search_aggregation")
	}
	if hasOrderBy && isSearchAggregation {
		return merr.WrapErrParameterInvalidMsg("order_by_fields is not supported with search_aggregation")
	}
	if hasHighlighter && isSearchAggregation {
		return merr.WrapErrParameterInvalidMsg("highlighter and search_aggregation cannot be used simultaneously")
	}
	return nil
}

func selectHybridRerankMeta(request *milvuspb.SearchRequest, schema *schemaInfo) (rerankMeta, error) {
	functionChains := request.GetFunctionChains()
	if len(functionChains) > 0 {
		if request.GetFunctionScore() != nil {
			return nil, merr.WrapErrParameterInvalidMsg("function_chains cannot be used with function_score")
		}
		if hasExplicitLegacyReranker(request.GetSearchParams()) {
			return nil, merr.WrapErrParameterInvalidMsg("function_chains cannot be used with rank_params strategy or params")
		}
		return newHybridFunctionChainRerankMeta(functionChains, schema, len(request.GetSubReqs()))
	}

	if request.GetFunctionScore() != nil {
		for index, subReq := range request.GetSubReqs() {
			if len(subReq.GetFunctionChains()) > 0 {
				return nil, merr.WrapErrParameterInvalidMsg(
					"function_score cannot be used with function_chains in sub-search[%d]", index)
			}
		}
		return newRerankMeta(schema.CollectionSchema, request.GetFunctionScore())
	}
	return newRerankMetaFromLegacy(request.GetSearchParams()), nil
}

func hasExplicitLegacyReranker(params []*commonpb.KeyValuePair) bool {
	for _, param := range params {
		if param == nil {
			continue
		}
		switch strings.ToLower(param.GetKey()) {
		case RankTypeKey:
			if strings.TrimSpace(param.GetValue()) != "" {
				return true
			}
		case ParamsKey:
			value := strings.TrimSpace(param.GetValue())
			if value != "" && !strings.EqualFold(value, "null") {
				return true
			}
		}
	}
	return false
}

func hasFunctionChainRerankStage(chains []*schemapb.FunctionChain) bool {
	for _, chainPB := range chains {
		if chainPB == nil {
			continue
		}
		switch chainPB.GetStage() {
		case schemapb.FunctionChainStage_FunctionChainStageL0Rerank,
			schemapb.FunctionChainStage_FunctionChainStageL1Rerank,
			schemapb.FunctionChainStage_FunctionChainStageL2Rerank:
			return true
		}
	}
	return false
}

func hasFunctionChainStage(chains []*schemapb.FunctionChain, target schemapb.FunctionChainStage) bool {
	for _, chainPB := range chains {
		if chainPB != nil && chainPB.GetStage() == target {
			return true
		}
	}
	return false
}

func splitFunctionChainsByStage(chains []*schemapb.FunctionChain) ([]*schemapb.FunctionChain, []*schemapb.FunctionChain, []*schemapb.FunctionChain, error) {
	l2Chains := make([]*schemapb.FunctionChain, 0)
	querynodeChains := make([]*schemapb.FunctionChain, 0)
	postProcessChains := make([]*schemapb.FunctionChain, 0)
	seenStages := make(map[schemapb.FunctionChainStage]struct{}, len(chains))

	for i, chainPB := range chains {
		if chainPB == nil {
			return nil, nil, nil, merr.WrapErrParameterInvalidMsg("function chain[%d] is nil", i)
		}
		stage := chainPB.GetStage()
		if _, ok := seenStages[stage]; ok {
			return nil, nil, nil, merr.WrapErrParameterInvalidMsg("function chain stage %s appears more than once", stage.String())
		}
		seenStages[stage] = struct{}{}

		switch stage {
		case schemapb.FunctionChainStage_FunctionChainStageL2Rerank:
			l2Chains = append(l2Chains, chainPB)
		case schemapb.FunctionChainStage_FunctionChainStageL0Rerank,
			schemapb.FunctionChainStage_FunctionChainStageL1Rerank:
			if len(chainPB.GetOps()) == 0 {
				return nil, nil, nil, merr.WrapErrParameterInvalidMsg("function chain[%d] must contain at least one op", i)
			}
			querynodeChains = append(querynodeChains, chainPB)
		case schemapb.FunctionChainStage_FunctionChainStagePostProcess:
			postProcessChains = append(postProcessChains, chainPB)
		default:
			return nil, nil, nil, merr.WrapErrParameterInvalidMsg("function chain[%d] stage %s is not supported in search request", i, stage.String())
		}
	}

	return l2Chains, querynodeChains, postProcessChains, nil
}

func validatePostProcessChain(chainPB *schemapb.FunctionChain) (*chain.ChainRepr, error) {
	if chainPB == nil {
		return nil, merr.WrapErrParameterInvalidMsg("post-process function chain is nil")
	}
	if chainPB.GetStage() != schemapb.FunctionChainStage_FunctionChainStagePostProcess {
		return nil, merr.WrapErrParameterInvalidMsg("expected post-process function chain, got stage %s", chainPB.GetStage().String())
	}
	if len(chainPB.GetOps()) == 0 {
		return nil, merr.WrapErrParameterInvalidMsg("post-process function chain must contain at least one op")
	}

	repr, err := chain.ProtoChainToRepr(chainPB)
	if err != nil {
		return nil, merr.Wrap(err, "invalid post-process function chain")
	}

	for i, op := range repr.Operators {
		switch op.Type {
		case chaintypes.OpTypeMap, chaintypes.OpTypeSort, chaintypes.OpTypeLimit:
		default:
			return nil, merr.WrapErrParameterInvalidMsg(
				"post-process function chain op[%d] type %q is not supported; only map, sort, and limit are allowed", i, op.Type)
		}

		for _, output := range op.Outputs {
			// A $meta["..."] output is a dynamic-field path, not a write to
			// the $meta system column itself. Its full syntax is validated by
			// the PostProcess column planner.
			isDynamicOutput := strings.HasPrefix(output, `$meta["`)
			if chain.IsFunctionChainSystemName(output) &&
				output != chaintypes.HighlightFieldName && !isDynamicOutput {
				return nil, merr.WrapErrParameterInvalidMsg(
					"post-process function chain cannot write system output %q; only %s is writable",
					output, chaintypes.HighlightFieldName)
			}
		}
	}
	return repr, nil
}

func validatePostProcessCurrentCapabilities(repr *chain.ChainRepr, schema *schemaInfo) error {
	if repr == nil {
		return merr.WrapErrParameterInvalidMsg("post-process function chain repr is nil")
	}

	for opIdx, op := range repr.Operators {
		for _, output := range op.Outputs {
			if isPostProcessDynamicPath(output) {
				return merr.WrapErrParameterInvalidMsg(
					"post-process function chain op[%d] dynamic field output %q is not supported yet", opIdx, output)
			}
			if isPostProcessJSONPath(output, schema) {
				return merr.WrapErrParameterInvalidMsg(
					"post-process function chain op[%d] JSON path output %q is not supported yet", opIdx, output)
			}
			if output == chaintypes.HighlightFieldName {
				return merr.WrapErrParameterInvalidMsg(
					"post-process function chain op[%d] output %q is not supported yet", opIdx, output)
			}
			if field := getPostProcessSchemaField(schema, output); field != nil {
				return merr.WrapErrParameterInvalidMsg(
					"post-process function chain op[%d] cannot overwrite schema field %q", opIdx, output)
			}
		}
	}

	if _, err := chain.FuncChainFromRepr(repr, memory.DefaultAllocator); err != nil {
		return merr.Wrap(err, "invalid post-process function chain")
	}
	return nil
}

func getPostProcessSchemaField(schema *schemaInfo, name string) *schemapb.FieldSchema {
	if schema == nil || schema.SchemaHelper == nil {
		return nil
	}
	field, err := schema.SchemaHelper.GetFieldFromName(name)
	if err != nil {
		return nil
	}
	return field
}

func isPostProcessJSONPath(name string, schema *schemaInfo) bool {
	pathStart := strings.IndexByte(name, '[')
	if pathStart <= 0 {
		return false
	}
	root := strings.TrimSpace(name[:pathStart])
	if root == "$meta" {
		return true
	}
	field := getPostProcessSchemaField(schema, root)
	return field != nil && field.GetDataType() == schemapb.DataType_JSON
}

func newFunctionChainRerankMeta(chains []*schemapb.FunctionChain, schema *schemaInfo) (*functionChainRerankMeta, error) {
	chainPB, repr, err := parseL2FunctionChain(chains)
	if err != nil || repr == nil {
		return nil, err
	}

	for i, op := range repr.Operators {
		if op.Type == chaintypes.OpTypeMerge {
			return nil, merr.WrapErrParameterInvalidMsg(
				"function chain operator[%d]: merge is not supported in ordinary search", i)
		}
	}

	return buildFunctionChainRerankMeta(chainPB, repr, schema)
}

func newHybridFunctionChainRerankMeta(chains []*schemapb.FunctionChain, schema *schemaInfo, subSearchCount int) (*functionChainRerankMeta, error) {
	if len(chains) != 1 {
		return nil, merr.WrapErrParameterInvalidMsg("hybrid search requires exactly one function chain, got %d", len(chains))
	}

	chainPB, repr, err := parseL2FunctionChain(chains)
	if err != nil {
		return nil, err
	}
	if err := validateHybridL2FunctionChain(repr, subSearchCount); err != nil {
		return nil, merr.Wrap(err, "function chain[0]")
	}

	return buildFunctionChainRerankMeta(chainPB, repr, schema)
}

func parseL2FunctionChain(chains []*schemapb.FunctionChain) (*schemapb.FunctionChain, *chain.ChainRepr, error) {
	if len(chains) == 0 {
		return nil, nil, nil
	}

	seenStages := make(map[schemapb.FunctionChainStage]struct{}, len(chains))
	var chainPB *schemapb.FunctionChain
	var repr *chain.ChainRepr

	for i, pb := range chains {
		if pb == nil {
			return nil, nil, merr.WrapErrParameterInvalidMsg("function chain[%d] is nil", i)
		}
		stage := pb.GetStage()
		if _, ok := seenStages[stage]; ok {
			return nil, nil, merr.WrapErrParameterInvalidMsg("function chain stage %s appears more than once", stage.String())
		}
		seenStages[stage] = struct{}{}

		if stage != schemapb.FunctionChainStage_FunctionChainStageL2Rerank {
			return nil, nil, merr.WrapErrParameterInvalidMsg("function chain[%d] stage %s is not supported in search request", i, stage.String())
		}
		if len(pb.GetOps()) == 0 {
			return nil, nil, merr.WrapErrParameterInvalidMsg("function chain[%d] must contain at least one op", i)
		}

		r, err := chain.ProtoChainToRepr(pb)
		if err != nil {
			return nil, nil, merr.Wrapf(err, "function chain[%d]", i)
		}
		if err := validateL2RerankSystemOutputs(r); err != nil {
			return nil, nil, merr.Wrapf(err, "function chain[%d]", i)
		}
		chainPB = pb
		repr = r
	}
	return chainPB, repr, nil
}

func validateHybridL2FunctionChain(repr *chain.ChainRepr, subSearchCount int) error {
	if repr == nil {
		return merr.WrapErrParameterInvalidMsg("function chain repr is nil")
	}

	mergeCount := 0
	mergeIndex := -1
	for i, op := range repr.Operators {
		if op.Type == chaintypes.OpTypeMerge {
			mergeCount++
			mergeIndex = i
		}
	}
	if mergeCount != 1 {
		return merr.WrapErrParameterInvalidMsg("hybrid function chain must contain exactly one merge operator")
	}
	if mergeIndex != 0 {
		return merr.WrapErrParameterInvalidMsg("hybrid function chain merge operator must be first")
	}
	return chain.ValidateMergeOpRepr(&repr.Operators[0], subSearchCount)
}

func buildFunctionChainRerankMeta(chainPB *schemapb.FunctionChain, repr *chain.ChainRepr, schema *schemaInfo) (*functionChainRerankMeta, error) {
	inputPlan, err := planFunctionChainInputs(repr, schema)
	if err != nil {
		return nil, err
	}

	return &functionChainRerankMeta{
		inputFieldNames: inputPlan.PhysicalFieldNames(),
		inputFieldIDs:   inputPlan.PhysicalFieldIDs(),
		inputPlan:       inputPlan,
		chainPB:         chainPB,
		repr:            repr,
	}, nil
}

func planFunctionChainInputs(repr *chain.ChainRepr, schema *schemaInfo) (*chain.DataFrameInputPlan, error) {
	if repr == nil {
		return nil, merr.WrapErrParameterInvalidMsg("function chain repr is nil")
	}
	if schema == nil || schema.CollectionSchema == nil {
		return nil, merr.WrapErrParameterInvalidMsg("collection schema is nil")
	}
	return chain.CompileDataFrameInputPlan(repr, schema.CollectionSchema)
}

func validateL2RerankSystemOutputs(repr *chain.ChainRepr) error {
	if repr == nil {
		return merr.WrapErrParameterInvalidMsg("function chain repr is nil")
	}
	for opIdx, op := range repr.Operators {
		for _, output := range op.Outputs {
			if !chain.IsFunctionChainSystemName(output) {
				continue
			}
			if err := validateL2RerankSystemOutput(output); err != nil {
				return merr.Wrapf(err, "op[%d] output %q", opIdx, output)
			}
		}
	}
	return nil
}

func validateL2RerankSystemOutput(name string) error {
	switch name {
	case chaintypes.ScoreFieldName:
		return nil
	default:
		return merr.WrapErrParameterInvalidMsg("system output %q is not writable by L2 rerank function chain", name)
	}
}

func isPostProcessDynamicPath(name string) bool {
	return strings.HasPrefix(name, `$meta["`)
}

// validatePostProcessSearchMode rejects combinations whose pagination or row
// identity contract is not part of the first PostProcess release.
func validatePostProcessSearchMode(isIterator, hasGroupBy, isArrayOfVector bool) error {
	if isIterator {
		return merr.WrapErrParameterInvalidMsg("post-process is not supported with search iterator")
	}
	if hasGroupBy {
		return merr.WrapErrParameterInvalidMsg("post-process is not supported with search group-by")
	}
	if isArrayOfVector {
		return merr.WrapErrParameterInvalidMsg("post-process is not supported with ArrayOfVector search")
	}
	return nil
}
