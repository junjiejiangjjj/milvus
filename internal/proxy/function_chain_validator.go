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
	chainPB         *schemapb.FunctionChain
	repr            *chain.ChainRepr
}

func (m *functionChainRerankMeta) GetInputFieldNames() []string { return m.inputFieldNames }
func (m *functionChainRerankMeta) GetInputFieldIDs() []int64    { return m.inputFieldIDs }

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
		return newRerankMeta(schema.CollectionSchema, request.GetFunctionScore()), nil
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

	for _, input := range repr.Info.RequiredInputs {
		if isPostProcessDynamicPath(input) {
			return merr.WrapErrParameterInvalidMsg(
				"dynamic field input %q is not supported by post-process yet", input)
		}
		if isPostProcessJSONPath(input, schema) {
			return merr.WrapErrParameterInvalidMsg(
				"JSON path input %q is not supported by post-process yet", input)
		}
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
		if err := validateL2RerankSystemNames(r); err != nil {
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
	inputFieldNames, inputFieldIDs, err := planL2FunctionChainInputs(repr, schema)
	if err != nil {
		return nil, err
	}

	return &functionChainRerankMeta{
		inputFieldNames: inputFieldNames,
		inputFieldIDs:   inputFieldIDs,
		chainPB:         chainPB,
		repr:            repr,
	}, nil
}

func planL2FunctionChainInputs(repr *chain.ChainRepr, schema *schemaInfo) ([]string, []int64, error) {
	if repr == nil {
		return nil, nil, merr.WrapErrParameterInvalidMsg("function chain repr is nil")
	}

	inputFieldNames := make([]string, 0)
	inputFieldIDs := make([]int64, 0)
	seenInputFields := make(map[string]struct{})

	for _, input := range repr.Info.RequiredInputs {
		if chain.IsFunctionChainSystemName(input) {
			if err := validateL2RerankSystemInput(input); err != nil {
				return nil, nil, err
			}
			continue
		}

		_, fieldID, err := getFunctionChainInputField(schema, input)
		if err != nil {
			return nil, nil, err
		}

		if _, ok := seenInputFields[input]; ok {
			continue
		}
		seenInputFields[input] = struct{}{}
		inputFieldNames = append(inputFieldNames, input)
		inputFieldIDs = append(inputFieldIDs, fieldID)
	}

	return inputFieldNames, inputFieldIDs, nil
}

func validateL2RerankSystemNames(repr *chain.ChainRepr) error {
	if repr == nil {
		return merr.WrapErrParameterInvalidMsg("function chain repr is nil")
	}
	for opIdx, op := range repr.Info.Ops {
		for _, input := range op.ReadNames {
			if !chain.IsFunctionChainSystemName(input) {
				continue
			}
			if err := validateL2RerankSystemInput(input); err != nil {
				return merr.WrapErrParameterInvalidMsg("op[%d] input %q: %v", opIdx, input, err)
			}
		}
		for _, output := range repr.Operators[opIdx].Outputs {
			if !chain.IsFunctionChainSystemName(output) {
				continue
			}
			if err := validateL2RerankSystemOutput(output); err != nil {
				return merr.WrapErrParameterInvalidMsg("op[%d] output %q: %v", opIdx, output, err)
			}
		}
	}
	return nil
}

func validateL2RerankSystemInput(name string) error {
	switch name {
	case chaintypes.IDFieldName, chaintypes.ScoreFieldName:
		return nil
	default:
		return merr.WrapErrParameterInvalidMsg("system input %q is not supported by L2 rerank function chain", name)
	}
}

func validateL2RerankSystemOutput(name string) error {
	switch name {
	case chaintypes.ScoreFieldName:
		return nil
	default:
		return merr.WrapErrParameterInvalidMsg("system output %q is not writable by L2 rerank function chain", name)
	}
}

func getFunctionChainInputField(schema *schemaInfo, name string) (*schemapb.FieldSchema, int64, error) {
	if schema == nil || schema.SchemaHelper == nil {
		return nil, 0, merr.WrapErrParameterInvalidMsg("function chain input %q is neither a previous output nor a collection field", name)
	}

	field, err := schema.SchemaHelper.GetFieldFromName(name)
	if err != nil {
		return nil, 0, merr.WrapErrParameterInvalidMsg("function chain input %q is neither a previous output nor a collection field", name)
	}

	return validateFunctionChainInputField(name, field, field.GetFieldID())
}

func validateFunctionChainInputField(name string, field *schemapb.FieldSchema, fieldID int64) (*schemapb.FieldSchema, int64, error) {
	if _, err := chain.ToArrowType(field.GetDataType()); err != nil {
		return nil, 0, merr.WrapErrParameterInvalidMsg("function chain input %q has unsupported field type %s", name, field.GetDataType().String())
	}
	return field, fieldID, nil
}
