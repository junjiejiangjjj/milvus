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

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/util/function/chain"
	chaintypes "github.com/milvus-io/milvus/internal/util/function/chain/types"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// PostProcessPlan represents an explicit post-process function chain and its
// schema dependencies. Legacy order-by and highlighter requests continue to
// use their existing search pipelines and do not produce this plan.
type PostProcessPlan struct {
	Chain     *schemapb.FunctionChain
	ChainRepr *chain.ChainRepr

	inputFieldNames []string
	inputFieldIDs   []int64
}

func (p *PostProcessPlan) GetInputFieldNames() []string {
	return append([]string(nil), p.inputFieldNames...)
}

func (p *PostProcessPlan) GetInputFieldIDs() []int64 {
	return append([]int64(nil), p.inputFieldIDs...)
}

func buildPostProcessPlan(
	postProcessChain *schemapb.FunctionChain,
	schema *schemaInfo,
) (*PostProcessPlan, error) {
	if postProcessChain == nil {
		return nil, nil
	}

	repr, err := validatePostProcessChain(postProcessChain)
	if err != nil {
		return nil, err
	}
	if err := validatePostProcessCurrentCapabilities(repr, schema); err != nil {
		return nil, err
	}
	inputFieldNames, inputFieldIDs, err := planPostProcessInputs(repr, schema)
	if err != nil {
		return nil, err
	}
	return &PostProcessPlan{
		Chain:           postProcessChain,
		ChainRepr:       repr,
		inputFieldNames: inputFieldNames,
		inputFieldIDs:   inputFieldIDs,
	}, nil
}

func planPostProcessInputs(repr *chain.ChainRepr, schema *schemaInfo) ([]string, []int64, error) {
	if repr == nil {
		return nil, nil, merr.WrapErrParameterInvalidMsg("post-process function chain repr is nil")
	}

	inputFieldNames := make([]string, 0)
	inputFieldIDs := make([]int64, 0)
	seenFields := make(map[string]struct{})

	for _, input := range repr.Info.RequiredInputs {
		if isPostProcessDynamicPath(input) {
			return nil, nil, merr.WrapErrParameterInvalidMsg(
				"dynamic field input %q is not supported by post-process yet", input)
		}
		if chain.IsFunctionChainSystemName(input) {
			switch input {
			case chaintypes.IDFieldName, chaintypes.ScoreFieldName:
				continue
			default:
				return nil, nil, merr.WrapErrParameterInvalidMsg(
					"system input %q is not supported by post-process function chain", input)
			}
		}

		_, fieldID, err := getFunctionChainInputField(schema, input)
		if err != nil {
			return nil, nil, err
		}
		if _, ok := seenFields[input]; ok {
			continue
		}
		seenFields[input] = struct{}{}
		inputFieldNames = append(inputFieldNames, input)
		inputFieldIDs = append(inputFieldIDs, fieldID)
	}

	return inputFieldNames, inputFieldIDs, nil
}

func isPostProcessDynamicPath(name string) bool {
	return strings.HasPrefix(name, `$meta["`)
}
