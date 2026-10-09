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
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/util/function/chain"
)

// PostProcessPlan represents an explicit post-process function chain and its
// schema dependencies. Legacy order-by and highlighter requests continue to
// use their existing search pipelines and do not produce this plan.
type PostProcessPlan struct {
	Chain     *schemapb.FunctionChain
	ChainRepr *chain.ChainRepr

	inputPlan *chain.DataFrameInputPlan
}

func (p *PostProcessPlan) GetInputFieldNames() []string {
	return p.inputPlan.PhysicalFieldNames()
}

func (p *PostProcessPlan) GetInputFieldIDs() []int64 {
	return p.inputPlan.PhysicalFieldIDs()
}

func (p *PostProcessPlan) GetInputPlan() *chain.DataFrameInputPlan {
	return p.inputPlan
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
	inputPlan, err := planFunctionChainInputs(repr, schema)
	if err != nil {
		return nil, err
	}
	return &PostProcessPlan{
		Chain:     postProcessChain,
		ChainRepr: repr,
		inputPlan: inputPlan,
	}, nil
}
