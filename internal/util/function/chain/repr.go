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
	"encoding/json"
	"fmt"

	"github.com/apache/arrow/go/v17/arrow/memory"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
)

// =============================================================================
// Operator Type Constants
// =============================================================================

const (
	OpTypeMap    = "map"
	OpTypeFilter = "filter"
	OpTypeSelect = "select"
	OpTypeSort   = "sort"
	OpTypeLimit  = "limit"
)

// =============================================================================
// Chain Type Constants
// =============================================================================

const (
	ChainTypeIngestion   = "ingestion" // insert/upsert/import stage
	ChainTypeL2Rerank    = "L2_rerank" // proxy rerank stage
	ChainTypeL1Rerank    = "L1_rerank" // QN/SN rerank stage
	ChainTypeL0Rerank    = "L0_rerank" // segment rerank stage
	ChainTypePreProcess  = "pre_process"
	ChainTypePostProcess = "post_process"
)

// =============================================================================
// JSON Representation Types (for deserialization)
// =============================================================================

// ChainJSON is the JSON representation of a FuncChain for serialization/deserialization.
type ChainJSON struct {
	Name      string         `json:"name,omitempty"`
	ChainType string         `json:"chain_type"`
	Operators []OperatorJSON `json:"operators"`
}

// OperatorJSON is the JSON representation of an Operator for serialization/deserialization.
type OperatorJSON struct {
	Type     string                 `json:"type"`
	Params   map[string]interface{} `json:"params,omitempty"`
	Function *FunctionJSON          `json:"function,omitempty"` // for map operator
	Inputs   []string               `json:"inputs,omitempty"`   // input column names
	Outputs  []string               `json:"outputs,omitempty"`  // output column names
}

// FunctionJSON is the JSON representation of a FunctionExpr for serialization/deserialization.
type FunctionJSON struct {
	Name   string                 `json:"name"`
	Params map[string]interface{} `json:"params"`
}

// =============================================================================
// Internal Representation Types (no JSON tags)
// =============================================================================

// ChainRepr is the internal representation of a FuncChain (no JSON tags).
type ChainRepr struct {
	Name      string
	ChainType string
	Operators []OperatorRepr
}

// OperatorRepr is the internal representation of an Operator (no JSON tags).
// It uses a unified Params map instead of specific fields.
type OperatorRepr struct {
	Type     string
	Params   map[string]interface{}
	Function *FunctionRepr // for map operator
	Inputs   []string      // input column names
	Outputs  []string      // output column names
}

// FunctionRepr is the internal representation of a FunctionExpr (no JSON tags).
type FunctionRepr struct {
	Name   string
	Params map[string]interface{}
}

// =============================================================================
// Conversion Functions
// =============================================================================

// functionJSONToRepr converts FunctionJSON to FunctionRepr.
func functionJSONToRepr(json *FunctionJSON) *FunctionRepr {
	if json == nil {
		return nil
	}
	return &FunctionRepr{
		Name:   json.Name,
		Params: json.Params,
	}
}

// operatorJSONToRepr converts OperatorJSON to OperatorRepr.
// It infers Inputs/Outputs based on operator type and parameters.
func operatorJSONToRepr(json *OperatorJSON) (*OperatorRepr, error) {
	if json == nil {
		return nil, fmt.Errorf("OperatorJSON is nil")
	}

	repr := &OperatorRepr{
		Type:     json.Type,
		Params:   make(map[string]interface{}),
		Function: functionJSONToRepr(json.Function),
		Inputs:   json.Inputs,
		Outputs:  json.Outputs,
	}

	// Copy params
	if json.Params != nil {
		for k, v := range json.Params {
			repr.Params[k] = v
		}
	}
	return repr, nil
}

// chainJSONToRepr converts ChainJSON to ChainRepr.
func chainJSONToRepr(json *ChainJSON) (*ChainRepr, error) {
	if json == nil {
		return nil, fmt.Errorf("ChainJSON is nil")
	}

	repr := &ChainRepr{
		Name:      json.Name,
		ChainType: json.ChainType,
		Operators: make([]OperatorRepr, 0, len(json.Operators)),
	}

	for i, opJSON := range json.Operators {
		opRepr, err := operatorJSONToRepr(&opJSON)
		if err != nil {
			return nil, fmt.Errorf("operator[%d]: %w", i, err)
		}
		repr.Operators = append(repr.Operators, *opRepr)
	}

	return repr, nil
}

// =============================================================================
// Parsing Functions
// =============================================================================

// ParseFuncChainRepr parses a JSON string and creates a FuncChain.
func ParseFuncChainRepr(jsonStr string) (*FuncChain, error) {
	return ParseFuncChainReprWithAllocator(jsonStr, memory.DefaultAllocator)
}

// ParseFuncChainReprWithAllocator parses a JSON string and creates a FuncChain with the given allocator.
func ParseFuncChainReprWithAllocator(jsonStr string, alloc memory.Allocator) (*FuncChain, error) {
	var chainJSON ChainJSON
	if err := json.Unmarshal([]byte(jsonStr), &chainJSON); err != nil {
		return nil, fmt.Errorf("failed to parse JSON: %w", err)
	}

	repr, err := chainJSONToRepr(&chainJSON)
	if err != nil {
		return nil, fmt.Errorf("failed to convert JSON to Repr: %w", err)
	}

	return funcChainFromRepr(repr, alloc)
}

// funcChainFromRepr creates a FuncChain from a ChainRepr.
func funcChainFromRepr(repr *ChainRepr, alloc memory.Allocator) (*FuncChain, error) {
	if alloc == nil {
		alloc = memory.DefaultAllocator
	}

	chain := NewFuncChainWithAllocator(alloc)
	if repr.Name != "" {
		chain.SetName(repr.Name)
	}

	for i, opRepr := range repr.Operators {
		op, err := operatorFromRepr(&opRepr)
		if err != nil {
			return nil, fmt.Errorf("operator[%d]: %w", i, err)
		}
		chain.Add(op)
	}

	return chain, nil
}

// operatorFromRepr creates an Operator from an OperatorRepr.
func operatorFromRepr(repr *OperatorRepr) (Operator, error) {
	factory, ok := GetOperatorFactory(repr.Type)
	if !ok {
		return nil, fmt.Errorf("unknown operator type: %s", repr.Type)
	}
	return factory(repr)
}

// FunctionFromRepr creates a FunctionExpr from a FunctionRepr.
func FunctionFromRepr(repr *FunctionRepr) (FunctionExpr, error) {
	if repr.Name == "" {
		return nil, fmt.Errorf("function name is required")
	}

	return globalRegistry.Create(repr.Name, repr.Params)
}

// =============================================================================
// FunctionScore Conversion
// =============================================================================

// functionScoreToRepr converts schemapb.FunctionScore to ChainRepr.
// This supports multiple functions for future multi-rerank scenarios.
func functionScoreToRepr(funcScore *schemapb.FunctionScore) (*ChainRepr, error) {
	if funcScore == nil {
		return nil, fmt.Errorf("funcScore is nil")
	}

	repr := &ChainRepr{
		Name:      "function_score_chain",
		Operators: make([]OperatorRepr, 0),
	}

	// Collect all function score column names
	funcScoreCols := []string{}

	// 1. Add map operation for each function
	for i, funcSchema := range funcScore.Functions {
		funcRepr, inputCols, err := functionSchemaToFunctionReprWithInputs(funcSchema)
		if err != nil {
			return nil, fmt.Errorf("function[%d]: %w", i, err)
		}

		// Each function produces an independent score column
		funcScoreCol := fmt.Sprintf("_func_score_%d", i)
		funcScoreCols = append(funcScoreCols, funcScoreCol)

		repr.Operators = append(repr.Operators, OperatorRepr{
			Type:     OpTypeMap,
			Params:   make(map[string]interface{}),
			Function: funcRepr,
			Inputs:   inputCols,
			Outputs:  []string{funcScoreCol},
		})
	}

	// 2. Add score_combine map operation (dynamic inputs)
	boostMode := getParamValue(funcScore.Params, "boost_mode", "multiply")

	// Input columns: $score + all function score columns
	combineInputCols := append([]string{ScoreFieldName}, funcScoreCols...)

	combineParams := map[string]interface{}{
		"mode":        boostMode,
		"input_count": len(combineInputCols),
	}

	// Parse weights if present (for weighted mode)
	weightsStr := getParamValue(funcScore.Params, "weights", "")
	if weightsStr != "" {
		var weights []float64
		if err := json.Unmarshal([]byte(weightsStr), &weights); err == nil {
			combineParams["weights"] = weights
		}
	}

	repr.Operators = append(repr.Operators, OperatorRepr{
		Type:   OpTypeMap,
		Params: make(map[string]interface{}),
		Function: &FunctionRepr{
			Name:   "score_combine",
			Params: combineParams,
		},
		Inputs:  combineInputCols,
		Outputs: []string{ScoreFieldName},
	})

	// 3. Add sort operation
	repr.Operators = append(repr.Operators, OperatorRepr{
		Type: OpTypeSort,
		Params: map[string]interface{}{
			"column": ScoreFieldName,
			"desc":   true,
		},
		Inputs:  []string{ScoreFieldName},
		Outputs: []string{ScoreFieldName},
	})

	return repr, nil
}

// ParseFuncChainFromFunctionScore creates FuncChain from FunctionScore.
// This is a convenience method that combines functionScoreToRepr + funcChainFromRepr.
func ParseFuncChainFromFunctionScore(funcScore *schemapb.FunctionScore, alloc memory.Allocator) (*FuncChain, error) {
	repr, err := functionScoreToRepr(funcScore)
	if err != nil {
		return nil, err
	}
	return funcChainFromRepr(repr, alloc)
}

// =============================================================================
// Helper Functions for FunctionScore Conversion
// =============================================================================

// functionSchemaToFunctionReprWithInputs converts schemapb.FunctionSchema to FunctionRepr
// and extracts input column names. Input columns are now specified at the operator level.
func functionSchemaToFunctionReprWithInputs(funcSchema *schemapb.FunctionSchema) (*FunctionRepr, []string, error) {
	if funcSchema == nil {
		return nil, nil, fmt.Errorf("funcSchema is nil")
	}

	// Extract reranker name, input columns, and other params from funcSchema.Params
	params := make(map[string]interface{})
	var funcName string
	var inputCols []string

	for _, kv := range funcSchema.Params {
		switch kv.Key {
		case "reranker":
			funcName = kv.Value
		case "input_column":
			// Single input column (e.g., for decay function)
			inputCols = []string{kv.Value, ScoreFieldName}
		case "input_cols":
			// Multiple input columns (JSON array)
			var cols []string
			if err := json.Unmarshal([]byte(kv.Value), &cols); err == nil {
				inputCols = cols
			}
		default:
			// Try to parse as JSON first for complex values
			var jsonVal interface{}
			if err := json.Unmarshal([]byte(kv.Value), &jsonVal); err == nil {
				params[kv.Key] = jsonVal
			} else {
				params[kv.Key] = kv.Value
			}
		}
	}

	if funcName == "" {
		return nil, nil, fmt.Errorf("function name (reranker) not found in params")
	}

	if len(inputCols) == 0 {
		return nil, nil, fmt.Errorf("input columns not specified for function %s", funcName)
	}

	return &FunctionRepr{Name: funcName, Params: params}, inputCols, nil
}

// getParamValue gets a parameter value from KeyValuePair slice with a default.
func getParamValue(params []*commonpb.KeyValuePair, key, defaultVal string) string {
	for _, kv := range params {
		if kv.Key == key {
			return kv.Value
		}
	}
	return defaultVal
}
