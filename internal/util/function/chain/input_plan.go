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

package chain

import (
	"strings"

	"github.com/apache/arrow/go/v17/arrow"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/parser/planparserv2"
	"github.com/milvus-io/milvus/internal/util/function/chain/types"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

// ResolvedChainInput is the schema-aware intermediate representation of one
// Function Chain input. It maps the logical column used by an operator to the
// physical schema field that must be read and, for JSON fields, the nested path
// that must be projected into a runtime Arrow column.
type ResolvedChainInput struct {
	// LogicalName is the complete name referenced by the chain and used for the
	// projected runtime column, for example metadata["price"] or $meta["ctr"].
	LogicalName string
	// SourceFieldID identifies the physical schema field to fetch. Multiple JSON
	// paths may share the same source field ID.
	SourceFieldID int64
	// FieldName is the physical root field name, for example metadata or $meta.
	FieldName string
	// DataType is the schema type of the physical root field, not the inferred
	// type of a nested JSON value.
	DataType schemapb.DataType
	// NestedPath contains the JSON keys below the root field. It is empty for an
	// ordinary scalar field or when the complete JSON root is requested.
	NestedPath []string
	// DataTypeHint controls the projected JSON-path column type. None requests
	// runtime inference; a complete JSON root is normalized to JSON.
	DataTypeHint schemapb.DataType
}

// DataFrameInputPlan contains the schema-resolved inputs needed to materialize
// a Function Chain DataFrame. Consumers decide how to load each input from its
// DataType and may group entries by SourceFieldID to batch JSON projections.
type DataFrameInputPlan struct {
	Inputs []ResolvedChainInput
}

// PhysicalFieldIDs returns deduplicated physical field IDs needed to build the DataFrame.
func (p *DataFrameInputPlan) PhysicalFieldIDs() []int64 {
	if p == nil {
		return nil
	}
	ids := make([]int64, 0, len(p.Inputs))
	seen := make(map[int64]struct{}, cap(ids))
	for _, input := range p.Inputs {
		if _, ok := seen[input.SourceFieldID]; ok {
			continue
		}
		seen[input.SourceFieldID] = struct{}{}
		ids = append(ids, input.SourceFieldID)
	}
	return ids
}

// PhysicalFieldNames returns deduplicated physical field names needed to build the DataFrame.
func (p *DataFrameInputPlan) PhysicalFieldNames() []string {
	if p == nil {
		return nil
	}
	names := make([]string, 0, len(p.Inputs))
	seen := make(map[string]struct{}, cap(names))
	for _, input := range p.Inputs {
		if _, ok := seen[input.FieldName]; ok {
			continue
		}
		seen[input.FieldName] = struct{}{}
		names = append(names, input.FieldName)
	}
	return names
}

// CompileDataFrameInputPlan resolves schema-backed chain inputs and builds their physical fetch plan.
func CompileDataFrameInputPlan(repr *ChainRepr, schema *schemapb.CollectionSchema) (*DataFrameInputPlan, error) {
	if repr == nil {
		return nil, merr.WrapErrParameterInvalidMsg("function chain repr is nil")
	}
	schemaHelper, err := typeutil.CreateSchemaHelper(schema)
	if err != nil {
		return nil, merr.Wrap(err, "function chain input plan: create schema helper")
	}

	plan := &DataFrameInputPlan{Inputs: make([]ResolvedChainInput, 0)}
	produced := make(map[string]struct{})
	inputOffsets := make(map[string]int)

	for opIdx := range repr.Operators {
		op := &repr.Operators[opIdx]
		if err := normalizeOperatorInputDataTypes(op); err != nil {
			return nil, merr.Wrapf(err, "op[%d]", opIdx)
		}

		for inputIdx, name := range op.Inputs {
			if _, ok := produced[name]; ok {
				continue
			}
			hint := op.InputDataTypes[inputIdx]
			if isRuntimeFunctionChainInput(name) {
				if hint != schemapb.DataType_None {
					return nil, merr.WrapErrParameterInvalidMsg(
						"op[%d] input %q: system input does not accept data type hint %s",
						opIdx, name, hint.String())
				}
				continue
			}

			resolved, err := resolveSchemaChainInput(schemaHelper, name, hint)
			if err != nil {
				return nil, merr.Wrapf(err, "op[%d] input %q", opIdx, name)
			}
			input := *resolved
			key := resolvedInputIdentity(input)
			if offset, ok := inputOffsets[key]; ok {
				if input.DataType == schemapb.DataType_JSON {
					mergedHint, err := mergeInputHints(plan.Inputs[offset].DataTypeHint, input.DataTypeHint)
					if err != nil {
						return nil, merr.Wrapf(err, "op[%d] input %q", opIdx, name)
					}
					plan.Inputs[offset].DataTypeHint = mergedHint
				}
				continue
			}
			inputOffsets[key] = len(plan.Inputs)
			plan.Inputs = append(plan.Inputs, input)
		}

		for _, output := range op.Outputs {
			produced[output] = struct{}{}
		}
	}
	return plan, nil
}

func normalizeOperatorInputDataTypes(op *OperatorRepr) error {
	if op.InputDataTypes == nil {
		op.InputDataTypes = make([]schemapb.DataType, len(op.Inputs))
		return nil
	}
	if len(op.InputDataTypes) != len(op.Inputs) {
		return merr.WrapErrParameterInvalidMsg(
			"input data types count %d does not match input count %d",
			len(op.InputDataTypes), len(op.Inputs))
	}
	return nil
}

func isRuntimeFunctionChainInput(name string) bool {
	switch name {
	case types.IDFieldName, types.ScoreFieldName:
		return true
	default:
		return false
	}
}

func resolveSchemaChainInput(
	schemaHelper *typeutil.SchemaHelper,
	name string,
	hint schemapb.DataType,
) (*ResolvedChainInput, error) {
	if IsFunctionChainSystemName(name) && !isExplicitMetaInput(name) {
		return nil, merr.WrapErrParameterInvalidMsg("unsupported function chain system input %q", name)
	}

	var columnInfo *planpb.ColumnInfo
	err := planparserv2.ParseIdentifier(schemaHelper, name, func(expr *planpb.Expr) error {
		columnInfo = expr.GetColumnExpr().GetInfo()
		return nil
	})
	if err != nil {
		return nil, merr.Wrap(err, "resolve function chain input")
	}
	if columnInfo == nil {
		return nil, merr.WrapErrParameterInvalidMsg("function chain input %q did not resolve to a column", name)
	}
	field, err := schemaHelper.GetFieldFromID(columnInfo.GetFieldId())
	if err != nil {
		return nil, merr.Wrap(err, "resolve function chain input field")
	}
	if field.GetIsDynamic() && !isExplicitMetaInput(name) {
		return nil, merr.WrapErrParameterInvalidMsg(
			"dynamic field input %q must use explicit %s[...] syntax", name, common.MetaFieldName)
	}
	if len(columnInfo.GetNestedPath()) > 0 && field.GetDataType() != schemapb.DataType_JSON {
		return nil, merr.WrapErrParameterInvalidMsg(
			"function chain input %q uses nested path on unsupported field type %s",
			name, field.GetDataType().String())
	}
	if err := validateResolvedInputHint(field.GetDataType(), columnInfo.GetNestedPath(), hint); err != nil {
		return nil, err
	}
	if field.GetDataType() != schemapb.DataType_JSON {
		if _, err := ToArrowType(field.GetDataType()); err != nil {
			return nil, merr.WrapErrParameterInvalidMsg(
				"function chain input %q has unsupported field type %s", name, field.GetDataType().String())
		}
	}

	if field.GetDataType() == schemapb.DataType_JSON && len(columnInfo.GetNestedPath()) == 0 && hint == schemapb.DataType_None {
		hint = schemapb.DataType_JSON
	} else if field.GetDataType() == schemapb.DataType_JSON {
		hint = canonicalJSONProjectionHint(hint)
	}
	return &ResolvedChainInput{
		LogicalName:   name,
		SourceFieldID: field.GetFieldID(),
		FieldName:     field.GetName(),
		DataType:      field.GetDataType(),
		NestedPath:    append([]string(nil), columnInfo.GetNestedPath()...),
		DataTypeHint:  hint,
	}, nil
}

func isExplicitMetaInput(name string) bool {
	return name == common.MetaFieldName || strings.HasPrefix(name, common.MetaFieldName+"[")
}

func validateResolvedInputHint(fieldType schemapb.DataType, nestedPath []string, hint schemapb.DataType) error {
	if fieldType != schemapb.DataType_JSON {
		if hint == schemapb.DataType_None {
			return nil
		}
		fieldArrowType, fieldErr := ToArrowType(fieldType)
		hintArrowType, hintErr := ToArrowType(hint)
		if fieldErr != nil || hintErr != nil || !arrow.TypeEqual(fieldArrowType, hintArrowType) {
			return merr.WrapErrParameterInvalidMsg(
				"data type hint %s is incompatible with schema field type %s", hint.String(), fieldType.String())
		}
		return nil
	}
	if len(nestedPath) == 0 {
		if hint != schemapb.DataType_None && hint != schemapb.DataType_JSON {
			return merr.WrapErrParameterInvalidMsg(
				"JSON root input only accepts JSON data type hint, got %s", hint.String())
		}
		return nil
	}
	if !isSupportedJSONProjectionHint(hint) {
		return merr.WrapErrParameterInvalidMsg("unsupported JSON path data type hint %s", hint.String())
	}
	return nil
}

func isSupportedJSONProjectionHint(hint schemapb.DataType) bool {
	switch hint {
	case schemapb.DataType_None,
		schemapb.DataType_Bool,
		schemapb.DataType_Int8,
		schemapb.DataType_Int16,
		schemapb.DataType_Int32,
		schemapb.DataType_Int64,
		schemapb.DataType_Float,
		schemapb.DataType_Double,
		schemapb.DataType_String,
		schemapb.DataType_VarChar,
		schemapb.DataType_Text,
		schemapb.DataType_JSON:
		return true
	default:
		return false
	}
}

func canonicalJSONProjectionHint(hint schemapb.DataType) schemapb.DataType {
	if isStringDataType(hint) {
		return schemapb.DataType_VarChar
	}
	return hint
}

func resolvedInputIdentity(input ResolvedChainInput) string {
	return input.LogicalName
}

func mergeInputHints(left, right schemapb.DataType) (schemapb.DataType, error) {
	if left == schemapb.DataType_None {
		return right, nil
	}
	if right == schemapb.DataType_None || left == right {
		return left, nil
	}
	return schemapb.DataType_None, merr.WrapErrParameterInvalidMsg(
		"conflicting data type hints %s and %s for the same JSON path", left.String(), right.String())
}
