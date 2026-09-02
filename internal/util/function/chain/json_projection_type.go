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
	"math"

	"github.com/apache/arrow/go/v17/arrow"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// JSONValueKind is the exact value family extracted from one JSON path.
type JSONValueKind int

const (
	JSONValueNull JSONValueKind = iota
	JSONValueBool
	JSONValueInt64
	JSONValueUint64
	JSONValueFloat64
	JSONValueString
	JSONValueArray
	JSONValueObject
)

// ExtractedJSONValue carries an exact JSON scalar or a raw complex value.
type ExtractedJSONValue struct {
	Kind JSONValueKind
	I64  int64
	U64  uint64
	F64  float64
	Bool bool
	Str  string
	Raw  []byte
}

// JSONTypeAccumulator infers one Arrow-compatible Milvus type from non-null JSON values.
type JSONTypeAccumulator struct {
	dataType schemapb.DataType
}

// Add merges one extracted JSON value into the inferred type.
func (a *JSONTypeAccumulator) Add(value ExtractedJSONValue) error {
	observed, err := inferredDataTypeForJSONValue(value)
	if err != nil {
		return err
	}
	merged, err := MergeJSONProjectionDataTypes(a.dataType, observed)
	if err != nil {
		return err
	}
	a.dataType = merged
	return nil
}

// DataType returns the inferred type. An all-null/missing input has no type.
func (a *JSONTypeAccumulator) DataType() (schemapb.DataType, error) {
	if a == nil || a.dataType == schemapb.DataType_None {
		return schemapb.DataType_None, merr.WrapErrFunctionFailedMsg(
			"cannot infer JSON path data type: all values are null or missing")
	}
	return a.dataType, nil
}

func inferredDataTypeForJSONValue(value ExtractedJSONValue) (schemapb.DataType, error) {
	switch value.Kind {
	case JSONValueNull:
		return schemapb.DataType_None, nil
	case JSONValueBool:
		return schemapb.DataType_Bool, nil
	case JSONValueInt64:
		return schemapb.DataType_Int64, nil
	case JSONValueUint64:
		if value.U64 > math.MaxInt64 {
			return schemapb.DataType_None, merr.WrapErrFunctionFailedMsg(
				"cannot infer JSON path data type from uint64 value beyond int64 range; specify DOUBLE or JSON")
		}
		return schemapb.DataType_Int64, nil
	case JSONValueFloat64:
		return schemapb.DataType_Double, nil
	case JSONValueString:
		return schemapb.DataType_VarChar, nil
	case JSONValueArray, JSONValueObject:
		return schemapb.DataType_None, merr.WrapErrFunctionFailedMsg(
			"cannot infer scalar data type from JSON array or object; specify JSON")
	default:
		return schemapb.DataType_None, merr.WrapErrFunctionFailedMsg(
			"cannot infer data type from unknown JSON value kind %d", value.Kind)
	}
}

// MergeJSONProjectionDataTypes reconciles locally inferred projection types.
func MergeJSONProjectionDataTypes(left, right schemapb.DataType) (schemapb.DataType, error) {
	if left == schemapb.DataType_None {
		return right, nil
	}
	if right == schemapb.DataType_None || left == right {
		return left, nil
	}
	if isIntegerDataType(left) && isFloatingDataType(right) ||
		isFloatingDataType(left) && isIntegerDataType(right) {
		return schemapb.DataType_Double, nil
	}
	if isStringDataType(left) && isStringDataType(right) {
		return schemapb.DataType_VarChar, nil
	}
	return schemapb.DataType_None, merr.WrapErrFunctionFailedMsg(
		"inconsistent JSON path data types: %s and %s", left.String(), right.String())
}

// ValidateJSONProjectionValue validates one non-null value against an explicit hint.
func ValidateJSONProjectionValue(value ExtractedJSONValue, hint schemapb.DataType) error {
	if value.Kind == JSONValueNull || hint == schemapb.DataType_JSON {
		return nil
	}
	valid := false
	switch hint {
	case schemapb.DataType_Bool:
		valid = value.Kind == JSONValueBool
	case schemapb.DataType_Int8:
		valid = value.Kind == JSONValueInt64 && value.I64 >= math.MinInt8 && value.I64 <= math.MaxInt8
	case schemapb.DataType_Int16:
		valid = value.Kind == JSONValueInt64 && value.I64 >= math.MinInt16 && value.I64 <= math.MaxInt16
	case schemapb.DataType_Int32:
		valid = value.Kind == JSONValueInt64 && value.I64 >= math.MinInt32 && value.I64 <= math.MaxInt32
	case schemapb.DataType_Int64:
		valid = value.Kind == JSONValueInt64
	case schemapb.DataType_Float:
		valid = isJSONNumber(value.Kind) && jsonNumberFitsFloat32(value)
	case schemapb.DataType_Double:
		valid = value.Kind == JSONValueInt64 || value.Kind == JSONValueUint64 || value.Kind == JSONValueFloat64
	case schemapb.DataType_String, schemapb.DataType_VarChar, schemapb.DataType_Text:
		valid = value.Kind == JSONValueString
	default:
		return merr.WrapErrParameterInvalidMsg("unsupported JSON path data type hint %s", hint.String())
	}
	if !valid {
		return merr.WrapErrFunctionFailedMsg(
			"JSON path value type %s is incompatible with data type hint %s",
			jsonValueKindName(value.Kind), hint.String())
	}
	return nil
}

func isJSONNumber(kind JSONValueKind) bool {
	return kind == JSONValueInt64 || kind == JSONValueUint64 || kind == JSONValueFloat64
}

func jsonNumberFitsFloat32(value ExtractedJSONValue) bool {
	if value.Kind != JSONValueFloat64 {
		return true
	}
	return !math.IsNaN(value.F64) && !math.IsInf(value.F64, 0) &&
		value.F64 >= -math.MaxFloat32 && value.F64 <= math.MaxFloat32
}

// JSONProjectionArrowType returns the Arrow type used for a resolved projection type.
func JSONProjectionArrowType(dataType schemapb.DataType) (arrow.DataType, error) {
	switch dataType {
	case schemapb.DataType_None:
		return arrow.Null, nil
	case schemapb.DataType_JSON:
		return arrow.BinaryTypes.Binary, nil
	default:
		return ToArrowType(dataType)
	}
}

func isIntegerDataType(dataType schemapb.DataType) bool {
	switch dataType {
	case schemapb.DataType_Int8, schemapb.DataType_Int16, schemapb.DataType_Int32, schemapb.DataType_Int64:
		return true
	default:
		return false
	}
}

func isFloatingDataType(dataType schemapb.DataType) bool {
	return dataType == schemapb.DataType_Float || dataType == schemapb.DataType_Double
}

func isStringDataType(dataType schemapb.DataType) bool {
	switch dataType {
	case schemapb.DataType_String, schemapb.DataType_VarChar, schemapb.DataType_Text:
		return true
	default:
		return false
	}
}

func jsonValueKindName(kind JSONValueKind) string {
	switch kind {
	case JSONValueNull:
		return "Null"
	case JSONValueBool:
		return "Bool"
	case JSONValueInt64:
		return "Int64"
	case JSONValueUint64:
		return "Uint64"
	case JSONValueFloat64:
		return "Float64"
	case JSONValueString:
		return "String"
	case JSONValueArray:
		return "Array"
	case JSONValueObject:
		return "Object"
	default:
		return "Unknown"
	}
}
