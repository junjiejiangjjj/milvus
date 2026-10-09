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
package chain

import (
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/pkg/v3/util/funcutil"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
	"google.golang.org/protobuf/reflect/protoreflect"
)

// ValidateResultField checks the shape of an internally produced response field.
func ValidateResultField(field *schemapb.FieldData, rows int) error {
	if field == nil {
		return merr.WrapErrServiceInternal("chain result contains a nil field")
	}
	valid := typeutil.GetFieldDataValidData(field)
	if len(valid) != 0 && len(valid) != rows {
		return merr.WrapErrServiceInternalMsg("chain result field %q validity count does not match rows", field.GetFieldName())
	}
	// Check message presence before using the row-count and row-copy helpers,
	// which assume concrete scalar/vector payload messages are non-nil.
	var payload protoreflect.Message
	if scalar := field.GetScalars(); scalar != nil {
		payload = scalar.ProtoReflect()
	} else if vector := field.GetVectors(); vector != nil {
		payload = vector.ProtoReflect()
	} else {
		return merr.WrapErrServiceInternalMsg("chain result field %q has no scalar/vector payload", field.GetFieldName())
	}
	oneof := payload.Descriptor().Oneofs().ByName("data")
	selected := payload.WhichOneof(oneof)
	if selected == nil || (selected.Kind() == protoreflect.MessageKind && !payload.Get(selected).Message().IsValid()) {
		return merr.WrapErrServiceInternalMsg("chain result field %q has no payload data", field.GetFieldName())
	}
	expectedPayload := map[schemapb.DataType]protoreflect.Name{
		schemapb.DataType_Bool: "bool_data",
		schemapb.DataType_Int8: "int_data", schemapb.DataType_Int16: "int_data", schemapb.DataType_Int32: "int_data",
		schemapb.DataType_Int64: "long_data", schemapb.DataType_Timestamptz: "timestamptz_data",
		schemapb.DataType_Float: "float_data", schemapb.DataType_Double: "double_data",
		schemapb.DataType_String: "string_data", schemapb.DataType_VarChar: "string_data", schemapb.DataType_Text: "string_data",
		schemapb.DataType_JSON: "json_data", schemapb.DataType_Array: "array_data", schemapb.DataType_Geometry: "geometry_data",
		schemapb.DataType_FloatVector: "float_vector", schemapb.DataType_BinaryVector: "binary_vector",
		schemapb.DataType_Float16Vector: "float16_vector", schemapb.DataType_BFloat16Vector: "bfloat16_vector",
		schemapb.DataType_SparseFloatVector: "sparse_float_vector", schemapb.DataType_Int8Vector: "int8_vector",
		schemapb.DataType_ArrayOfVector: "vector_array",
	}[field.GetType()]
	geometryWKT := field.GetType() == schemapb.DataType_Geometry && selected.Name() == "geometry_wkt_data"
	if expectedPayload == "" || (selected.Name() != expectedPayload && !geometryWKT) {
		return merr.WrapErrServiceInternalMsg("chain result field %q payload does not match type %s", field.GetFieldName(), field.GetType())
	}
	var count uint64
	var err error
	switch {
	case geometryWKT:
		// Query/requery has already converted stored WKB to response WKT.
		count = uint64(len(field.GetScalars().GetGeometryWktData().GetData()))
	case field.GetType() == schemapb.DataType_ArrayOfVector:
		count = uint64(len(field.GetVectors().GetVectorArray().GetData()))
	default:
		count, err = funcutil.GetNumRowOfFieldData(field)
	}
	if err != nil {
		// This helper validates client data elsewhere, but here the source is
		// an internal search result. Deliberately classify a broken result as System.
		return merr.WrapErrServiceInternalErr(err, "invalid chain result field %q", field.GetFieldName())
	}
	if count != uint64(rows) {
		return merr.WrapErrServiceInternalMsg("chain result field %q has %d rows, expected %d", field.GetFieldName(), count, rows)
	}
	return nil
}
