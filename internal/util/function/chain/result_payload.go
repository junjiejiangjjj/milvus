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
	"encoding/base64"
	"math"
	"strconv"
	"strings"

	"github.com/apache/arrow/go/v17/arrow"
	"github.com/apache/arrow/go/v17/arrow/array"
	"github.com/apache/arrow/go/v17/arrow/memory"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

const resultFieldPrefix = "$result_field:"
const resultDistanceColumn = "$result_distances"
const resultRecallMetadata = "result_recalls"

func isResultField(name string) bool { return strings.HasPrefix(name, resultFieldPrefix) }

// FromSearchResultDataWithPayload includes response data as actual Arrow
// columns, alongside the logical inputs used for computation. Private names
// keep map outputs from overwriting the original response fields. The field
// descriptors contain schema only; no row IDs or external row lookup is used.
func FromSearchResultDataWithPayload(data *schemapb.SearchResultData, pool memory.Allocator, plan *DataFrameInputPlan) (*DataFrame, error) {
	if data == nil || pool == nil {
		return nil, merr.WrapErrServiceInternal("result payload: missing input or allocator")
	}
	offsets := make([]int64, len(data.Topks)+1)
	for i, n := range data.Topks {
		if n < 0 || n > math.MaxInt64-offsets[i] {
			return nil, merr.WrapErrServiceInternal("result payload: invalid chunk sizes")
		}
		offsets[i+1] = offsets[i] + n
	}
	total := offsets[len(offsets)-1]
	input, err := FromSearchResultData(data, pool, plan)
	if err != nil {
		return nil, err
	}
	defer input.Release()
	builder := NewDataFrameBuilder()
	defer builder.Release()
	builder.SetChunkSizes(data.Topks).CopyAllMetadata(input)
	for _, name := range input.ColumnNames() {
		if err := builder.AddColumnFrom(input, name); err != nil {
			return nil, err
		}
	}
	if len(data.FieldsData) > 0 {
		builder.result.resultFields = make(map[string]*schemapb.FieldData, len(data.FieldsData))
	}
	for _, field := range data.FieldsData {
		if err := ValidateResultField(field, int(total)); err != nil {
			return nil, err
		}
		name := resultFieldPrefix + strconv.FormatInt(field.GetFieldId(), 10) + ":" + field.GetFieldName()
		if err := importResultField(builder, field, name, offsets, pool); err != nil {
			return nil, err
		}
		// Empty protobuf payloads retain dimension/element type/wire variant.
		builder.result.resultFields[name] = typeutil.PrepareResultFieldData([]*schemapb.FieldData{field}, 0)[0]
	}
	if len(data.Distances) > 0 {
		if int64(len(data.Distances)) != total {
			return nil, merr.WrapErrServiceInternal("result payload: distance count differs")
		}
		chunks := importChunkedBatch(data.Distances, offsets, func(int) []bool { return nil }, array.NewFloat32Builder, pool)
		if err := builder.AddColumnFromChunks(resultDistanceColumn, chunks); err != nil {
			return nil, err
		}
	}
	// Recall is produced once per query, not once per hit. It must not be
	// sorted or truncated with the row columns.
	if len(data.Recalls) > 0 {
		if len(data.Recalls) != len(data.Topks) {
			return nil, merr.WrapErrServiceInternal("result payload: recall count differs from query count")
		}
		encoded, err := proto.Marshal(&schemapb.FloatArray{Data: data.Recalls})
		if err != nil {
			return nil, merr.WrapErrServiceInternalErr(err, "encode result recalls")
		}
		builder.SetMetadata(resultRecallMetadata, base64.StdEncoding.EncodeToString(encoded))
	}
	return builder.Build(), nil
}

func importResultField(builder *DataFrameBuilder, field *schemapb.FieldData, name string, offsets []int64, pool memory.Allocator) error {
	if _, err := ToArrowType(field.GetType()); err == nil {
		return importFieldDataWithName(builder, field, name, offsets, pool)
	}
	dt, err := resultFieldArrowType(field)
	if err != nil {
		return err
	}
	validity := typeutil.GetFieldDataValidData(field)
	compactVector := typeutil.IsSupportedNullableVectorType(field.GetType()) && len(validity) > 0
	chunks := make([]arrow.Array, 0, len(offsets)-1)
	defer func() {
		for _, chunk := range chunks {
			if chunk != nil {
				chunk.Release()
			}
		}
	}()
	physical := 0
	for q := 0; q+1 < len(offsets); q++ {
		b := array.NewBuilder(pool, dt)
		for row := offsets[q]; row < offsets[q+1]; row++ {
			valid := len(validity) == 0 || validity[row]
			idx := int(row)
			if compactVector {
				idx = physical
				if valid {
					physical++
				}
			}
			if !valid {
				b.AppendNull()
				continue
			}
			if err := appendResultValue(b, field, idx); err != nil {
				b.Release()
				return err
			}
		}
		chunks = append(chunks, b.NewArray())
		b.Release()
	}
	builder.SetFieldType(name, field.GetType()).SetFieldID(name, field.GetFieldId()).SetFieldNullable(name, len(validity) > 0)
	owned := chunks
	chunks = nil
	return builder.AddColumnFromChunks(name, owned)
}

func resultFieldArrowType(field *schemapb.FieldData) (arrow.DataType, error) {
	switch field.GetType() {
	case schemapb.DataType_FloatVector:
		dim := field.GetVectors().GetDim()
		if dim <= 0 || dim > math.MaxInt32 {
			return nil, merr.WrapErrServiceInternal("result payload: invalid float vector dimension")
		}
		return arrow.FixedSizeListOf(int32(dim), arrow.PrimitiveTypes.Float32), nil
	case schemapb.DataType_BinaryVector, schemapb.DataType_Float16Vector, schemapb.DataType_BFloat16Vector, schemapb.DataType_Int8Vector:
		width := field.GetVectors().GetDim()
		switch field.GetType() {
		case schemapb.DataType_BinaryVector:
			width /= 8
		case schemapb.DataType_Float16Vector, schemapb.DataType_BFloat16Vector:
			if width > math.MaxInt32/2 {
				return nil, merr.WrapErrServiceInternal("result payload: invalid vector dimension")
			}
			width *= 2
		}
		if width <= 0 || width > math.MaxInt32 {
			return nil, merr.WrapErrServiceInternal("result payload: invalid vector width")
		}
		return &arrow.FixedSizeBinaryType{ByteWidth: int(width)}, nil
	case schemapb.DataType_Geometry:
		if field.GetScalars().GetGeometryWktData() != nil {
			return arrow.BinaryTypes.String, nil
		}
		return arrow.BinaryTypes.Binary, nil
	case schemapb.DataType_JSON, schemapb.DataType_SparseFloatVector, schemapb.DataType_Array, schemapb.DataType_ArrayOfVector:
		return arrow.BinaryTypes.Binary, nil
	default:
		return nil, merr.WrapErrServiceInternalMsg("result payload: unsupported field type %s", field.GetType())
	}
}

func appendResultValue(builder array.Builder, field *schemapb.FieldData, row int) error {
	switch b := builder.(type) {
	case *array.FixedSizeListBuilder:
		dim := int(field.GetVectors().GetDim())
		b.Append(true)
		b.ValueBuilder().(*array.Float32Builder).AppendValues(field.GetVectors().GetFloatVector().Data[row*dim:(row+1)*dim], nil)
	case *array.FixedSizeBinaryBuilder:
		width := b.Type().(*arrow.FixedSizeBinaryType).ByteWidth
		var data []byte
		switch field.GetType() {
		case schemapb.DataType_BinaryVector:
			data = field.GetVectors().GetBinaryVector()
		case schemapb.DataType_Float16Vector:
			data = field.GetVectors().GetFloat16Vector()
		case schemapb.DataType_BFloat16Vector:
			data = field.GetVectors().GetBfloat16Vector()
		case schemapb.DataType_Int8Vector:
			data = field.GetVectors().GetInt8Vector()
		}
		b.Append(data[row*width : (row+1)*width])
	case *array.StringBuilder:
		b.Append(field.GetScalars().GetGeometryWktData().Data[row])
	case *array.BinaryBuilder:
		var data []byte
		var err error
		switch field.GetType() {
		case schemapb.DataType_JSON:
			data = field.GetScalars().GetJsonData().Data[row]
		case schemapb.DataType_Geometry:
			data = field.GetScalars().GetGeometryData().Data[row]
		case schemapb.DataType_SparseFloatVector:
			data = field.GetVectors().GetSparseFloatVector().Contents[row]
		// Nested Milvus array rows retain their own element type/nullability.
		// Their values, not row references, travel inside the Arrow Binary column.
		case schemapb.DataType_Array:
			data, err = proto.Marshal(field.GetScalars().GetArrayData().Data[row])
		case schemapb.DataType_ArrayOfVector:
			data, err = proto.Marshal(field.GetVectors().GetVectorArray().Data[row])
		}
		if err != nil {
			return merr.WrapErrServiceInternalErr(err, "encode nested result field")
		}
		b.Append(data)
	default:
		return merr.WrapErrServiceInternal("result payload: unexpected Arrow builder")
	}
	return nil
}

func exportResultField(df *DataFrame, name string) (*schemapb.FieldData, error) {
	template := df.resultFields[name]
	if template == nil {
		return nil, merr.WrapErrServiceInternal("result payload: descriptor is missing")
	}
	if _, err := ToArrowType(template.GetType()); err == nil {
		field, err := exportFieldData(df, name)
		if err != nil {
			return nil, err
		}
		field.FieldName = template.FieldName
		field.IsDynamic = template.IsDynamic
		if df.Column(name).NullN() > 0 {
			typeutil.SetFieldDataValidData(field, exportValidData(df.Column(name)))
		}
		return field, nil
	}
	field := proto.Clone(template).(*schemapb.FieldData)
	col := df.Column(name)
	if col == nil {
		return nil, merr.WrapErrServiceInternal("result payload: column is missing")
	}
	dtype, err := resultFieldArrowType(template)
	if err != nil {
		return nil, err
	}
	if !arrow.TypeEqual(dtype, col.DataType()) {
		return nil, merr.WrapErrServiceInternal("result payload: Arrow type changed")
	}
	validity := make([]bool, 0, col.Len())
	for _, chunk := range col.Chunks() {
		for i := 0; i < chunk.Len(); i++ {
			valid := !chunk.IsNull(i)
			validity = append(validity, valid)
			if typeutil.IsSupportedNullableVectorType(field.GetType()) && !valid {
				continue
			}
			if err := appendExportedResultValue(field, chunk, i, valid); err != nil {
				return nil, err
			}
		}
	}
	if df.fieldNullables[name] || col.NullN() > 0 {
		typeutil.SetFieldDataValidData(field, validity)
	}
	return field, nil
}

func appendExportedResultValue(field *schemapb.FieldData, chunk arrow.Array, row int, valid bool) error {
	switch a := chunk.(type) {
	case *array.FixedSizeList:
		begin, end := a.ValueOffsets(row)
		field.GetVectors().GetFloatVector().Data = append(field.GetVectors().GetFloatVector().Data, a.ListValues().(*array.Float32).Float32Values()[begin:end]...)
	case *array.FixedSizeBinary:
		data := a.Value(row)
		v := field.GetVectors()
		switch field.GetType() {
		case schemapb.DataType_BinaryVector:
			v.Data = &schemapb.VectorField_BinaryVector{BinaryVector: append(v.GetBinaryVector(), data...)}
		case schemapb.DataType_Float16Vector:
			v.Data = &schemapb.VectorField_Float16Vector{Float16Vector: append(v.GetFloat16Vector(), data...)}
		case schemapb.DataType_BFloat16Vector:
			v.Data = &schemapb.VectorField_Bfloat16Vector{Bfloat16Vector: append(v.GetBfloat16Vector(), data...)}
		case schemapb.DataType_Int8Vector:
			v.Data = &schemapb.VectorField_Int8Vector{Int8Vector: append(v.GetInt8Vector(), data...)}
		}
	case *array.String:
		value := ""
		if valid {
			value = a.Value(row)
		}
		field.GetScalars().GetGeometryWktData().Data = append(field.GetScalars().GetGeometryWktData().Data, value)
	case *array.Binary:
		var value []byte
		if valid {
			value = append([]byte(nil), a.Value(row)...)
		}
		switch field.GetType() {
		case schemapb.DataType_JSON:
			field.GetScalars().GetJsonData().Data = append(field.GetScalars().GetJsonData().Data, value)
		case schemapb.DataType_Geometry:
			field.GetScalars().GetGeometryData().Data = append(field.GetScalars().GetGeometryData().Data, value)
		case schemapb.DataType_SparseFloatVector:
			sparse := field.GetVectors().GetSparseFloatVector()
			sparse.Contents = append(sparse.Contents, value)
			sparse.Dim = max(sparse.Dim, typeutil.SparseFloatRowDim(value))
		case schemapb.DataType_Array:
			scalar := &schemapb.ScalarField{}
			if valid {
				if err := proto.Unmarshal(value, scalar); err != nil {
					return merr.WrapErrServiceInternalErr(err, "decode nested result field")
				}
			}
			field.GetScalars().GetArrayData().Data = append(field.GetScalars().GetArrayData().Data, scalar)
		case schemapb.DataType_ArrayOfVector:
			vector := &schemapb.VectorField{}
			if valid {
				if err := proto.Unmarshal(value, vector); err != nil {
					return merr.WrapErrServiceInternalErr(err, "decode nested vector result field")
				}
			}
			field.GetVectors().GetVectorArray().Data = append(field.GetVectors().GetVectorArray().Data, vector)
		}
	default:
		return merr.WrapErrServiceInternal("result payload: unexpected Arrow array")
	}
	return nil
}

func exportResultRecalls(df *DataFrame) ([]float32, error) {
	encoded, ok := df.metadata[resultRecallMetadata]
	if !ok {
		return nil, nil
	}
	raw, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, merr.WrapErrServiceInternalErr(err, "decode result recall metadata")
	}
	var values schemapb.FloatArray
	if err := proto.Unmarshal(raw, &values); err != nil {
		return nil, merr.WrapErrServiceInternalErr(err, "decode result recalls")
	}
	if len(values.Data) != df.NumChunks() {
		return nil, merr.WrapErrServiceInternal("result payload: recall query count changed")
	}
	return values.Data, nil
}

// ToSearchResultDataWithPayload exports only the original response fields
// carried through a PostProcess chain. Existing rerank exporters are unchanged.
func ToSearchResultDataWithPayload(df *DataFrame) (*schemapb.SearchResultData, error) {
	result, err := ToSearchResultDataWithOptions(df, &ExportOptions{SkipColumns: df.ColumnNames()})
	if err != nil {
		return nil, err
	}
	for _, name := range df.ColumnNames() {
		if isResultField(name) {
			field, err := exportResultField(df, name)
			if err != nil {
				return nil, err
			}
			result.FieldsData = append(result.FieldsData, field)
		}
	}
	if col := df.Column(resultDistanceColumn); col != nil {
		values, err := exportChunkedValues[float32, *array.Float32](col, resultDistanceColumn)
		if err != nil {
			return nil, err
		}
		result.Distances = values
	}
	result.Recalls, err = exportResultRecalls(df)
	if err != nil {
		return nil, err
	}
	return result, nil
}
