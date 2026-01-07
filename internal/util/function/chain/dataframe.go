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
	"fmt"

	"github.com/apache/arrow/go/v17/arrow"
	// "github.com/apache/arrow/go/v17/arrow/array"
	"github.com/apache/arrow/go/v17/arrow/memory"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
)

const (
	IDFieldName    = "$id"
	ScoreFieldName = "$score"
	QueryFieldName = "$query"
)

type ChainContext struct {
}

type DataFrame struct {
	pool       *memory.Allocator
	table      arrow.Table
	chunkCount int
	nameIndex  map[string]int
}

func (df *DataFrame) release() {
}

func (df *DataFrame) GetChunkCount() int {
	return df.chunkCount
}

func (df *DataFrame) GetColumns(colNames []string, chunkIndex int) ([]arrow.Array, error) {
	if chunkIndex < 0 || chunkIndex >= df.chunkCount {
		return nil, fmt.Errorf("chunk index out of range: %d", chunkIndex)
	}
	columns := make([]arrow.Array, 0, len(colNames))
	for _, colName := range colNames {
		colIdx, exists := df.nameIndex[colName]
		if !exists {
			return nil, fmt.Errorf("column %s not found", colName)
		}
		columns = append(columns, df.table.Column(colIdx).Data().Chunk(chunkIndex))
	}
	return columns, nil
}

func FromSearchResultProto(collSchema *schemapb.CollectionSchema, searchData *milvuspb.SearchResults) (*DataFrame, error) {
	return nil, nil
}

func (df *DataFrame) ToSearchResultProto() (*milvuspb.SearchResults, error) {
	df.release()
	return nil, nil
}

/*

func FromSearchResultProto(collSchema *schemapb.CollectionSchema, searchData *milvuspb.SearchResults) (*DataFrame, error) {

	resultData := searchData.Results
	topks := resultData.GetTopks()

	// Build arrow schema and arrays
	pool := memory.DefaultAllocator
	fields := make([]arrow.Field, 0)
	arrays := make([]arrow.Array, 0)

	// Add ID field
	ids := resultData.GetIds()
	if ids != nil {
		idArray, err := buildIDArray(pool, ids, totalRows)
		if err != nil {
			return nil, fmt.Errorf("failed to build id array: %w", err)
		}
		fields = append(fields, arrow.Field{Name: "id", Type: idArray.DataType(), Nullable: false})
		arrays = append(arrays, idArray)
	}

	// Add score field
	scores := resultData.GetScores()
	if len(scores) > 0 {
		scoreBuilder := array.NewFloat32Builder(pool)
		defer scoreBuilder.Release()
		scoreBuilder.AppendValues(scores, nil)
		scoreArray := scoreBuilder.NewArray()
		fields = append(fields, arrow.Field{Name: "score", Type: arrow.PrimitiveTypes.Float32, Nullable: false})
		arrays = append(arrays, scoreArray)
	}

	// Add field data
	fieldsData := resultData.GetFieldsData()
	for _, fieldData := range fieldsData {
		fieldArray, err := buildFieldArray(pool, fieldData, totalRows)
		if err != nil {
			return nil, fmt.Errorf("failed to build field array for %s: %w", fieldData.GetFieldName(), err)
		}
		fields = append(fields, arrow.Field{
			Name:     fieldData.GetFieldName(),
			Type:     fieldArray.DataType(),
			Nullable: true,
		})
		arrays = append(arrays, fieldArray)
	}

	// Create arrow schema
	schema := arrow.NewSchema(fields, nil)

	// Create arrow record
	record := array.NewRecord(schema, arrays, totalRows)
	defer record.Release()

	// Split records by topks
	records := make([]arrow.Record, 0, len(topks))
	offset := int64(0)
	for _, topk := range topks {
		if topk == 0 {
			continue
		}
		slicedRecord := record.NewSlice(offset, offset+topk)
		records = append(records, slicedRecord)
		offset += topk
	}

	// Create arrow table from records
	table := array.NewTableFromRecords(schema, records)

	// Release sliced records
	for _, rec := range records {
		rec.Release()
	}

	return &DataFrame{
		pool:  &pool,
		table: &table,
	}, nil
}

/*
// buildIDArray builds arrow array from IDs
func buildIDArray(pool memory.Allocator, ids *schemapb.IDs, topks []int64) ([]arrow.Array, error) {
	switch ids.IdField.(type) {
	case *schemapb.IDs_IntId:
		intIDs := ids.GetIntId().GetData()
		intArrays := make([]*arrow.Array, 0, len(topks))
		for _, topk := range topks {
			intArray := array.NewInt64Builder(pool)
			defer intArray.Release()
			intArray.AppendValues(intIDs[:topk], nil)
			intArrays = append(intArrays, intArray.NewInt64Array())
		}
		return intArrays, nil
	case *schemapb.IDs_StrId:
		strIDs := ids.GetStrId().GetData()
		builder := array.NewStringBuilder(pool)
		defer builder.Release()
		builder.AppendValues(strIDs, nil)
		return builder.NewArray(), nil
	default:
		return nil, fmt.Errorf("ids type not supported")
	}
}

func buildInt64Array(pool memory.Allocator, data []int64, topks []int64) ([]arrow.Array, error) {
	for _, topk := range topks {
		builder := array.NewInt64Builder(pool)
		defer builder.Release()
		builder.AppendValues(data[:topk], nil)
		intArrays = append(intArrays, builder.NewArray())
	}
	return intArrays, nil
}

// buildFieldArray builds arrow array from FieldData
func buildFieldArray(pool memory.Allocator, fieldData *schemapb.FieldData, numRows int64) (arrow.Array, error) {
	switch fieldData.GetType() {
	case schemapb.DataType_Bool:
		data := fieldData.GetScalars().GetBoolData().GetData()
		if int64(len(data)) != numRows {
			return nil, fmt.Errorf("field data count mismatch: expected %d, got %d", numRows, len(data))
		}
		builder := array.NewBooleanBuilder(pool)
		defer builder.Release()
		builder.AppendValues(data, nil)
		return builder.NewArray(), nil

	case schemapb.DataType_Int8:
		data := fieldData.GetScalars().GetIntData().GetData()
		if int64(len(data)) != numRows {
			return nil, fmt.Errorf("field data count mismatch: expected %d, got %d", numRows, len(data))
		}
		builder := array.NewInt8Builder(pool)
		defer builder.Release()
		for _, v := range data {
			builder.Append(int8(v))
		}
		return builder.NewArray(), nil

	case schemapb.DataType_Int16:
		data := fieldData.GetScalars().GetIntData().GetData()
		if int64(len(data)) != numRows {
			return nil, fmt.Errorf("field data count mismatch: expected %d, got %d", numRows, len(data))
		}
		builder := array.NewInt16Builder(pool)
		defer builder.Release()
		for _, v := range data {
			builder.Append(int16(v))
		}
		return builder.NewArray(), nil

	case schemapb.DataType_Int32:
		data := fieldData.GetScalars().GetIntData().GetData()
		if int64(len(data)) != numRows {
			return nil, fmt.Errorf("field data count mismatch: expected %d, got %d", numRows, len(data))
		}
		builder := array.NewInt32Builder(pool)
		defer builder.Release()
		builder.AppendValues(data, nil)
		return builder.NewArray(), nil

	case schemapb.DataType_Int64:
		data := fieldData.GetScalars().GetLongData().GetData()
		if int64(len(data)) != numRows {
			return nil, fmt.Errorf("field data count mismatch: expected %d, got %d", numRows, len(data))
		}
		builder := array.NewInt64Builder(pool)
		defer builder.Release()
		builder.AppendValues(data, nil)
		return builder.NewArray(), nil

	case schemapb.DataType_Float:
		data := fieldData.GetScalars().GetFloatData().GetData()
		if int64(len(data)) != numRows {
			return nil, fmt.Errorf("field data count mismatch: expected %d, got %d", numRows, len(data))
		}
		builder := array.NewFloat32Builder(pool)
		defer builder.Release()
		builder.AppendValues(data, nil)
		return builder.NewArray(), nil

	case schemapb.DataType_Double:
		data := fieldData.GetScalars().GetDoubleData().GetData()
		if int64(len(data)) != numRows {
			return nil, fmt.Errorf("field data count mismatch: expected %d, got %d", numRows, len(data))
		}
		builder := array.NewFloat64Builder(pool)
		defer builder.Release()
		builder.AppendValues(data, nil)
		return builder.NewArray(), nil

	case schemapb.DataType_String, schemapb.DataType_VarChar, schemapb.DataType_Text:
		data := fieldData.GetScalars().GetStringData().GetData()
		if int64(len(data)) != numRows {
			return nil, fmt.Errorf("field data count mismatch: expected %d, got %d", numRows, len(data))
		}
		builder := array.NewStringBuilder(pool)
		defer builder.Release()
		builder.AppendValues(data, nil)
		return builder.NewArray(), nil

	case schemapb.DataType_JSON:
		data := fieldData.GetScalars().GetJsonData().GetData()
		if int64(len(data)) != numRows {
			return nil, fmt.Errorf("field data count mismatch: expected %d, got %d", numRows, len(data))
		}
		builder := array.NewBinaryBuilder(pool, arrow.BinaryTypes.Binary)
		defer builder.Release()
		for _, v := range data {
			builder.Append(v)
		}
		return builder.NewArray(), nil

	default:
		return nil, fmt.Errorf("unsupported field data type: %s", fieldData.GetType().String())
	}
}

func (d *DataFrame) Map() (*DataFrame, error) {
	return nil, nil
}

func (d *DataFrame) Filter() (*DataFrame, error) {
	return nil, nil
}

func (d *DataFrame) Sort() (*DataFrame, error) {
	return nil, nil
}

func toArrowSchema(collSchema *schemapb.CollectionSchema, allInputFieldNames []string) (*arrow.Schema, error) {
	arrowFields := make([]arrow.Field, 0, len(allInputFieldNames))
	fieldMap := make(map[string]*schemapb.FieldSchema)
	for _, field := range collSchema.GetFields() {
		fieldMap[field.GetName()] = field
	}
	for _, fieldName := range allInputFieldNames {
		field, ok := fieldMap[fieldName]
		if !ok {
			return nil, fmt.Errorf("field %s not found in collection schema", fieldName)
		}
		arrowType, err := toArrowType(field.GetDataType())
		if err != nil {
			return nil, fmt.Errorf("failed to convert field %s to arrow type: %w", fieldName, err)
		}
		arrowFields = append(arrowFields, arrow.Field{
			Name:     fieldName,
			Type:     arrowType,
			Nullable: field.GetNullable(),
		})
	}
	return arrow.NewSchema(arrowFields, nil), nil
}
*/

func toArrowType(t schemapb.DataType) (arrow.DataType, error) {
	// TODO: support more data types, such as Array, JSON, Geometry, etc.
	switch t {
	case schemapb.DataType_Bool:
		return arrow.FixedWidthTypes.Boolean, nil
	case schemapb.DataType_Int8:
		return arrow.PrimitiveTypes.Int8, nil
	case schemapb.DataType_Int16:
		return arrow.PrimitiveTypes.Int16, nil
	case schemapb.DataType_Int32:
		return arrow.PrimitiveTypes.Int32, nil
	case schemapb.DataType_Int64:
		return arrow.PrimitiveTypes.Int64, nil
	case schemapb.DataType_Float:
		return arrow.PrimitiveTypes.Float32, nil
	case schemapb.DataType_Double:
		return arrow.PrimitiveTypes.Float64, nil
	case schemapb.DataType_String, schemapb.DataType_VarChar, schemapb.DataType_Text:
		return arrow.BinaryTypes.String, nil
	default:
		return nil, fmt.Errorf("unsupported data type: %s", t.String())
	}
}
