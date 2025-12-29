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
	"maps"
	"strings"

	"github.com/apache/arrow/go/v17/arrow"
	"github.com/apache/arrow/go/v17/arrow/array"
	"github.com/apache/arrow/go/v17/arrow/memory"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
)

// Special field name constants
const (
	IDFieldName    = "$id"
	ScoreFieldName = "$score"
)

// ToArrowType converts Milvus DataType to Arrow DataType.
func ToArrowType(t schemapb.DataType) (arrow.DataType, error) {
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

// ToMilvusType converts Arrow DataType to Milvus DataType.
func ToMilvusType(t arrow.DataType) (schemapb.DataType, error) {
	switch t.ID() {
	case arrow.BOOL:
		return schemapb.DataType_Bool, nil
	case arrow.INT8:
		return schemapb.DataType_Int8, nil
	case arrow.INT16:
		return schemapb.DataType_Int16, nil
	case arrow.INT32:
		return schemapb.DataType_Int32, nil
	case arrow.INT64:
		return schemapb.DataType_Int64, nil
	case arrow.FLOAT32:
		return schemapb.DataType_Float, nil
	case arrow.FLOAT64:
		return schemapb.DataType_Double, nil
	case arrow.STRING:
		return schemapb.DataType_VarChar, nil
	default:
		return schemapb.DataType_None, fmt.Errorf("unsupported arrow type: %s", t.Name())
	}
}

// DataFrame is a pure data container that stores Milvus data using Arrow Chunked Arrays.
// Each chunk corresponds to a query result (NQ), enabling per-query access.
type DataFrame struct {
	schema     *arrow.Schema                // Arrow schema with field metadata
	columns    []*arrow.Chunked             // Chunked arrays, one per column
	chunkSizes []int64                      // Row count per chunk (corresponds to Topks), import/insert data, chunkSizes always equals to 1, search result, chunkSizes equals to Topks
	nameIndex  map[string]int               // Column name to index mapping
	fieldTypes map[string]schemapb.DataType // Preserve Milvus type info for export
	fieldIDs   map[string]int64             // Field IDs for export
}

// NewDataFrame creates an empty DataFrame.
func NewDataFrame() *DataFrame {
	return &DataFrame{
		schema:     nil,
		columns:    make([]*arrow.Chunked, 0),
		chunkSizes: make([]int64, 0),
		nameIndex:  make(map[string]int),
		fieldTypes: make(map[string]schemapb.DataType),
		fieldIDs:   make(map[string]int64),
	}
}

// =============================================================================
// Metadata Methods
// =============================================================================

// NumRows returns the total number of rows across all chunks.
func (df *DataFrame) NumRows() int64 {
	var total int64
	for _, size := range df.chunkSizes {
		total += size
	}
	return total
}

// NumChunks returns the number of chunks (NQ for search results).
func (df *DataFrame) NumChunks() int {
	return len(df.chunkSizes)
}

// GetColumns returns columns for a specific chunk by column names.
func (df *DataFrame) GetColumns(colNames []string, chunkIdx int) ([]arrow.Array, error) {
	if chunkIdx < 0 || chunkIdx >= len(df.chunkSizes) {
		return nil, fmt.Errorf("chunk index out of range: %d", chunkIdx)
	}
	columns := make([]arrow.Array, 0, len(colNames))
	for _, colName := range colNames {
		colIdx, exists := df.nameIndex[colName]
		if !exists {
			return nil, fmt.Errorf("column %s not found", colName)
		}
		columns = append(columns, df.columns[colIdx].Chunk(chunkIdx))
	}
	return columns, nil
}

// ChunkSizes returns the row count per chunk (same as Topks for search results).
func (df *DataFrame) ChunkSizes() []int64 {
	result := make([]int64, len(df.chunkSizes))
	copy(result, df.chunkSizes)
	return result
}

// Schema returns the Arrow schema.
func (df *DataFrame) Schema() *arrow.Schema {
	return df.schema
}

// =============================================================================
// Column Access Methods
// =============================================================================

// Column returns a column by name.
func (df *DataFrame) Column(name string) (*arrow.Chunked, error) {
	idx, exists := df.nameIndex[name]
	if !exists {
		return nil, fmt.Errorf("column %s not found", name)
	}
	return df.columns[idx], nil
}

// ColumnNames returns all column names in schema insertion order.
func (df *DataFrame) ColumnNames() []string {
	if df.schema == nil {
		return nil
	}
	fields := df.schema.Fields()
	names := make([]string, len(fields))
	for i, f := range fields {
		names[i] = f.Name
	}
	return names
}

// HasColumn checks if a column exists.
func (df *DataFrame) HasColumn(name string) bool {
	_, exists := df.nameIndex[name]
	return exists
}

// =============================================================================
// Column Operations
// =============================================================================

// AddColumn adds a new column to the DataFrame and returns a new DataFrame.
// IMPORTANT: This function takes ownership of the chunks array elements.
// After calling AddColumn, the caller should NOT use or release the chunks,
// as they will be released internally after being added to the new DataFrame.
// If you need to keep using the chunks, call Retain() on them before passing.
func (df *DataFrame) AddColumn(name string, chunks []arrow.Array, dataType arrow.DataType) (*DataFrame, error) {
	if _, exists := df.nameIndex[name]; exists {
		return nil, fmt.Errorf("column %s already exists", name)
	}

	if len(chunks) != len(df.chunkSizes) && len(df.chunkSizes) > 0 {
		return nil, fmt.Errorf("chunk count mismatch: expected %d, got %d", len(df.chunkSizes), len(chunks))
	}

	// Validate chunk sizes
	for i, chunk := range chunks {
		if int64(chunk.Len()) != df.chunkSizes[i] {
			return nil, fmt.Errorf("chunk %d size mismatch: expected %d, got %d", i, df.chunkSizes[i], chunk.Len())
		}
	}

	// Create new DataFrame
	newDF := &DataFrame{
		chunkSizes: make([]int64, len(df.chunkSizes)),
		nameIndex:  make(map[string]int),
		fieldTypes: make(map[string]schemapb.DataType),
		fieldIDs:   make(map[string]int64),
	}
	copy(newDF.chunkSizes, df.chunkSizes)

	// Copy existing columns
	newDF.columns = make([]*arrow.Chunked, len(df.columns)+1)
	for i, col := range df.columns {
		col.Retain()
		newDF.columns[i] = col
	}

	// Add new column
	chunked := arrow.NewChunked(dataType, chunks)
	newDF.columns[len(df.columns)] = chunked

	// Release individual arrays after creating chunked
	for _, chunk := range chunks {
		chunk.Release()
	}

	// Update name index
	maps.Copy(newDF.nameIndex, df.nameIndex)
	newDF.nameIndex[name] = len(df.columns)

	// Copy field types and IDs
	maps.Copy(newDF.fieldTypes, df.fieldTypes)
	maps.Copy(newDF.fieldIDs, df.fieldIDs)

	// Infer Milvus type from Arrow type
	milvusType, err := ToMilvusType(dataType)
	if err == nil {
		newDF.fieldTypes[name] = milvusType
	}

	// Update schema
	fields := make([]arrow.Field, 0, len(df.columns)+1)
	if df.schema != nil {
		fields = append(fields, df.schema.Fields()...)
	}
	fields = append(fields, arrow.Field{Name: name, Type: dataType, Nullable: true})
	newDF.schema = arrow.NewSchema(fields, nil)

	return newDF, nil
}

// RemoveColumn removes a column from the DataFrame and returns a new DataFrame.
func (df *DataFrame) RemoveColumn(name string) (*DataFrame, error) {
	idx, exists := df.nameIndex[name]
	if !exists {
		return nil, fmt.Errorf("column %s not found", name)
	}

	// Create new DataFrame
	newDF := &DataFrame{
		chunkSizes: make([]int64, len(df.chunkSizes)),
		nameIndex:  make(map[string]int),
		fieldTypes: make(map[string]schemapb.DataType),
		fieldIDs:   make(map[string]int64),
	}
	copy(newDF.chunkSizes, df.chunkSizes)

	// Copy columns except the removed one
	newDF.columns = make([]*arrow.Chunked, 0, len(df.columns)-1)
	for i, col := range df.columns {
		if i != idx {
			col.Retain()
			newDF.columns = append(newDF.columns, col)
		}
	}

	// Update schema
	fields := make([]arrow.Field, 0, len(df.columns)-1)
	for i, f := range df.schema.Fields() {
		if i != idx {
			fields = append(fields, f)
		}
	}
	newDF.schema = arrow.NewSchema(fields, nil)

	// Update name index
	for n, i := range df.nameIndex {
		if n == name {
			continue
		}
		if i > idx {
			newDF.nameIndex[n] = i - 1
		} else {
			newDF.nameIndex[n] = i
		}
	}

	// Copy field types and IDs except the removed one
	for n, t := range df.fieldTypes {
		if n != name {
			newDF.fieldTypes[n] = t
		}
	}
	for n, id := range df.fieldIDs {
		if n != name {
			newDF.fieldIDs[n] = id
		}
	}

	return newDF, nil
}

// =============================================================================
// Lifecycle Methods
// =============================================================================

// Release releases all Arrow resources held by the DataFrame.
func (df *DataFrame) Release() {
	for _, col := range df.columns {
		if col != nil {
			col.Release()
		}
	}
	df.columns = nil
	df.schema = nil
}

// Retain increments the reference count for all columns.
func (df *DataFrame) Retain() {
	for _, col := range df.columns {
		if col != nil {
			col.Retain()
		}
	}
}

// Clone creates a shallow copy of the DataFrame.
// The new DataFrame shares the same underlying Arrow arrays (with incremented reference counts).
// Both the original and cloned DataFrame must be released independently.
func (df *DataFrame) Clone() *DataFrame {
	newDF := &DataFrame{
		chunkSizes: make([]int64, len(df.chunkSizes)),
		columns:    make([]*arrow.Chunked, len(df.columns)),
		nameIndex:  make(map[string]int),
		fieldTypes: make(map[string]schemapb.DataType),
		fieldIDs:   make(map[string]int64),
	}

	// Copy chunk sizes
	copy(newDF.chunkSizes, df.chunkSizes)

	// Copy schema
	newDF.schema = df.schema

	// Copy columns with Retain
	for i, col := range df.columns {
		if col != nil {
			col.Retain()
			newDF.columns[i] = col
		}
	}

	// Copy maps
	maps.Copy(newDF.nameIndex, df.nameIndex)
	maps.Copy(newDF.fieldTypes, df.fieldTypes)
	maps.Copy(newDF.fieldIDs, df.fieldIDs)

	return newDF
}

// =============================================================================
// Import Functions
// =============================================================================

// FromSearchResultData creates a DataFrame from SearchResultData.
// Each query's results become a separate chunk.
func FromSearchResultData(resultData *schemapb.SearchResultData) (*DataFrame, error) {
	if resultData == nil {
		return nil, fmt.Errorf("resultData is nil")
	}

	df := NewDataFrame()

	topks := resultData.GetTopks()
	if len(topks) == 0 {
		return df, nil
	}

	// Set chunk sizes (one chunk per query)
	df.chunkSizes = make([]int64, len(topks))
	copy(df.chunkSizes, topks)

	// Calculate offsets for data splitting
	offsets := make([]int64, len(topks)+1)
	for i, topk := range topks {
		offsets[i+1] = offsets[i] + topk
	}

	// Import ID column ($id)
	if ids := resultData.GetIds(); ids != nil {
		if err := df.importIDs(ids, offsets); err != nil {
			return nil, err
		}
	}

	// Import Score column ($score)
	if scores := resultData.GetScores(); len(scores) > 0 {
		if err := df.importScores(scores, offsets); err != nil {
			return nil, err
		}
	}

	// Import other fields
	for _, fieldData := range resultData.GetFieldsData() {
		if err := df.importFieldData(fieldData, offsets); err != nil {
			return nil, err
		}
	}

	return df, nil
}

// importIDs imports IDs into the DataFrame.
func (df *DataFrame) importIDs(ids *schemapb.IDs, offsets []int64) error {
	numChunks := len(offsets) - 1
	chunks := make([]arrow.Array, numChunks)

	switch ids.IdField.(type) {
	case *schemapb.IDs_IntId:
		data := ids.GetIntId().GetData()
		for i := range numChunks {
			builder := array.NewInt64Builder(memory.DefaultAllocator)
			if offsets[i+1] > offsets[i] {
				builder.AppendValues(data[offsets[i]:offsets[i+1]], nil)
			}
			chunks[i] = builder.NewArray()
			builder.Release()
		}
		df.fieldTypes[IDFieldName] = schemapb.DataType_Int64

	case *schemapb.IDs_StrId:
		data := ids.GetStrId().GetData()
		for i := range numChunks {
			builder := array.NewStringBuilder(memory.DefaultAllocator)
			if offsets[i+1] > offsets[i] {
				builder.AppendValues(data[offsets[i]:offsets[i+1]], nil)
			}
			chunks[i] = builder.NewArray()
			builder.Release()
		}
		df.fieldTypes[IDFieldName] = schemapb.DataType_VarChar

	default:
		return fmt.Errorf("unsupported ID type")
	}

	return df.addChunkedColumn(IDFieldName, chunks)
}

// importScores imports scores into the DataFrame.
func (df *DataFrame) importScores(scores []float32, offsets []int64) error {
	numChunks := len(offsets) - 1
	chunks := make([]arrow.Array, numChunks)

	for i := range numChunks {
		builder := array.NewFloat32Builder(memory.DefaultAllocator)
		if offsets[i+1] > offsets[i] {
			builder.AppendValues(scores[offsets[i]:offsets[i+1]], nil)
		}
		chunks[i] = builder.NewArray()
		builder.Release()
	}

	df.fieldTypes[ScoreFieldName] = schemapb.DataType_Float
	return df.addChunkedColumn(ScoreFieldName, chunks)
}

// importFieldData imports a FieldData into the DataFrame.
func (df *DataFrame) importFieldData(fieldData *schemapb.FieldData, offsets []int64) error {
	numChunks := len(offsets) - 1
	chunks := make([]arrow.Array, numChunks)
	fieldName := fieldData.GetFieldName()

	switch fieldData.GetType() {
	case schemapb.DataType_Bool:
		data := fieldData.GetScalars().GetBoolData().GetData()
		for i := range numChunks {
			builder := array.NewBooleanBuilder(memory.DefaultAllocator)
			if offsets[i+1] > offsets[i] {
				builder.AppendValues(data[offsets[i]:offsets[i+1]], nil)
			}
			chunks[i] = builder.NewArray()
			builder.Release()
		}

	case schemapb.DataType_Int8:
		data := fieldData.GetScalars().GetIntData().GetData()
		for i := range numChunks {
			builder := array.NewInt8Builder(memory.DefaultAllocator)
			for j := offsets[i]; j < offsets[i+1]; j++ {
				builder.Append(int8(data[j]))
			}
			chunks[i] = builder.NewArray()
			builder.Release()
		}

	case schemapb.DataType_Int16:
		data := fieldData.GetScalars().GetIntData().GetData()
		for i := range numChunks {
			builder := array.NewInt16Builder(memory.DefaultAllocator)
			for j := offsets[i]; j < offsets[i+1]; j++ {
				builder.Append(int16(data[j]))
			}
			chunks[i] = builder.NewArray()
			builder.Release()
		}

	case schemapb.DataType_Int32:
		data := fieldData.GetScalars().GetIntData().GetData()
		for i := range numChunks {
			builder := array.NewInt32Builder(memory.DefaultAllocator)
			if offsets[i+1] > offsets[i] {
				builder.AppendValues(data[offsets[i]:offsets[i+1]], nil)
			}
			chunks[i] = builder.NewArray()
			builder.Release()
		}

	case schemapb.DataType_Int64:
		data := fieldData.GetScalars().GetLongData().GetData()
		for i := range numChunks {
			builder := array.NewInt64Builder(memory.DefaultAllocator)
			if offsets[i+1] > offsets[i] {
				builder.AppendValues(data[offsets[i]:offsets[i+1]], nil)
			}
			chunks[i] = builder.NewArray()
			builder.Release()
		}

	case schemapb.DataType_Float:
		data := fieldData.GetScalars().GetFloatData().GetData()
		for i := range numChunks {
			builder := array.NewFloat32Builder(memory.DefaultAllocator)
			if offsets[i+1] > offsets[i] {
				builder.AppendValues(data[offsets[i]:offsets[i+1]], nil)
			}
			chunks[i] = builder.NewArray()
			builder.Release()
		}

	case schemapb.DataType_Double:
		data := fieldData.GetScalars().GetDoubleData().GetData()
		for i := range numChunks {
			builder := array.NewFloat64Builder(memory.DefaultAllocator)
			if offsets[i+1] > offsets[i] {
				builder.AppendValues(data[offsets[i]:offsets[i+1]], nil)
			}
			chunks[i] = builder.NewArray()
			builder.Release()
		}

	case schemapb.DataType_String, schemapb.DataType_VarChar, schemapb.DataType_Text:
		data := fieldData.GetScalars().GetStringData().GetData()
		for i := range numChunks {
			builder := array.NewStringBuilder(memory.DefaultAllocator)
			if offsets[i+1] > offsets[i] {
				builder.AppendValues(data[offsets[i]:offsets[i+1]], nil)
			}
			chunks[i] = builder.NewArray()
			builder.Release()
		}

	default:
		return fmt.Errorf("unsupported field data type: %s", fieldData.GetType().String())
	}

	df.fieldTypes[fieldName] = fieldData.GetType()
	df.fieldIDs[fieldName] = fieldData.GetFieldId()
	return df.addChunkedColumn(fieldName, chunks)
}

// addChunkedColumn is an internal helper to add a chunked column.
func (df *DataFrame) addChunkedColumn(name string, chunks []arrow.Array) error {
	if len(chunks) == 0 {
		return nil
	}

	arrowType := chunks[0].DataType()
	chunked := arrow.NewChunked(arrowType, chunks)

	// Update schema
	fields := make([]arrow.Field, 0)
	if df.schema != nil {
		fields = append(fields, df.schema.Fields()...)
	}
	fields = append(fields, arrow.Field{Name: name, Type: arrowType, Nullable: true})
	df.schema = arrow.NewSchema(fields, nil)

	// Add column
	df.columns = append(df.columns, chunked)
	df.nameIndex[name] = len(df.columns) - 1

	// Release individual arrays after creating chunked
	for _, chunk := range chunks {
		chunk.Release()
	}

	return nil
}

// addChunkedColumnDirect adds a ChunkedArray column directly (internal use).
// This method does not create a new DataFrame, it modifies the current DataFrame directly.
// It should only be used when building a new DataFrame.
func (df *DataFrame) addChunkedColumnDirect(name string, col *arrow.Chunked) error {
	if col == nil {
		return fmt.Errorf("column %s is nil", name)
	}

	// Update schema
	fields := make([]arrow.Field, 0)
	if df.schema != nil {
		fields = append(fields, df.schema.Fields()...)
	}
	fields = append(fields, arrow.Field{Name: name, Type: col.DataType(), Nullable: true})
	df.schema = arrow.NewSchema(fields, nil)

	// Add column
	df.columns = append(df.columns, col)
	df.nameIndex[name] = len(df.columns) - 1

	return nil
}

// =============================================================================
// Export Functions
// =============================================================================

// ToSearchResultData exports the DataFrame to SearchResultData.
func (df *DataFrame) ToSearchResultData() (*schemapb.SearchResultData, error) {
	result := &schemapb.SearchResultData{
		NumQueries: int64(df.NumChunks()),
		TopK:       df.maxChunkSize(),
		Topks:      df.ChunkSizes(),
		FieldsData: make([]*schemapb.FieldData, 0),
		Scores:     []float32{},
		Ids:        &schemapb.IDs{},
	}

	// Export ID
	if df.HasColumn(IDFieldName) {
		ids, err := df.exportIDs()
		if err != nil {
			return nil, err
		}
		result.Ids = ids
	}

	// Export Score
	if df.HasColumn(ScoreFieldName) {
		scores, err := df.exportScores()
		if err != nil {
			return nil, err
		}
		result.Scores = scores
	}

	// Export other fields
	for _, name := range df.ColumnNames() {
		if name == IDFieldName || name == ScoreFieldName {
			continue
		}
		// Skip temporary columns (starting with _)
		if strings.HasPrefix(name, "_") {
			continue
		}

		fieldData, err := df.exportFieldData(name)
		if err != nil {
			return nil, err
		}
		result.FieldsData = append(result.FieldsData, fieldData)
	}

	return result, nil
}

// exportIDs exports IDs from the DataFrame.
func (df *DataFrame) exportIDs() (*schemapb.IDs, error) {
	col, _ := df.Column(IDFieldName)
	dataType := df.fieldTypes[IDFieldName]

	switch dataType {
	case schemapb.DataType_Int64:
		data := make([]int64, 0, col.Len())
		for i := 0; i < len(col.Chunks()); i++ {
			chunk, ok := col.Chunk(i).(*array.Int64)
			if !ok {
				return nil, fmt.Errorf("exportIDs: chunk %d is not Int64 array", i)
			}
			for j := 0; j < chunk.Len(); j++ {
				data = append(data, chunk.Value(j))
			}
		}
		return &schemapb.IDs{
			IdField: &schemapb.IDs_IntId{
				IntId: &schemapb.LongArray{Data: data},
			},
		}, nil

	case schemapb.DataType_VarChar, schemapb.DataType_String:
		data := make([]string, 0, col.Len())
		for i := 0; i < len(col.Chunks()); i++ {
			chunk, ok := col.Chunk(i).(*array.String)
			if !ok {
				return nil, fmt.Errorf("exportIDs: chunk %d is not String array", i)
			}
			for j := 0; j < chunk.Len(); j++ {
				data = append(data, chunk.Value(j))
			}
		}
		return &schemapb.IDs{
			IdField: &schemapb.IDs_StrId{
				StrId: &schemapb.StringArray{Data: data},
			},
		}, nil
	}

	return nil, fmt.Errorf("unsupported ID type: %v", dataType)
}

// exportScores exports scores from the DataFrame.
func (df *DataFrame) exportScores() ([]float32, error) {
	col, _ := df.Column(ScoreFieldName)
	data := make([]float32, 0, col.Len())

	for i := 0; i < len(col.Chunks()); i++ {
		chunk, ok := col.Chunk(i).(*array.Float32)
		if !ok {
			return nil, fmt.Errorf("exportScores: chunk %d is not Float32 array", i)
		}
		for j := 0; j < chunk.Len(); j++ {
			data = append(data, chunk.Value(j))
		}
	}
	return data, nil
}

// exportFieldData exports a field from the DataFrame.
func (df *DataFrame) exportFieldData(name string) (*schemapb.FieldData, error) {
	col, _ := df.Column(name)
	dataType := df.fieldTypes[name]
	fieldID := df.fieldIDs[name]

	fieldData := &schemapb.FieldData{
		Type:      dataType,
		FieldName: name,
		FieldId:   fieldID,
	}

	switch dataType {
	case schemapb.DataType_Bool:
		data := make([]bool, 0, col.Len())
		for i := 0; i < len(col.Chunks()); i++ {
			chunk := col.Chunk(i).(*array.Boolean)
			for j := 0; j < chunk.Len(); j++ {
				data = append(data, chunk.Value(j))
			}
		}
		fieldData.Field = &schemapb.FieldData_Scalars{
			Scalars: &schemapb.ScalarField{
				Data: &schemapb.ScalarField_BoolData{
					BoolData: &schemapb.BoolArray{Data: data},
				},
			},
		}

	case schemapb.DataType_Int8, schemapb.DataType_Int16, schemapb.DataType_Int32:
		data := make([]int32, 0, col.Len())
		for i := 0; i < len(col.Chunks()); i++ {
			chunk := col.Chunk(i)
			for j := 0; j < chunk.Len(); j++ {
				val := getArrayValue(chunk, j)
				switch v := val.(type) {
				case int8:
					data = append(data, int32(v))
				case int16:
					data = append(data, int32(v))
				case int32:
					data = append(data, v)
				}
			}
		}
		fieldData.Field = &schemapb.FieldData_Scalars{
			Scalars: &schemapb.ScalarField{
				Data: &schemapb.ScalarField_IntData{
					IntData: &schemapb.IntArray{Data: data},
				},
			},
		}

	case schemapb.DataType_Int64:
		data := make([]int64, 0, col.Len())
		for i := 0; i < len(col.Chunks()); i++ {
			chunk := col.Chunk(i).(*array.Int64)
			for j := 0; j < chunk.Len(); j++ {
				data = append(data, chunk.Value(j))
			}
		}
		fieldData.Field = &schemapb.FieldData_Scalars{
			Scalars: &schemapb.ScalarField{
				Data: &schemapb.ScalarField_LongData{
					LongData: &schemapb.LongArray{Data: data},
				},
			},
		}

	case schemapb.DataType_Float:
		data := make([]float32, 0, col.Len())
		for i := 0; i < len(col.Chunks()); i++ {
			chunk := col.Chunk(i).(*array.Float32)
			for j := 0; j < chunk.Len(); j++ {
				data = append(data, chunk.Value(j))
			}
		}
		fieldData.Field = &schemapb.FieldData_Scalars{
			Scalars: &schemapb.ScalarField{
				Data: &schemapb.ScalarField_FloatData{
					FloatData: &schemapb.FloatArray{Data: data},
				},
			},
		}

	case schemapb.DataType_Double:
		data := make([]float64, 0, col.Len())
		for i := 0; i < len(col.Chunks()); i++ {
			chunk := col.Chunk(i).(*array.Float64)
			for j := 0; j < chunk.Len(); j++ {
				data = append(data, chunk.Value(j))
			}
		}
		fieldData.Field = &schemapb.FieldData_Scalars{
			Scalars: &schemapb.ScalarField{
				Data: &schemapb.ScalarField_DoubleData{
					DoubleData: &schemapb.DoubleArray{Data: data},
				},
			},
		}

	case schemapb.DataType_String, schemapb.DataType_VarChar, schemapb.DataType_Text:
		data := make([]string, 0, col.Len())
		for i := 0; i < len(col.Chunks()); i++ {
			chunk := col.Chunk(i).(*array.String)
			for j := 0; j < chunk.Len(); j++ {
				data = append(data, chunk.Value(j))
			}
		}
		fieldData.Field = &schemapb.FieldData_Scalars{
			Scalars: &schemapb.ScalarField{
				Data: &schemapb.ScalarField_StringData{
					StringData: &schemapb.StringArray{Data: data},
				},
			},
		}

	default:
		return nil, fmt.Errorf("unsupported field type: %v", dataType)
	}

	return fieldData, nil
}

// =============================================================================
// Helper Functions
// =============================================================================

// maxChunkSize returns the maximum chunk size.
func (df *DataFrame) maxChunkSize() int64 {
	var max int64
	for _, size := range df.chunkSizes {
		if size > max {
			max = size
		}
	}
	return max
}

// getArrayValue gets a value from an arrow array at the specified index.
func getArrayValue(arr arrow.Array, idx int) any {
	if arr.IsNull(idx) {
		return nil
	}
	switch a := arr.(type) {
	case *array.Boolean:
		return a.Value(idx)
	case *array.Int8:
		return a.Value(idx)
	case *array.Int16:
		return a.Value(idx)
	case *array.Int32:
		return a.Value(idx)
	case *array.Int64:
		return a.Value(idx)
	case *array.Float32:
		return a.Value(idx)
	case *array.Float64:
		return a.Value(idx)
	case *array.String:
		return a.Value(idx)
	default:
		return nil
	}
}

// SetFieldType sets the Milvus data type for a column.
func (df *DataFrame) SetFieldType(name string, dataType schemapb.DataType) {
	df.fieldTypes[name] = dataType
}

// SetFieldID sets the field ID for a column.
func (df *DataFrame) SetFieldID(name string, fieldID int64) {
	df.fieldIDs[name] = fieldID
}

// GetFieldType returns the Milvus DataType for a column.
func (df *DataFrame) GetFieldType(name string) (schemapb.DataType, bool) {
	dt, exists := df.fieldTypes[name]
	return dt, exists
}

// GetFieldID returns the field ID for a column.
func (df *DataFrame) GetFieldID(name string) (int64, bool) {
	id, exists := df.fieldIDs[name]
	return id, exists
}

// NumColumns returns the number of columns.
func (df *DataFrame) NumColumns() int {
	return len(df.columns)
}
