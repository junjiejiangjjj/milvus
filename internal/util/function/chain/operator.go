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
	"cmp"
	"fmt"
	"sort"

	"github.com/apache/arrow/go/v17/arrow"
	"github.com/apache/arrow/go/v17/arrow/array"
	"github.com/apache/arrow/go/v17/arrow/memory"
)

// =============================================================================
// Generic Array Helpers
// =============================================================================

// typedArray is an Arrow array that provides typed value access.
type typedArray[T any] interface {
	Len() int
	IsNull(int) bool
	Value(int) T
}

// typedBuilder is an Arrow builder that supports typed append.
type typedBuilder[T any] interface {
	Append(T)
	AppendNull()
	NewArray() arrow.Array
	Release()
}

// pickByIndices creates a new array by picking elements at the given indices.
func pickByIndices[T any, A typedArray[T], B typedBuilder[T]](arr A, builder B, indices []int) arrow.Array {
	defer builder.Release()
	for _, idx := range indices {
		if arr.IsNull(idx) {
			builder.AppendNull()
		} else {
			builder.Append(arr.Value(idx))
		}
	}
	return builder.NewArray()
}

// compareTyped compares two values in a typed array using cmp.Ordered.
func compareTyped[T cmp.Ordered, A typedArray[T]](arr A, i, j int) int {
	return cmp.Compare(arr.Value(i), arr.Value(j))
}

// =============================================================================
// Type Dispatch Functions
// =============================================================================

// dispatchPickByIndices dispatches pickByIndices to the correct Arrow type.
func dispatchPickByIndices(pool memory.Allocator, data arrow.Array, indices []int) (arrow.Array, error) {
	switch arr := data.(type) {
	case *array.Boolean:
		return pickByIndices(arr, array.NewBooleanBuilder(pool), indices), nil
	case *array.Int8:
		return pickByIndices(arr, array.NewInt8Builder(pool), indices), nil
	case *array.Int16:
		return pickByIndices(arr, array.NewInt16Builder(pool), indices), nil
	case *array.Int32:
		return pickByIndices(arr, array.NewInt32Builder(pool), indices), nil
	case *array.Int64:
		return pickByIndices(arr, array.NewInt64Builder(pool), indices), nil
	case *array.Float32:
		return pickByIndices(arr, array.NewFloat32Builder(pool), indices), nil
	case *array.Float64:
		return pickByIndices(arr, array.NewFloat64Builder(pool), indices), nil
	case *array.String:
		return pickByIndices(arr, array.NewStringBuilder(pool), indices), nil
	default:
		return nil, fmt.Errorf("unsupported array type %T", data)
	}
}

// =============================================================================
// Operators
// =============================================================================

// BaseOp is the base operator with common fields.
type BaseOp struct {
	inputs  []string
	outputs []string
}

func (o *BaseOp) Inputs() []string  { return o.inputs }
func (o *BaseOp) Outputs() []string { return o.outputs }

// -----------------------------------------------------------------------------
// MapOp
// -----------------------------------------------------------------------------

// MapOp applies a function to specified columns of the DataFrame.
// Column mapping is handled at the Operator layer, not the Function layer.
type MapOp struct {
	BaseOp
	function   FunctionExpr // The function to apply
	inputCols  []string     // Input column names to read from DataFrame
	outputCols []string     // Output column names to write to DataFrame
}

// NewMapOp creates a new MapOp with explicit column mappings.
func NewMapOp(function FunctionExpr, inputCols, outputCols []string) (*MapOp, error) {
	if function == nil {
		return nil, fmt.Errorf("MapOp: function is nil")
	}

	// Validate: outputCols length must match function.OutputDataTypes() length
	outputTypes := function.OutputDataTypes()
	if len(outputCols) != len(outputTypes) {
		return nil, fmt.Errorf("MapOp: output columns count (%d) must match function output types count (%d)",
			len(outputCols), len(outputTypes))
	}

	return &MapOp{
		function:   function,
		inputCols:  inputCols,
		outputCols: outputCols,
	}, nil
}

func (o *MapOp) Name() string { return "Map" }

// Inputs returns the input column names.
func (o *MapOp) Inputs() []string { return o.inputCols }

// Outputs returns the output column names.
func (o *MapOp) Outputs() []string { return o.outputCols }

func (o *MapOp) Execute(ctx *FuncContext, input *DataFrame) (*DataFrame, error) {
	if o.function == nil {
		return nil, fmt.Errorf("MapOp: function is nil")
	}

	// 1. Read input columns from DataFrame using inputCols
	inputs := make([]*arrow.Chunked, len(o.inputCols))
	for i, name := range o.inputCols {
		col, err := input.Column(name)
		if err != nil {
			return nil, fmt.Errorf("MapOp: %w", err)
		}
		inputs[i] = col
	}

	// 2. Call FunctionExpr to process columns
	outputs, err := o.function.Execute(ctx, inputs)
	if err != nil {
		return nil, err
	}

	// 3. Create new output DataFrame
	result := NewDataFrame()
	result.chunkSizes = make([]int64, len(input.chunkSizes))
	copy(result.chunkSizes, input.chunkSizes)

	// Build set of output column names (these will replace any existing columns with the same name)
	outputColSet := make(map[string]struct{})
	for _, name := range o.outputCols {
		outputColSet[name] = struct{}{}
	}

	// 4. Copy input columns that are not in output columns
	for _, colName := range input.ColumnNames() {
		if _, isOutput := outputColSet[colName]; isOutput {
			continue // Skip, will be replaced by output
		}
		col, _ := input.Column(colName)
		col.Retain()
		if err := result.addChunkedColumnDirect(colName, col); err != nil {
			return nil, err
		}
		// Copy field type and ID
		if ft, ok := input.GetFieldType(colName); ok {
			result.SetFieldType(colName, ft)
		}
		if fid, ok := input.GetFieldID(colName); ok {
			result.SetFieldID(colName, fid)
		}
	}

	// 5. Add output columns with names from outputCols
	for i, colName := range o.outputCols {
		if err := result.addChunkedColumnDirect(colName, outputs[i]); err != nil {
			return nil, fmt.Errorf("MapOp: failed to add column %s: %w", colName, err)
		}
	}

	return result, nil
}

func (o *MapOp) String() string {
	if o.function != nil {
		return fmt.Sprintf("Map(%s)", o.function.Name())
	}
	return "Map(nil)"
}

// -----------------------------------------------------------------------------
// FilterOp
// -----------------------------------------------------------------------------

// FilterOp filters the DataFrame based on a boolean column.
type FilterOp struct {
	BaseOp
	column string
}

func (o *FilterOp) Name() string      { return "Filter" }
func (o *FilterOp) Inputs() []string  { return []string{o.column} }
func (o *FilterOp) Outputs() []string { return []string{} } // Filter doesn't produce new columns

func (o *FilterOp) Execute(ctx *FuncContext, input *DataFrame) (*DataFrame, error) {
	// Get the filter column
	filterCol, err := input.Column(o.column)
	if err != nil {
		return nil, fmt.Errorf("FilterOp: %w", err)
	}

	// Validate filter column type
	if filterCol.DataType().ID() != arrow.BOOL {
		return nil, fmt.Errorf("FilterOp: column %s must be boolean type, got %s", o.column, filterCol.DataType().Name())
	}

	// Create new DataFrame
	newDF := NewDataFrame()
	newChunkSizes := make([]int64, 0, input.NumChunks())

	// Process each chunk - extract boolean chunks with type safety
	filterChunks := make([]*array.Boolean, input.NumChunks())
	for chunkIdx := range input.NumChunks() {
		boolChunk, ok := filterCol.Chunk(chunkIdx).(*array.Boolean)
		if !ok {
			return nil, fmt.Errorf("FilterOp: chunk %d is not a boolean array", chunkIdx)
		}
		filterChunks[chunkIdx] = boolChunk

		// Count true values
		trueCount := int64(0)
		for i := range boolChunk.Len() {
			if boolChunk.IsValid(i) && boolChunk.Value(i) {
				trueCount++
			}
		}
		newChunkSizes = append(newChunkSizes, trueCount)
	}

	newDF.chunkSizes = newChunkSizes

	// Filter each column
	for _, colName := range input.ColumnNames() {
		col, _ := input.Column(colName)
		newChunks := make([]arrow.Array, input.NumChunks())

		for chunkIdx := range input.NumChunks() {
			dataChunk := col.Chunk(chunkIdx)

			filtered, err := filterArray(ctx.Pool(), dataChunk, filterChunks[chunkIdx])
			if err != nil {
				// Release already created chunks on error
				for i := 0; i < chunkIdx; i++ {
					if newChunks[i] != nil {
						newChunks[i].Release()
					}
				}
				return nil, fmt.Errorf("FilterOp: column %s: %w", colName, err)
			}
			newChunks[chunkIdx] = filtered
		}

		if err := newDF.addChunkedColumn(colName, newChunks); err != nil {
			return nil, err
		}

		// Copy field type and ID
		if ft, ok := input.GetFieldType(colName); ok {
			newDF.SetFieldType(colName, ft)
		}
		if fid, ok := input.GetFieldID(colName); ok {
			newDF.SetFieldID(colName, fid)
		}
	}

	return newDF, nil
}

func (o *FilterOp) String() string {
	return fmt.Sprintf("Filter(%s)", o.column)
}

// filterArray filters an array based on a boolean mask.
func filterArray(pool memory.Allocator, data arrow.Array, mask *array.Boolean) (arrow.Array, error) {
	indices := make([]int, 0, mask.Len())
	for i := range mask.Len() {
		if mask.IsValid(i) && mask.Value(i) {
			indices = append(indices, i)
		}
	}
	return dispatchPickByIndices(pool, data, indices)
}

// -----------------------------------------------------------------------------
// SelectOp
// -----------------------------------------------------------------------------

// SelectOp selects specific columns from the DataFrame.
type SelectOp struct {
	BaseOp
	columns []string
}

func (o *SelectOp) Name() string      { return "Select" }
func (o *SelectOp) Inputs() []string  { return o.columns }
func (o *SelectOp) Outputs() []string { return o.columns }

func (o *SelectOp) Execute(ctx *FuncContext, input *DataFrame) (*DataFrame, error) {
	newDF := NewDataFrame()
	newDF.chunkSizes = make([]int64, len(input.chunkSizes))
	copy(newDF.chunkSizes, input.chunkSizes)

	for _, colName := range o.columns {
		col, err := input.Column(colName)
		if err != nil {
			return nil, fmt.Errorf("SelectOp: %w", err)
		}

		// Copy chunks
		chunks := make([]arrow.Array, len(col.Chunks()))
		for i, chunk := range col.Chunks() {
			chunk.Retain()
			chunks[i] = chunk
		}

		if err := newDF.addChunkedColumn(colName, chunks); err != nil {
			// Release retained chunks on error
			for _, chunk := range chunks {
				chunk.Release()
			}
			return nil, err
		}

		// Copy field type and ID
		if ft, ok := input.GetFieldType(colName); ok {
			newDF.SetFieldType(colName, ft)
		}
		if fid, ok := input.GetFieldID(colName); ok {
			newDF.SetFieldID(colName, fid)
		}
	}

	return newDF, nil
}

func (o *SelectOp) String() string {
	return fmt.Sprintf("Select(%v)", o.columns)
}

// -----------------------------------------------------------------------------
// SortOp
// -----------------------------------------------------------------------------

// SortOp sorts the DataFrame by a column.
type SortOp struct {
	BaseOp
	column string
	desc   bool
}

func (o *SortOp) Name() string      { return "Sort" }
func (o *SortOp) Inputs() []string  { return []string{o.column} }
func (o *SortOp) Outputs() []string { return []string{} } // Sort doesn't produce new columns

func (o *SortOp) Execute(ctx *FuncContext, input *DataFrame) (*DataFrame, error) {
	sortCol, err := input.Column(o.column)
	if err != nil {
		return nil, fmt.Errorf("SortOp: %w", err)
	}

	// Validate sort column type is comparable
	if !isComparableType(sortCol.DataType()) {
		return nil, fmt.Errorf("SortOp: column %s has non-comparable type %s", o.column, sortCol.DataType().Name())
	}

	// Collect all column chunks first
	colNames := input.ColumnNames()
	allChunks := make(map[string][]arrow.Array)
	for _, name := range colNames {
		allChunks[name] = make([]arrow.Array, input.NumChunks())
	}

	newChunkSizes := make([]int64, input.NumChunks())

	// releaseAllChunks is a helper to clean up on error
	releaseAllChunks := func() {
		for _, chunks := range allChunks {
			for _, chunk := range chunks {
				if chunk != nil {
					chunk.Release()
				}
			}
		}
	}

	// Process each chunk independently
	for chunkIdx := range input.NumChunks() {
		sortChunk := sortCol.Chunk(chunkIdx)
		chunkLen := sortChunk.Len()

		// Build sort indices
		indices := make([]int, chunkLen)
		for i := range chunkLen {
			indices[i] = i
		}

		// Sort indices based on values
		sort.Slice(indices, func(i, j int) bool {
			vi := indices[i]
			vj := indices[j]
			cmp := compareArrayValues(sortChunk, vi, vj)
			if o.desc {
				return cmp > 0
			}
			return cmp < 0
		})

		newChunkSizes[chunkIdx] = int64(chunkLen)

		// Reorder each column
		for _, colName := range colNames {
			col, _ := input.Column(colName)
			dataChunk := col.Chunk(chunkIdx)
			reordered, err := reorderArray(ctx.Pool(), dataChunk, indices)
			if err != nil {
				releaseAllChunks()
				return nil, fmt.Errorf("SortOp: column %s: %w", colName, err)
			}
			allChunks[colName][chunkIdx] = reordered
		}
	}

	// Create new DataFrame with all chunks
	newDF := NewDataFrame()
	newDF.chunkSizes = newChunkSizes

	for _, colName := range colNames {
		if err := newDF.addChunkedColumn(colName, allChunks[colName]); err != nil {
			releaseAllChunks()
			return nil, err
		}

		// Copy field type and ID
		if ft, ok := input.GetFieldType(colName); ok {
			newDF.SetFieldType(colName, ft)
		}
		if fid, ok := input.GetFieldID(colName); ok {
			newDF.SetFieldID(colName, fid)
		}
	}

	return newDF, nil
}

// isComparableType checks if an Arrow data type is comparable for sorting.
func isComparableType(dt arrow.DataType) bool {
	switch dt.ID() {
	case arrow.INT8, arrow.INT16, arrow.INT32, arrow.INT64,
		arrow.UINT8, arrow.UINT16, arrow.UINT32, arrow.UINT64,
		arrow.FLOAT32, arrow.FLOAT64,
		arrow.STRING, arrow.LARGE_STRING:
		return true
	default:
		return false
	}
}

func (o *SortOp) String() string {
	order := "ASC"
	if o.desc {
		order = "DESC"
	}
	return fmt.Sprintf("Sort(%s %s)", o.column, order)
}

// compareArrayValues compares two values in an array.
func compareArrayValues(arr arrow.Array, i, j int) int {
	// Handle nulls
	if arr.IsNull(i) && arr.IsNull(j) {
		return 0
	}
	if arr.IsNull(i) {
		return -1
	}
	if arr.IsNull(j) {
		return 1
	}

	switch a := arr.(type) {
	case *array.Int8:
		return compareTyped(a, i, j)
	case *array.Int16:
		return compareTyped(a, i, j)
	case *array.Int32:
		return compareTyped(a, i, j)
	case *array.Int64:
		return compareTyped(a, i, j)
	case *array.Float32:
		return compareTyped(a, i, j)
	case *array.Float64:
		return compareTyped(a, i, j)
	case *array.String:
		return compareTyped(a, i, j)
	default:
		return 0
	}
}

// reorderArray reorders an array based on indices.
func reorderArray(pool memory.Allocator, data arrow.Array, indices []int) (arrow.Array, error) {
	return dispatchPickByIndices(pool, data, indices)
}

// -----------------------------------------------------------------------------
// LimitOp
// -----------------------------------------------------------------------------

// LimitOp limits the number of rows in each chunk.
type LimitOp struct {
	BaseOp
	limit  int64
	offset int64
}

func (o *LimitOp) Name() string      { return "Limit" }
func (o *LimitOp) Inputs() []string  { return []string{} } // Limit works on all columns
func (o *LimitOp) Outputs() []string { return []string{} } // Limit doesn't produce new columns

func (o *LimitOp) Execute(ctx *FuncContext, input *DataFrame) (*DataFrame, error) {
	// Collect all column chunks first
	colNames := input.ColumnNames()
	allChunks := make(map[string][]arrow.Array)
	for _, name := range colNames {
		allChunks[name] = make([]arrow.Array, input.NumChunks())
	}

	newChunkSizes := make([]int64, input.NumChunks())

	// releaseAllChunks is a helper to clean up on error
	releaseAllChunks := func() {
		for _, chunks := range allChunks {
			for _, chunk := range chunks {
				if chunk != nil {
					chunk.Release()
				}
			}
		}
	}

	// Process each chunk
	for chunkIdx := range input.NumChunks() {
		chunkSize := input.chunkSizes[chunkIdx]

		// Calculate actual offset and limit for this chunk
		start := min(o.offset, chunkSize)
		end := min(start+o.limit, chunkSize)

		newChunkSizes[chunkIdx] = end - start

		// Slice each column
		for _, colName := range colNames {
			col, _ := input.Column(colName)
			dataChunk := col.Chunk(chunkIdx)
			sliced, err := sliceArray(ctx.Pool(), dataChunk, int(start), int(end))
			if err != nil {
				releaseAllChunks()
				return nil, fmt.Errorf("LimitOp: column %s: %w", colName, err)
			}
			allChunks[colName][chunkIdx] = sliced
		}
	}

	// Create new DataFrame with all chunks
	newDF := NewDataFrame()
	newDF.chunkSizes = newChunkSizes

	for _, colName := range colNames {
		if err := newDF.addChunkedColumn(colName, allChunks[colName]); err != nil {
			releaseAllChunks()
			return nil, fmt.Errorf("LimitOp: %w", err)
		}

		// Copy field type and ID
		if ft, ok := input.GetFieldType(colName); ok {
			newDF.SetFieldType(colName, ft)
		}
		if fid, ok := input.GetFieldID(colName); ok {
			newDF.SetFieldID(colName, fid)
		}
	}

	return newDF, nil
}

func (o *LimitOp) String() string {
	if o.offset > 0 {
		return fmt.Sprintf("Limit(%d, offset=%d)", o.limit, o.offset)
	}
	return fmt.Sprintf("Limit(%d)", o.limit)
}

// =============================================================================
// Operator Registry
// =============================================================================

// OperatorFactory is a function that creates an Operator from OperatorRepr.
type OperatorFactory func(repr *OperatorRepr) (Operator, error)

var operatorRegistry = make(map[string]OperatorFactory)

// RegisterOperator registers an operator factory.
func RegisterOperator(opType string, factory OperatorFactory) {
	operatorRegistry[opType] = factory
}

// GetOperatorFactory returns the factory for the given operator type.
func GetOperatorFactory(opType string) (OperatorFactory, bool) {
	factory, ok := operatorRegistry[opType]
	return factory, ok
}

// =============================================================================
// Operator Factory Functions
// =============================================================================

// NewMapOpFromRepr creates a MapOp from an OperatorRepr.
func NewMapOpFromRepr(repr *OperatorRepr) (Operator, error) {
	if repr.Function == nil {
		return nil, fmt.Errorf("map operator requires function")
	}
	fn, err := FunctionFromRepr(repr.Function)
	if err != nil {
		return nil, fmt.Errorf("map function: %w", err)
	}
	if len(repr.Inputs) == 0 {
		return nil, fmt.Errorf("map operator requires inputs")
	}
	if len(repr.Outputs) == 0 {
		return nil, fmt.Errorf("map operator requires outputs")
	}
	return NewMapOp(fn, repr.Inputs, repr.Outputs)
}

// NewFilterOpFromRepr creates a FilterOp from an OperatorRepr.
func NewFilterOpFromRepr(repr *OperatorRepr) (Operator, error) {
	column, ok := repr.Params["column"].(string)
	if !ok || column == "" {
		return nil, fmt.Errorf("filter operator requires column")
	}
	return &FilterOp{column: column}, nil
}

// NewSelectOpFromRepr creates a SelectOp from an OperatorRepr.
func NewSelectOpFromRepr(repr *OperatorRepr) (Operator, error) {
	columnsInterface, ok := repr.Params["columns"]
	if !ok {
		return nil, fmt.Errorf("select operator requires columns")
	}
	columns, ok := columnsInterface.([]interface{})
	if !ok {
		// Try []string
		if colsStr, ok := columnsInterface.([]string); ok {
			return &SelectOp{columns: colsStr}, nil
		}
		return nil, fmt.Errorf("select operator requires columns to be a list")
	}
	if len(columns) == 0 {
		return nil, fmt.Errorf("select operator requires columns")
	}
	colsStr := make([]string, len(columns))
	for i, col := range columns {
		if colStr, ok := col.(string); ok {
			colsStr[i] = colStr
		} else {
			return nil, fmt.Errorf("select operator column[%d] must be a string", i)
		}
	}
	return &SelectOp{columns: colsStr}, nil
}

// NewSortOpFromRepr creates a SortOp from an OperatorRepr.
func NewSortOpFromRepr(repr *OperatorRepr) (Operator, error) {
	column, ok := repr.Params["column"].(string)
	if !ok || column == "" {
		return nil, fmt.Errorf("sort operator requires column")
	}
	desc := false
	if descVal, ok := repr.Params["desc"]; ok {
		if descBool, ok := descVal.(bool); ok {
			desc = descBool
		}
	}
	return &SortOp{column: column, desc: desc}, nil
}

// NewLimitOpFromRepr creates a LimitOp from an OperatorRepr.
func NewLimitOpFromRepr(repr *OperatorRepr) (Operator, error) {
	limitVal, ok := repr.Params["limit"]
	if !ok {
		return nil, fmt.Errorf("limit operator requires limit")
	}
	var limit int64
	switch v := limitVal.(type) {
	case int64:
		limit = v
	case int:
		limit = int64(v)
	case float64:
		limit = int64(v)
	default:
		return nil, fmt.Errorf("limit operator limit must be a number")
	}
	if limit <= 0 {
		return nil, fmt.Errorf("limit operator requires positive limit")
	}
	offset := int64(0)
	if offsetVal, ok := repr.Params["offset"]; ok {
		switch v := offsetVal.(type) {
		case int64:
			offset = v
		case int:
			offset = int64(v)
		case float64:
			offset = int64(v)
		}
	}
	return &LimitOp{limit: limit, offset: offset}, nil
}

// init registers all built-in operator factories.
func init() {
	RegisterOperator(OpTypeMap, NewMapOpFromRepr)
	RegisterOperator(OpTypeFilter, NewFilterOpFromRepr)
	RegisterOperator(OpTypeSelect, NewSelectOpFromRepr)
	RegisterOperator(OpTypeSort, NewSortOpFromRepr)
	RegisterOperator(OpTypeLimit, NewLimitOpFromRepr)
}

// sliceArray slices an array from start to end using zero-copy.
func sliceArray(_ memory.Allocator, data arrow.Array, start, end int) (arrow.Array, error) {
	result := array.NewSlice(data, int64(start), int64(end))
	return result, nil
}

// =============================================================================
// Parallel Processing Utilities
// =============================================================================

// ChunkProcessor is a function type that processes a single chunk.
// It receives the chunk index and returns the processed result or an error.
type ChunkProcessor func(chunkIdx int) (arrow.Array, error)

// ProcessChunksParallel processes chunks in parallel using goroutines.
// numChunks: number of chunks to process
// processor: function to process each chunk
// parallelism: maximum number of concurrent goroutines (0 means sequential)
// Returns: slice of processed arrays in order, or first error encountered
func ProcessChunksParallel(numChunks int, processor ChunkProcessor, parallelism int) ([]arrow.Array, error) {
	if numChunks == 0 {
		return []arrow.Array{}, nil
	}

	// Sequential processing if parallelism is 0 or 1, or only 1 chunk
	if parallelism <= 1 || numChunks == 1 {
		results := make([]arrow.Array, numChunks)
		for i := 0; i < numChunks; i++ {
			arr, err := processor(i)
			if err != nil {
				// Release already processed chunks
				for j := 0; j < i; j++ {
					if results[j] != nil {
						results[j].Release()
					}
				}
				return nil, err
			}
			results[i] = arr
		}
		return results, nil
	}

	// Parallel processing
	results := make([]arrow.Array, numChunks)
	errors := make([]error, numChunks)

	// Use a semaphore channel to limit parallelism
	sem := make(chan struct{}, parallelism)
	done := make(chan struct{})

	for i := 0; i < numChunks; i++ {
		sem <- struct{}{} // Acquire
		go func(idx int) {
			defer func() { <-sem }() // Release
			arr, err := processor(idx)
			results[idx] = arr
			errors[idx] = err
		}(i)
	}

	// Wait for all goroutines to complete
	for i := 0; i < parallelism; i++ {
		sem <- struct{}{}
	}
	close(done)

	// Check for errors and cleanup on failure
	var firstErr error
	for i := 0; i < numChunks; i++ {
		if errors[i] != nil && firstErr == nil {
			firstErr = errors[i]
		}
	}

	if firstErr != nil {
		// Release all results on error
		for _, arr := range results {
			if arr != nil {
				arr.Release()
			}
		}
		return nil, firstErr
	}

	return results, nil
}
