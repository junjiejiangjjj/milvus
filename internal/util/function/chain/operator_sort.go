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
	"sort"
	"strings"

	"github.com/apache/arrow/go/v17/arrow"
	"github.com/apache/arrow/go/v17/arrow/array"
	"github.com/apache/arrow/go/v17/arrow/memory"

	"github.com/milvus-io/milvus/internal/util/function/chain/types"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

func init() {
	MustRegisterOperator(
		types.OpTypeSort,
		statelessOperatorFactory(NewSortOpFromRepr),
	)
}

const (
	sortParamDesc       = "desc"
	sortParamOrders     = "orders"
	sortParamNullOrders = "null_orders"
	sortParamStable     = "stable"
	sortOrderAsc        = "asc"
	sortOrderDesc       = "desc"
	sortNullsFirst      = "nulls_first"
	sortNullsLast       = "nulls_last"
)

// SortKey describes one ordered SortOp input.
type SortKey struct {
	Column     string
	Descending bool
	NullsFirst bool
}

// SortOp sorts every DataFrame chunk independently using an ordered key list.
// All DataFrame columns are reordered with the same indices.
type SortOp struct {
	BaseOp
	keys   []SortKey
	stable bool
}

func newSortOp(column string, desc bool, tieBreakCol string) *SortOp {
	// Preserve the legacy fluent/Repr format: the primary key follows desc,
	// NULLs are always last, and the optional tie-break key is ascending.
	keys := []SortKey{{Column: column, Descending: desc, NullsFirst: false}}
	if tieBreakCol != "" && tieBreakCol != column {
		keys = append(keys, SortKey{Column: tieBreakCol, NullsFirst: false})
	}
	return newSortOpWithKeys(keys, true)
}

func newSortOpWithKeys(keys []SortKey, stable bool) *SortOp {
	inputs := make([]string, len(keys))
	for i, key := range keys {
		inputs[i] = key.Column
	}
	return &SortOp{
		BaseOp: BaseOp{
			inputs:  inputs,
			outputs: []string{}, // Sort doesn't produce new columns
		},
		keys:   append([]SortKey(nil), keys...),
		stable: stable,
	}
}

// NewSortOp creates a SortOp using explicit ordered keys.
func NewSortOp(keys []SortKey, stable bool) (*SortOp, error) {
	if len(keys) == 0 {
		return nil, merr.WrapErrParameterMissingMsg("sort_op: at least one sort key is required")
	}
	for i, key := range keys {
		if strings.TrimSpace(key.Column) == "" {
			return nil, merr.WrapErrParameterInvalidMsg("sort_op: sort key[%d] column is empty", i)
		}
	}
	return newSortOpWithKeys(keys, stable), nil
}

// Column returns the sort column name.
func (o *SortOp) Column() string {
	if len(o.keys) > 0 {
		return o.keys[0].Column
	}
	return ""
}

// Keys returns a copy of the ordered sort keys.
func (o *SortOp) Keys() []SortKey {
	return append([]SortKey(nil), o.keys...)
}

// Stable reports whether equal rows preserve their upstream order.
func (o *SortOp) Stable() bool { return o.stable }

func (o *SortOp) Name() string { return "Sort" }

// Inputs and Outputs are inherited from BaseOp

func (o *SortOp) Execute(ctx *types.FuncContext, input *DataFrame) (*DataFrame, error) {
	if input == nil {
		return nil, merr.WrapErrServiceInternalMsg("sort_op: input DataFrame is nil")
	}
	if len(o.keys) == 0 {
		return nil, merr.WrapErrServiceInternalMsg("sort_op: no sort keys configured")
	}

	keyColumns := make([]*arrow.Chunked, len(o.keys))
	for i, key := range o.keys {
		col := input.Column(key.Column)
		if col == nil {
			return nil, merr.WrapErrServiceInternalMsg("sort_op: key[%d] column %q not found", i, key.Column)
		}
		if !isComparableType(col.DataType()) {
			return nil, merr.WrapErrServiceInternalMsg(
				"sort_op: key[%d] column %q has non-comparable type %s", i, key.Column, col.DataType().Name())
		}
		keyColumns[i] = col
	}

	colNames := input.ColumnNames()
	collector := NewChunkCollector(colNames, input.NumChunks())
	defer collector.Release()

	newChunkSizes := input.ChunkSizes()

	// Process each chunk independently
	for chunkIdx := range input.NumChunks() {
		keyChunks := make([]arrow.Array, len(keyColumns))
		for i, col := range keyColumns {
			keyChunks[i] = col.Chunk(chunkIdx)
		}
		chunkLen := int(newChunkSizes[chunkIdx])

		// Build sort indices
		indices := make([]int, chunkLen)
		for i := 0; i < chunkLen; i++ {
			indices[i] = i
		}

		less := makeSortRowLess(keyChunks, o.keys)
		if o.stable {
			sort.SliceStable(indices, func(i, j int) bool {
				return less(indices[i], indices[j])
			})
		} else {
			sort.Slice(indices, func(i, j int) bool {
				return less(indices[i], indices[j])
			})
		}

		// Reorder each column
		for _, colName := range colNames {
			col := input.Column(colName)
			dataChunk := col.Chunk(chunkIdx)
			reordered, err := reorderArray(ctx.Pool(), dataChunk, indices)
			if err != nil {
				return nil, merr.WrapErrServiceInternalMsg("sort_op: column %s: %v", colName, err)
			}
			collector.Set(colName, chunkIdx, reordered)
		}
	}

	// Create new DataFrame with all chunks
	builder := NewDataFrameBuilder()
	defer builder.Release()

	builder.SetChunkSizes(newChunkSizes)
	builder.CopyAllMetadata(input)

	for _, colName := range colNames {
		if err := builder.AddColumnFromChunks(colName, collector.Consume(colName)); err != nil {
			return nil, err
		}
		builder.CopyFieldMetadata(input, colName)
	}

	return builder.Build(), nil
}

type rowLessFunc func(i, j int) bool

func makeSortRowLess(keyChunks []arrow.Array, keys []SortKey) rowLessFunc {
	// Keep the hot rerank path specialized: Float32 score DESC followed by
	// Int64 id ASC. With no NULLs, the configured NULL ordering is irrelevant.
	if len(keys) == 2 && keys[0].Descending && !keys[1].Descending {
		if scoreArr, ok := keyChunks[0].(*array.Float32); ok && scoreArr.NullN() == 0 {
			if idArr, ok := keyChunks[1].(*array.Int64); ok && idArr.NullN() == 0 {
				return func(i, j int) bool {
					si := scoreArr.Value(i)
					sj := scoreArr.Value(j)
					if si != sj {
						return si > sj
					}
					return idArr.Value(i) < idArr.Value(j)
				}
			}
		}
	}

	return func(i, j int) bool {
		for keyIdx, key := range keys {
			arr := keyChunks[keyIdx]
			iNull := arr.IsNull(i)
			jNull := arr.IsNull(j)
			switch {
			case iNull && jNull:
				continue
			case iNull:
				return key.NullsFirst
			case jNull:
				return !key.NullsFirst
			}

			cmp := compareArrayValues(arr, i, j)
			if cmp == 0 {
				continue
			}
			if key.Descending {
				return cmp > 0
			}
			return cmp < 0
		}
		return false
	}
}

// isComparableType checks if an Arrow data type is comparable for sorting.
func isComparableType(dt arrow.DataType) bool {
	switch dt.ID() {
	case arrow.BOOL,
		arrow.INT8, arrow.INT16, arrow.INT32, arrow.INT64,
		arrow.UINT8, arrow.UINT16, arrow.UINT32, arrow.UINT64,
		arrow.FLOAT32, arrow.FLOAT64,
		arrow.STRING:
		return true
	default:
		return false
	}
}

func (o *SortOp) String() string {
	parts := make([]string, len(o.keys))
	for i, key := range o.keys {
		order := "ASC"
		if key.Descending {
			order = "DESC"
		}
		parts[i] = fmt.Sprintf("%s %s", key.Column, order)
	}
	return fmt.Sprintf("Sort(%s)", strings.Join(parts, ", "))
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
	case *array.Boolean:
		iv := a.Value(i)
		jv := a.Value(j)
		switch {
		case iv == jv:
			return 0
		case !iv:
			return -1
		default:
			return 1
		}
	case *array.Int8:
		return compareTyped(a, i, j)
	case *array.Int16:
		return compareTyped(a, i, j)
	case *array.Int32:
		return compareTyped(a, i, j)
	case *array.Int64:
		return compareTyped(a, i, j)
	case *array.Uint8:
		return compareTyped(a, i, j)
	case *array.Uint16:
		return compareTyped(a, i, j)
	case *array.Uint32:
		return compareTyped(a, i, j)
	case *array.Uint64:
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

// NewSortOpFromRepr creates a SortOp from an OperatorRepr.
func NewSortOpFromRepr(repr *OperatorRepr) (Operator, error) {
	if repr == nil {
		return nil, merr.WrapErrParameterInvalidMsg("sort_op: representation is nil")
	}
	if len(repr.Inputs) == 0 {
		return nil, merr.WrapErrParameterMissingMsg("sort_op: column is required")
	}

	if _, hasOrders := repr.Params[sortParamOrders]; hasOrders {
		return newMultiKeySortOpFromRepr(repr)
	}
	return newLegacySortOpFromRepr(repr)
}

func newMultiKeySortOpFromRepr(repr *OperatorRepr) (Operator, error) {
	reader := types.NewParamReader("sort_op", repr.Params)
	stable, err := reader.Bool(sortParamStable, false, true)
	if err != nil {
		return nil, err
	}

	if _, hasDesc := repr.Params[sortParamDesc]; hasDesc {
		return nil, merr.WrapErrParameterInvalidMsg(
			"sort_op: parameters %q and %q cannot be used together", sortParamOrders, sortParamDesc)
	}
	orders, err := reader.StringSlice(sortParamOrders, true)
	if err != nil {
		return nil, err
	}
	if len(orders) != len(repr.Inputs) {
		return nil, merr.WrapErrParameterInvalidMsg(
			"sort_op: orders count (%d) must match inputs count (%d)", len(orders), len(repr.Inputs))
	}

	var nullOrders []string
	if _, hasNullOrders := repr.Params[sortParamNullOrders]; hasNullOrders {
		nullOrders, err = reader.StringSlice(sortParamNullOrders, true)
		if err != nil {
			return nil, err
		}
		if len(nullOrders) != len(repr.Inputs) {
			return nil, merr.WrapErrParameterInvalidMsg(
				"sort_op: null_orders count (%d) must match inputs count (%d)", len(nullOrders), len(repr.Inputs))
		}
	}

	keys := make([]SortKey, len(repr.Inputs))
	for i, column := range repr.Inputs {
		order := strings.ToLower(strings.TrimSpace(orders[i]))
		switch order {
		case sortOrderAsc, sortOrderDesc:
		default:
			return nil, merr.WrapErrParameterInvalidMsg(
				"sort_op: orders[%d] must be %q or %q", i, sortOrderAsc, sortOrderDesc)
		}
		descending := order == sortOrderDesc
		nullsFirst := descending // ASC defaults LAST; DESC defaults FIRST.
		if nullOrders != nil {
			nullOrder := strings.ToLower(strings.TrimSpace(nullOrders[i]))
			switch nullOrder {
			case sortNullsFirst:
				nullsFirst = true
			case sortNullsLast:
				nullsFirst = false
			default:
				return nil, merr.WrapErrParameterInvalidMsg(
					"sort_op: null_orders[%d] must be %q or %q", i, sortNullsFirst, sortNullsLast)
			}
		}
		keys[i] = SortKey{Column: column, Descending: descending, NullsFirst: nullsFirst}
	}
	return NewSortOp(keys, stable)
}

func newLegacySortOpFromRepr(repr *OperatorRepr) (Operator, error) {
	reader := types.NewParamReader("sort_op", repr.Params)
	stable, err := reader.Bool(sortParamStable, false, true)
	if err != nil {
		return nil, err
	}
	if _, hasNullOrders := repr.Params[sortParamNullOrders]; hasNullOrders {
		return nil, merr.WrapErrParameterInvalidMsg(
			"sort_op: parameter %q requires %q", sortParamNullOrders, sortParamOrders)
	}
	if len(repr.Inputs) > 2 {
		return nil, merr.WrapErrParameterInvalidMsg(
			"sort_op: legacy format expects at most 2 input columns, got %d", len(repr.Inputs))
	}

	desc, err := reader.Bool(sortParamDesc, false, false)
	if err != nil {
		return nil, err
	}
	column := repr.Inputs[0]
	tieBreakCol := types.IDFieldName
	if len(repr.Inputs) > 1 {
		tieBreakCol = repr.Inputs[1]
	}
	op := newSortOp(column, desc, tieBreakCol)
	op.stable = stable
	return op, nil
}
