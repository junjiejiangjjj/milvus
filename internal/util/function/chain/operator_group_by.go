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

	"github.com/apache/arrow/go/v17/arrow"
	"github.com/apache/arrow/go/v17/arrow/array"

	"github.com/milvus-io/milvus/internal/util/function/chain/types"
)

// GroupByOp groups rows by a field, keeps top N rows per group, and limits the number of groups.
// This operator is designed for grouping search scenarios.
//
// Parameters:
//   - groupByField: the field to group by
//   - groupSize: maximum rows per group (sorted by $score DESC)
//   - limit: maximum number of groups to return
//   - offset: number of groups to skip
//
// The operator also adds a $group_score column containing the max score of each group.
type GroupByOp struct {
	BaseOp
	groupByField string
	groupSize    int64
	limit        int64
	offset       int64
}

// NewGroupByOp creates a new GroupByOp.
func NewGroupByOp(groupByField string, groupSize, limit, offset int64) *GroupByOp {
	return &GroupByOp{
		BaseOp: BaseOp{
			inputs:  []string{groupByField, types.ScoreFieldName},
			outputs: []string{GroupScoreFieldName},
		},
		groupByField: groupByField,
		groupSize:    groupSize,
		limit:        limit,
		offset:       offset,
	}
}

// GroupScoreFieldName is the name of the group score column added by GroupByOp.
const GroupScoreFieldName = "$group_score"

func (o *GroupByOp) Name() string { return "GroupBy" }

func (o *GroupByOp) String() string {
	return fmt.Sprintf("GroupBy(%s, groupSize=%d, limit=%d, offset=%d)",
		o.groupByField, o.groupSize, o.limit, o.offset)
}

func (o *GroupByOp) Execute(ctx *types.FuncContext, input *DataFrame) (*DataFrame, error) {
	// Validate columns exist
	groupCol := input.Column(o.groupByField)
	if groupCol == nil {
		return nil, fmt.Errorf("group_by_op: column %q not found", o.groupByField)
	}
	scoreCol := input.Column(types.ScoreFieldName)
	if scoreCol == nil {
		return nil, fmt.Errorf("group_by_op: column %q not found", types.ScoreFieldName)
	}

	numChunks := input.NumChunks()
	colNames := input.ColumnNames()

	// Prepare collectors
	collector := NewChunkCollector(colNames, numChunks)
	defer collector.Release()

	// Prepare group score builder
	groupScoreChunks := make([]arrow.Array, numChunks)
	newChunkSizes := make([]int64, numChunks)

	// Process each chunk independently
	for chunkIdx := 0; chunkIdx < numChunks; chunkIdx++ {
		result, err := o.processChunk(ctx, input, chunkIdx)
		if err != nil {
			// Release any already built chunks
			for i := 0; i < chunkIdx; i++ {
				if groupScoreChunks[i] != nil {
					groupScoreChunks[i].Release()
				}
			}
			return nil, err
		}

		newChunkSizes[chunkIdx] = int64(len(result.indices))
		groupScoreChunks[chunkIdx] = result.groupScores

		// Reorder existing columns by indices
		for _, colName := range colNames {
			col := input.Column(colName)
			dataChunk := col.Chunk(chunkIdx)
			reordered, err := dispatchPickByIndices(ctx.Pool(), dataChunk, result.indices)
			if err != nil {
				result.groupScores.Release()
				for i := 0; i < chunkIdx; i++ {
					if groupScoreChunks[i] != nil {
						groupScoreChunks[i].Release()
					}
				}
				return nil, fmt.Errorf("group_by_op: reorder column %s: %w", colName, err)
			}
			collector.Set(colName, chunkIdx, reordered)
		}
	}

	// Build output DataFrame
	builder := NewDataFrameBuilder()
	defer builder.Release()

	builder.SetChunkSizes(newChunkSizes)

	// Add existing columns
	for _, colName := range colNames {
		if err := builder.AddColumnFromChunks(colName, collector.Consume(colName)); err != nil {
			for _, chunk := range groupScoreChunks {
				if chunk != nil {
					chunk.Release()
				}
			}
			return nil, err
		}
		builder.CopyFieldMetadata(input, colName)
	}

	// Add group score column
	if err := builder.AddColumnFromChunks(GroupScoreFieldName, groupScoreChunks); err != nil {
		return nil, err
	}

	return builder.Build(), nil
}

// chunkResult holds the result of processing a single chunk.
type chunkResult struct {
	indices     []int       // Row indices in output order
	groupScores arrow.Array // Group score for each output row
}

// processChunk processes a single chunk and returns the result.
func (o *GroupByOp) processChunk(ctx *types.FuncContext, input *DataFrame, chunkIdx int) (*chunkResult, error) {
	groupCol := input.Column(o.groupByField)
	scoreCol := input.Column(types.ScoreFieldName)

	groupChunk := groupCol.Chunk(chunkIdx)
	scoreChunk := scoreCol.Chunk(chunkIdx).(*array.Float32)
	chunkLen := groupChunk.Len()

	// Step 1: Build groups
	groups := o.buildGroups(groupChunk, scoreChunk, chunkLen)

	// Step 2: Sort rows within each group by score DESC, keep top groupSize
	for _, g := range groups {
		o.sortAndLimitGroup(g, scoreChunk)
	}

	// Step 3: Sort groups by group score DESC
	sort.Slice(groups, func(i, j int) bool {
		return groups[i].maxScore > groups[j].maxScore
	})

	// Step 4: Apply offset and limit on groups
	startGroup := int(o.offset)
	endGroup := int(o.offset + o.limit)
	if startGroup > len(groups) {
		startGroup = len(groups)
	}
	if endGroup > len(groups) {
		endGroup = len(groups)
	}
	selectedGroups := groups[startGroup:endGroup]

	// Step 5: Build output indices and group scores
	indices := make([]int, 0)
	groupScores := make([]float32, 0)

	for _, g := range selectedGroups {
		for _, idx := range g.rowIndices {
			indices = append(indices, idx)
			groupScores = append(groupScores, g.maxScore)
		}
	}

	// Build group score array
	groupScoreBuilder := array.NewFloat32Builder(ctx.Pool())
	defer groupScoreBuilder.Release()
	groupScoreBuilder.AppendValues(groupScores, nil)

	return &chunkResult{
		indices:     indices,
		groupScores: groupScoreBuilder.NewArray(),
	}, nil
}

// group represents a group of rows.
type group struct {
	key        any     // Group key value
	rowIndices []int   // Row indices belonging to this group
	maxScore   float32 // Max score in this group
}

// buildGroups builds groups from the chunk.
func (o *GroupByOp) buildGroups(groupChunk arrow.Array, scoreChunk *array.Float32, chunkLen int) []*group {
	groupMap := make(map[any]*group)
	groupOrder := make([]any, 0) // Maintain appearance order

	for i := 0; i < chunkLen; i++ {
		key := getArrayValue(groupChunk, i)
		score := scoreChunk.Value(i)

		if g, exists := groupMap[key]; exists {
			g.rowIndices = append(g.rowIndices, i)
			if score > g.maxScore {
				g.maxScore = score
			}
		} else {
			g := &group{
				key:        key,
				rowIndices: []int{i},
				maxScore:   score,
			}
			groupMap[key] = g
			groupOrder = append(groupOrder, key)
		}
	}

	// Return groups in appearance order
	result := make([]*group, 0, len(groupOrder))
	for _, key := range groupOrder {
		result = append(result, groupMap[key])
	}
	return result
}

// sortAndLimitGroup sorts rows within a group by score DESC and keeps top groupSize.
func (o *GroupByOp) sortAndLimitGroup(g *group, scoreChunk *array.Float32) {
	// Sort by score DESC
	sort.Slice(g.rowIndices, func(i, j int) bool {
		return scoreChunk.Value(g.rowIndices[i]) > scoreChunk.Value(g.rowIndices[j])
	})

	// Keep top groupSize
	if int64(len(g.rowIndices)) > o.groupSize {
		g.rowIndices = g.rowIndices[:o.groupSize]
	}

	// Update maxScore (should be the first element after sorting)
	if len(g.rowIndices) > 0 {
		g.maxScore = scoreChunk.Value(g.rowIndices[0])
	}
}

// NewGroupByOpFromRepr creates a GroupByOp from an OperatorRepr.
func NewGroupByOpFromRepr(repr *OperatorRepr) (Operator, error) {
	field, ok := repr.Params["field"].(string)
	if !ok || field == "" {
		return nil, fmt.Errorf("group_by_op: field is required")
	}

	groupSize, err := getInt64Param(repr.Params, "group_size")
	if err != nil {
		return nil, fmt.Errorf("group_by_op: %w", err)
	}
	if groupSize <= 0 {
		return nil, fmt.Errorf("group_by_op: group_size must be positive")
	}

	limit, err := getInt64Param(repr.Params, "limit")
	if err != nil {
		return nil, fmt.Errorf("group_by_op: %w", err)
	}
	if limit <= 0 {
		return nil, fmt.Errorf("group_by_op: limit must be positive")
	}

	offset := int64(0)
	if _, ok := repr.Params["offset"]; ok {
		offset, err = getInt64Param(repr.Params, "offset")
		if err != nil {
			return nil, fmt.Errorf("group_by_op: %w", err)
		}
	}

	return NewGroupByOp(field, groupSize, limit, offset), nil
}

// getInt64Param extracts an int64 parameter from a map.
func getInt64Param(params map[string]interface{}, key string) (int64, error) {
	val, ok := params[key]
	if !ok {
		return 0, fmt.Errorf("%s is required", key)
	}
	switch v := val.(type) {
	case int64:
		return v, nil
	case int:
		return int64(v), nil
	case float64:
		return int64(v), nil
	default:
		return 0, fmt.Errorf("%s must be a number", key)
	}
}

func init() {
	MustRegisterOperator(types.OpTypeGroupBy, NewGroupByOpFromRepr)
}
