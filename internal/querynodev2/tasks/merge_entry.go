/*
 * Licensed to the LF AI & Data foundation under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package tasks

import (
	"math"
	"sort"

	"github.com/apache/arrow/go/v17/arrow"
	"github.com/apache/arrow/go/v17/arrow/array"
)

// EPSILON matches C++ common/Consts.h: const float EPSILON = 0.0000000119
const EPSILON = 0.0000000119

// mergeEntry corresponds to C++ SearchResultPair (ReduceStructure.h:23-82).
// It represents one segment's sorted results for a single NQ chunk,
// with a cursor that advances through sorted indices.
type mergeEntry struct {
	inputIdx int   // which input DataFrame this came from
	cursor   int   // current position in indices
	indices  []int // pre-sorted indices: score DESC, PK ASC

	idInt64  *array.Int64   // int64 PK array (one of idInt64/idString is set)
	idString *array.String  // varchar PK array
	scoreArr *array.Float32 // $score array

	segOffsetArr *array.Int64 // $seg_offset array (for Late Materialization)
	groupByArr   arrow.Array  // $group_by array (optional, for GroupBy mode)
}

func (e *mergeEntry) scoreVal() float32 {
	return e.scoreArr.Value(e.indices[e.cursor])
}

func (e *mergeEntry) idInt64Val() int64 {
	return e.idInt64.Value(e.indices[e.cursor])
}

func (e *mergeEntry) idStringVal() string {
	return e.idString.Value(e.indices[e.cursor])
}

func (e *mergeEntry) segOffsetVal() int64 {
	if e.segOffsetArr == nil {
		return -1
	}
	return e.segOffsetArr.Value(e.indices[e.cursor])
}

func (e *mergeEntry) done() bool {
	return e.cursor >= len(e.indices)
}

func (e *mergeEntry) advance() bool {
	e.cursor++
	return e.cursor < len(e.indices)
}

// greaterInt64 corresponds to C++ SearchResultPair::operator> (ReduceStructure.h:58-64)
// for int64 PK: if scores are equal within EPSILON, compare PK ASC.
func (e *mergeEntry) greaterInt64(other *mergeEntry) bool {
	diff := float64(e.scoreVal()) - float64(other.scoreVal())
	if math.Abs(diff) < EPSILON {
		return e.idInt64Val() < other.idInt64Val() // equal score → PK ASC
	}
	return diff > 0 // score DESC
}

// greaterString is the varchar PK variant of greater.
func (e *mergeEntry) greaterString(other *mergeEntry) bool {
	diff := float64(e.scoreVal()) - float64(other.scoreVal())
	if math.Abs(diff) < EPSILON {
		return e.idStringVal() < other.idStringVal()
	}
	return diff > 0
}

// mergeHeapInt64 implements heap.Interface for max-heap with int64 PK.
// Corresponds to C++ SearchResultPairComparator (ReduceStructure.h:84-88):
// rhs > lhs for max-heap.
type mergeHeapInt64 []*mergeEntry

func (h mergeHeapInt64) Len() int           { return len(h) }
func (h mergeHeapInt64) Less(i, j int) bool { return h[i].greaterInt64(h[j]) }
func (h mergeHeapInt64) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *mergeHeapInt64) Push(x interface{}) {
	*h = append(*h, x.(*mergeEntry))
}

func (h *mergeHeapInt64) Pop() interface{} {
	old := *h
	n := len(old)
	item := old[n-1]
	old[n-1] = nil // avoid memory leak
	*h = old[:n-1]
	return item
}

// mergeHeapString implements heap.Interface for max-heap with varchar PK.
type mergeHeapString []*mergeEntry

func (h mergeHeapString) Len() int           { return len(h) }
func (h mergeHeapString) Less(i, j int) bool { return h[i].greaterString(h[j]) }
func (h mergeHeapString) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *mergeHeapString) Push(x interface{}) {
	*h = append(*h, x.(*mergeEntry))
}

func (h *mergeHeapString) Pop() interface{} {
	old := *h
	n := len(old)
	item := old[n-1]
	old[n-1] = nil
	*h = old[:n-1]
	return item
}

// buildSortedIndicesInt64 corresponds to C++ SortEqualScoresByPks (Reduce.cpp:180-282).
// Returns indices sorted by score DESC, with equal-score ranges sorted by PK ASC.
func buildSortedIndicesInt64(idArr *array.Int64, scoreArr *array.Float32) []int {
	n := idArr.Len()
	if n == 0 {
		return nil
	}
	indices := make([]int, n)
	for i := range indices {
		indices[i] = i
	}
	sort.SliceStable(indices, func(a, b int) bool {
		sa := float64(scoreArr.Value(indices[a]))
		sb := float64(scoreArr.Value(indices[b]))
		diff := sa - sb
		if math.Abs(diff) < EPSILON {
			return idArr.Value(indices[a]) < idArr.Value(indices[b]) // PK ASC
		}
		return diff > 0 // score DESC
	})
	return indices
}

// buildSortedIndicesString is the varchar PK variant.
func buildSortedIndicesString(idArr *array.String, scoreArr *array.Float32) []int {
	n := idArr.Len()
	if n == 0 {
		return nil
	}
	indices := make([]int, n)
	for i := range indices {
		indices[i] = i
	}
	sort.SliceStable(indices, func(a, b int) bool {
		sa := float64(scoreArr.Value(indices[a]))
		sb := float64(scoreArr.Value(indices[b]))
		diff := sa - sb
		if math.Abs(diff) < EPSILON {
			return idArr.Value(indices[a]) < idArr.Value(indices[b])
		}
		return diff > 0
	})
	return indices
}
