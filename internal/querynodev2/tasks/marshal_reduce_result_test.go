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
	"testing"

	"github.com/apache/arrow/go/v17/arrow/memory"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMarshalReduceResult_Basic(t *testing.T) {
	pool := memory.NewGoAllocator()

	// Build a simple ReduceResult with 2 NQs
	// NQ0: 3 results, NQ1: 2 results
	df := buildTestDF(pool,
		[][]int64{{1, 2, 3}, {4, 5}},
		[][]float32{{0.9, 0.8, 0.7}, {0.6, 0.5}},
	)
	defer df.Release()

	result := &ReduceResult{
		DF: df,
		Sources: [][]SegmentSource{
			{{InputIdx: 0, SegOffset: 10}, {InputIdx: 0, SegOffset: 20}, {InputIdx: 1, SegOffset: 30}},
			{{InputIdx: 0, SegOffset: 40}, {InputIdx: 1, SegOffset: 50}},
		},
	}

	data, err := MarshalReduceResult(result)
	require.NoError(t, err)

	// Verify structure
	assert.Equal(t, int64(2), data.NumQueries)
	assert.Equal(t, int64(3), data.TopK) // max of 3, 2
	assert.Equal(t, []int64{3, 2}, data.Topks)

	// Verify IDs
	require.NotNil(t, data.Ids.GetIntId())
	assert.Equal(t, []int64{1, 2, 3, 4, 5}, data.Ids.GetIntId().Data)

	// Verify Scores
	assert.Len(t, data.Scores, 5)
	assert.InDelta(t, float32(0.9), data.Scores[0], 0.001)
	assert.InDelta(t, float32(0.5), data.Scores[4], 0.001)
}

func TestMarshalReduceResult_Empty(t *testing.T) {
	pool := memory.NewGoAllocator()

	df := buildTestDF(pool, [][]int64{{}}, [][]float32{{}})
	defer df.Release()

	result := &ReduceResult{
		DF:      df,
		Sources: [][]SegmentSource{{}},
	}

	data, err := MarshalReduceResult(result)
	require.NoError(t, err)
	assert.Equal(t, int64(1), data.NumQueries)
	assert.Equal(t, int64(0), data.TopK)
	assert.Len(t, data.Scores, 0)
}

func TestMarshalReduceResult_Nil(t *testing.T) {
	_, err := MarshalReduceResult(nil)
	assert.Error(t, err)
}

func TestGroupBySegment(t *testing.T) {
	sources := [][]SegmentSource{
		// NQ0
		{
			{InputIdx: 0, SegOffset: 10, OriginalIdx: 0},
			{InputIdx: 1, SegOffset: 20, OriginalIdx: 0},
			{InputIdx: 0, SegOffset: 30, OriginalIdx: 1},
		},
		// NQ1
		{
			{InputIdx: 1, SegOffset: 40, OriginalIdx: 1},
		},
	}

	groups := GroupBySegment(sources)

	// Segment 0: offsets 10 (pos 0), 30 (pos 2)
	require.Len(t, groups[0], 2)
	assert.Equal(t, int64(10), groups[0][0].SegOffset)
	assert.Equal(t, 0, groups[0][0].ResultPos)
	assert.Equal(t, int64(30), groups[0][1].SegOffset)
	assert.Equal(t, 2, groups[0][1].ResultPos)

	// Segment 1: offsets 20 (pos 1), 40 (pos 3)
	require.Len(t, groups[1], 2)
	assert.Equal(t, int64(20), groups[1][0].SegOffset)
	assert.Equal(t, 1, groups[1][0].ResultPos)
	assert.Equal(t, int64(40), groups[1][1].SegOffset)
	assert.Equal(t, 3, groups[1][1].ResultPos)
}
