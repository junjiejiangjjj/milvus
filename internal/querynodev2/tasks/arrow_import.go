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
	"github.com/apache/arrow/go/v17/arrow"
	"github.com/apache/arrow/go/v17/arrow/array"

	"github.com/milvus-io/milvus/internal/util/function/chain"
	"github.com/milvus-io/milvus/pkg/v2/util/merr"
)

// DataFrameFromArrowRecord converts an Arrow RecordBatch into a chain DataFrame,
// splitting rows into NQ chunks based on the provided topkPerNQ slice.
//
// The RecordBatch columns (e.g., $id, $score, $seg_offset) are sliced into chunks
// where chunk[i] has topkPerNQ[i] rows. This preserves the per-query grouping
// needed by HeapMergeReduce.
func DataFrameFromArrowRecord(record arrow.Record, topkPerNQ []int64) (*chain.DataFrame, error) {
	if record == nil {
		return nil, merr.WrapErrServiceInternal("nil Arrow record")
	}

	nq := len(topkPerNQ)
	numCols := int(record.NumCols())

	if numCols == 0 {
		return nil, merr.WrapErrServiceInternal("Arrow record has no columns")
	}

	// Verify total rows match
	var totalRows int64
	for _, k := range topkPerNQ {
		totalRows += k
	}
	if totalRows != record.NumRows() {
		return nil, merr.WrapErrServiceInternal(
			"topkPerNQ sum does not match record row count")
	}

	builder := chain.NewDataFrameBuilder()
	defer builder.Release()
	builder.SetChunkSizes(topkPerNQ)

	// Split each column into NQ chunks
	for colIdx := 0; colIdx < numCols; colIdx++ {
		col := record.Column(colIdx)
		colName := record.Schema().Field(colIdx).Name

		chunks := make([]arrow.Array, nq)
		offset := int64(0)
		for i := 0; i < nq; i++ {
			length := topkPerNQ[i]
			chunks[i] = array.NewSlice(col, offset, offset+length)
			offset += length
		}

		if err := builder.AddColumnFromChunks(colName, chunks); err != nil {
			return nil, err
		}
	}

	return builder.Build(), nil
}
