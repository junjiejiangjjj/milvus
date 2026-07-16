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

//go:build cgo

package pyudf

import (
	"testing"

	"github.com/apache/arrow/go/v17/arrow"
	"github.com/apache/arrow/go/v17/arrow/array"
	"github.com/apache/arrow/go/v17/arrow/cdata"
	"github.com/apache/arrow/go/v17/arrow/memory"
	"github.com/apache/arrow/go/v17/arrow/memory/mallocator"
	"github.com/cockroachdb/errors"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

func TestPyUDFIdentityRoundTrip(t *testing.T) {
	allocator := memory.NewCheckedAllocator(mallocator.NewMallocator())
	defer allocator.AssertSize(t, 0)

	intInput := newPyUDFTestInt64Chunked(t, allocator)
	stringInput := newPyUDFTestStringChunked(t, allocator)
	inputs := []*arrow.Chunked{intInput, stringInput}
	inputTypes := []arrow.DataType{intInput.DataType(), stringInput.DataType()}
	outputs, err := runPyUDFIdentity(inputs)
	require.NoError(t, err)
	require.Len(t, outputs, len(inputs))

	intInput.Release()
	stringInput.Release()

	for outputIdx, output := range outputs {
		require.Len(t, output.Chunks(), 2)
		assert.True(t, arrow.TypeEqual(inputTypes[outputIdx], output.DataType()))
	}

	assert.Equal(t, 1, outputs[0].Chunk(0).Data().Offset())
	assert.Equal(t, 1, outputs[1].Chunk(0).Data().Offset())
	assert.Equal(t, 2, outputs[0].Len())
	assert.Equal(t, 2, outputs[1].Len())
	assert.Equal(t, 1, outputs[0].NullN())
	assert.Equal(t, 1, outputs[1].NullN())
	assert.Equal(t, "20", outputs[0].Chunk(0).ValueStr(0))
	assert.True(t, outputs[0].Chunk(0).IsNull(1))
	assert.Equal(t, "beta", outputs[1].Chunk(0).ValueStr(0))
	assert.True(t, outputs[1].Chunk(0).IsNull(1))
	assert.Zero(t, outputs[0].Chunk(1).Len())
	assert.Zero(t, outputs[1].Chunk(1).Len())

	releasePyUDFChunked(outputs)
}

func TestPyUDFIdentityValidation(t *testing.T) {
	allocator := memory.NewCheckedAllocator(mallocator.NewMallocator())
	defer allocator.AssertSize(t, 0)

	oneChunk := newPyUDFTestInt64Values(allocator, []int64{1}, nil)
	twoChunksFirst := newPyUDFTestInt64Values(allocator, []int64{1}, nil)
	twoChunksSecond := newPyUDFTestInt64Values(allocator, []int64{2}, nil)
	mismatchedRows := newPyUDFTestInt64Values(allocator, []int64{1, 2}, nil)
	defer oneChunk.Release()
	defer twoChunksFirst.Release()
	defer twoChunksSecond.Release()
	defer mismatchedRows.Release()

	columnOneChunk := newPyUDFTestChunked(arrow.PrimitiveTypes.Int64, oneChunk)
	columnTwoChunks := newPyUDFTestChunked(arrow.PrimitiveTypes.Int64, twoChunksFirst, twoChunksSecond)
	columnMismatchedRows := newPyUDFTestChunked(arrow.PrimitiveTypes.Int64, mismatchedRows)
	defer columnOneChunk.Release()
	defer columnTwoChunks.Release()
	defer columnMismatchedRows.Release()

	_, err := runPyUDFIdentity([]*arrow.Chunked{columnOneChunk, columnTwoChunks})
	assert.ErrorIs(t, err, merr.ErrServiceInternal)
	_, err = runPyUDFIdentity([]*arrow.Chunked{columnOneChunk, columnMismatchedRows})
	assert.ErrorIs(t, err, merr.ErrServiceInternal)
	_, err = runPyUDFIdentity([]*arrow.Chunked{nil})
	assert.ErrorIs(t, err, merr.ErrServiceInternal)
}

func TestPyUDFIdentityPartialImportFailureReleasesDescriptors(t *testing.T) {
	allocator := memory.NewCheckedAllocator(mallocator.NewMallocator())
	defer allocator.AssertSize(t, 0)

	input := newPyUDFTestInt64Chunked(t, allocator)
	importCount := 0
	outputs, err := runPyUDFIdentityWithImporter([]*arrow.Chunked{input}, func(arr *cdata.CArrowArray, schema *cdata.CArrowSchema) (arrow.Field, arrow.Array, error) {
		importCount++
		if importCount == 2 {
			return arrow.Field{}, nil, errors.New("injected import failure")
		}
		return cdata.ImportCArray(arr, schema)
	})
	require.Error(t, err)
	assert.ErrorIs(t, err, merr.ErrServiceInternal)
	assert.Nil(t, outputs)
	input.Release()
}

func TestPyUDFIdentityEmptyInputs(t *testing.T) {
	outputs, err := runPyUDFIdentity(nil)
	require.NoError(t, err)
	assert.Empty(t, outputs)
}

func newPyUDFTestInt64Chunked(t *testing.T, allocator memory.Allocator) *arrow.Chunked {
	t.Helper()
	base := newPyUDFTestInt64Values(allocator, []int64{10, 20, 30}, []bool{true, true, false})
	slice := array.NewSlice(base, 1, 3)
	base.Release()
	empty := newPyUDFTestInt64Values(allocator, nil, nil)
	chunked := newPyUDFTestChunked(arrow.PrimitiveTypes.Int64, slice, empty)
	slice.Release()
	empty.Release()
	return chunked
}

func newPyUDFTestStringChunked(t *testing.T, allocator memory.Allocator) *arrow.Chunked {
	t.Helper()
	builder := array.NewStringBuilder(allocator)
	builder.Append("alpha")
	builder.Append("beta")
	builder.AppendNull()
	base := builder.NewArray()
	builder.Release()
	slice := array.NewSlice(base, 1, 3)
	base.Release()

	emptyBuilder := array.NewStringBuilder(allocator)
	empty := emptyBuilder.NewArray()
	emptyBuilder.Release()
	chunked := newPyUDFTestChunked(arrow.BinaryTypes.String, slice, empty)
	slice.Release()
	empty.Release()
	return chunked
}

func newPyUDFTestInt64Values(allocator memory.Allocator, values []int64, valid []bool) arrow.Array {
	builder := array.NewInt64Builder(allocator)
	builder.AppendValues(values, valid)
	result := builder.NewArray()
	builder.Release()
	return result
}

func newPyUDFTestChunked(dataType arrow.DataType, chunks ...arrow.Array) *arrow.Chunked {
	return arrow.NewChunked(dataType, chunks)
}
