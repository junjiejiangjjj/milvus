package segcore

import (
	"context"
	"io"
	"testing"

	"github.com/apache/arrow/go/v17/arrow"
	"github.com/apache/arrow/go/v17/arrow/array"
	"github.com/apache/arrow/go/v17/arrow/memory"
	"github.com/cockroachdb/errors"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

func TestExportSearchResultAsArrowStreamValidation(t *testing.T) {
	reader, err := ExportSearchResultAsArrowStream(context.Background(), nil, NewDummySearchPlanForTest(t), nil)
	require.Error(t, err)
	assert.Nil(t, reader)
	assert.ErrorIs(t, err, merr.ErrParameterInvalid)
	assert.Contains(t, err.Error(), "nil search result")

	reader, err = ExportSearchResultAsArrowStream(context.Background(), &SearchResult{}, nil, nil)
	require.Error(t, err)
	assert.Nil(t, reader)
	assert.ErrorIs(t, err, merr.ErrParameterInvalid)
	assert.Contains(t, err.Error(), "nil search plan")
}

func TestFillOutputFieldsOrderedValidation(t *testing.T) {
	blob, err := FillOutputFieldsOrdered(context.Background(), []*SearchResult{{}}, nil, nil, nil)
	require.Error(t, err)
	assert.Nil(t, blob)
	assert.ErrorIs(t, err, merr.ErrParameterInvalid)
	assert.Contains(t, err.Error(), "nil search plan")

	plan := NewDummySearchPlanForTest(t)
	blob, err = FillOutputFieldsOrdered(context.Background(), nil, plan, nil, nil)
	require.Error(t, err)
	assert.Nil(t, blob)
	assert.ErrorIs(t, err, merr.ErrParameterInvalid)
	assert.Contains(t, err.Error(), "empty search results")

	blob, err = FillOutputFieldsOrdered(context.Background(), []*SearchResult{{}}, plan, []int32{0}, nil)
	require.Error(t, err)
	assert.Nil(t, blob)
	assert.ErrorIs(t, err, merr.ErrParameterInvalid)
	assert.Contains(t, err.Error(), "unaligned segment indices")

	blob, err = FillOutputFieldsOrdered(context.Background(), []*SearchResult{nil}, plan, nil, nil)
	require.Error(t, err)
	assert.Nil(t, blob)
	assert.ErrorIs(t, err, merr.ErrParameterInvalid)
	assert.Contains(t, err.Error(), "nil search result at index 0")
}

func TestCRecordBatchReaderReadEOFAndAccessors(t *testing.T) {
	pool := memory.NewGoAllocator()
	schema := arrow.NewSchema([]arrow.Field{{Name: "id", Type: arrow.PrimitiveTypes.Int64}}, nil)

	idBuilder := array.NewInt64Builder(pool)
	idBuilder.AppendValues([]int64{10, 20}, nil)
	ids := idBuilder.NewArray()
	idBuilder.Release()
	defer ids.Release()

	record := array.NewRecord(schema, []arrow.Array{ids}, int64(ids.Len()))
	defer record.Release()

	reader, err := newTestCRecordBatchReader(schema, []arrow.Record{record})
	require.NoError(t, err)
	defer reader.Release()

	assert.True(t, schema.Equal(reader.Schema()))
	assert.Nil(t, reader.Record())
	assert.NoError(t, reader.Err())

	rec, err := reader.Read()
	require.NoError(t, err)
	require.NotNil(t, rec)
	assert.Same(t, rec, reader.Record())
	assert.Equal(t, int64(2), rec.NumRows())
	gotIDs := rec.Column(0).(*array.Int64)
	assert.Equal(t, []int64{10, 20}, []int64{gotIDs.Value(0), gotIDs.Value(1)})

	rec, err = reader.Read()
	require.ErrorIs(t, err, io.EOF)
	assert.Nil(t, rec)
	assert.NoError(t, reader.Err(), "EOF should not be reported as reader error")
	assert.Nil(t, reader.Record(), "previous record should be released before EOF is returned")

	reader.Release()
	assert.Nil(t, reader.stream)
	assert.Nil(t, reader.arr)
	reader.Release()
}

func TestCRecordBatchReaderReadReleasesPreviousRecord(t *testing.T) {
	pool := memory.NewGoAllocator()
	schema := arrow.NewSchema([]arrow.Field{{Name: "score", Type: arrow.PrimitiveTypes.Float32}}, nil)

	firstValues := newFloat32Array(t, pool, []float32{1.0})
	defer firstValues.Release()
	secondValues := newFloat32Array(t, pool, []float32{2.0, 3.0})
	defer secondValues.Release()

	first := array.NewRecord(schema, []arrow.Array{firstValues}, int64(firstValues.Len()))
	defer first.Release()
	second := array.NewRecord(schema, []arrow.Array{secondValues}, int64(secondValues.Len()))
	defer second.Release()

	reader, err := newTestCRecordBatchReader(schema, []arrow.Record{first, second})
	require.NoError(t, err)
	defer reader.Release()

	rec1, err := reader.Read()
	require.NoError(t, err)
	assert.Equal(t, int64(1), rec1.NumRows())

	rec2, err := reader.Read()
	require.NoError(t, err)
	assert.Equal(t, int64(2), rec2.NumRows())
	assert.Same(t, rec2, reader.Record())
	assert.NotSame(t, rec1, rec2)

	_, err = reader.Read()
	require.ErrorIs(t, err, io.EOF)
}

func TestCRecordBatchReaderSchemaAndNextErrors(t *testing.T) {
	reader, err := newTestSchemaErrorCRecordBatchReader("schema failed")
	require.NoError(t, err)
	assert.Nil(t, reader.Schema())
	require.Error(t, reader.Err())
	assert.Contains(t, reader.Err().Error(), "input/output error: schema failed")
	reader.Release()

	reader, err = newTestSchemaErrorCRecordBatchReader("")
	require.NoError(t, err)
	assert.Nil(t, reader.Schema())
	require.Error(t, reader.Err())
	assert.Contains(t, reader.Err().Error(), "input/output error")
	reader.Release()

	pool := memory.NewGoAllocator()
	schema := arrow.NewSchema([]arrow.Field{{Name: "id", Type: arrow.PrimitiveTypes.Int64}}, nil)

	idBuilder := array.NewInt64Builder(pool)
	idBuilder.AppendValues([]int64{1}, nil)
	ids := idBuilder.NewArray()
	idBuilder.Release()
	defer ids.Release()

	record := array.NewRecord(schema, []arrow.Array{ids}, int64(ids.Len()))
	defer record.Release()

	reader, err = newTestCRecordBatchReader(schema, []arrow.Record{record})
	require.NoError(t, err)
	defer reader.Release()
	require.NotNil(t, reader.Schema())

	replaceTestReaderStreamWithNextError(reader, "next failed")
	rec, err := reader.Read()
	require.Error(t, err)
	assert.Nil(t, rec)
	assert.Contains(t, err.Error(), "input/output error: next failed")
	assert.Same(t, err, reader.Err())
}

func TestCRecordBatchReaderErrAndNilRelease(t *testing.T) {
	var nilReader *cRecordBatchReader
	require.NotPanics(t, func() {
		nilReader.Release()
	})

	reader := &cRecordBatchReader{err: io.EOF}
	assert.NoError(t, reader.Err())

	wantErr := errors.New("stream failed")
	reader.err = wantErr
	assert.ErrorIs(t, reader.Err(), wantErr)
}

func newFloat32Array(t *testing.T, pool memory.Allocator, values []float32) arrow.Array {
	t.Helper()

	builder := array.NewFloat32Builder(pool)
	builder.AppendValues(values, nil)
	arr := builder.NewArray()
	builder.Release()
	return arr
}
