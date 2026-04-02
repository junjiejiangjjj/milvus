//go:build test
// +build test

package segcore

/*
#cgo pkg-config: milvus_core

#include <stdlib.h>
#include <stdint.h>

struct ArrowSchema;
struct ArrowArray;

struct ArrowArrayStream {
    int (*get_schema)(struct ArrowArrayStream*, struct ArrowSchema*);
    int (*get_next)(struct ArrowArrayStream*, struct ArrowArray*);
    const char* (*get_last_error)(struct ArrowArrayStream*);
    void (*release)(struct ArrowArrayStream*);
    void* private_data;
};

int
MilvusGoTestArrowStreamGetSchemaError(struct ArrowArrayStream* stream, struct ArrowSchema* out) {
    return 5;
}

int
MilvusGoTestArrowStreamGetNextError(struct ArrowArrayStream* stream, struct ArrowArray* out) {
    return 5;
}

const char*
MilvusGoTestArrowStreamGetLastError(struct ArrowArrayStream* stream) {
    return (const char*)stream->private_data;
}

void
MilvusGoTestArrowStreamRelease(struct ArrowArrayStream* stream) {
    if (stream == NULL || stream->release == NULL) {
        return;
    }
    if (stream->private_data != NULL) {
        free(stream->private_data);
        stream->private_data = NULL;
    }
    stream->release = NULL;
}

void
MilvusGoTestArrowStreamReleaseAndFree(struct ArrowArrayStream* stream) {
    if (stream == NULL) {
        return;
    }
    if (stream->release != NULL) {
        stream->release(stream);
    }
    free(stream);
}
*/
import "C"

import (
	"unsafe"

	"github.com/apache/arrow/go/v17/arrow"
	"github.com/apache/arrow/go/v17/arrow/array"
	"github.com/apache/arrow/go/v17/arrow/cdata"

	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

func newTestCRecordBatchReader(schema *arrow.Schema, records []arrow.Record) (*cRecordBatchReader, error) {
	recordReader, err := array.NewRecordReader(schema, records)
	if err != nil {
		return nil, err
	}

	cStream := (*C.struct_ArrowArrayStream)(C.calloc(1, C.size_t(unsafe.Sizeof(C.struct_ArrowArrayStream{}))))
	if cStream == nil {
		recordReader.Release()
		return nil, merr.WrapErrServiceInternal("failed to allocate ArrowArrayStream")
	}

	cdata.ExportRecordReader(recordReader, (*cdata.CArrowArrayStream)(unsafe.Pointer(cStream)))
	return newCRecordBatchReader(cStream)
}

func newTestSchemaErrorCRecordBatchReader(lastError string) (*cRecordBatchReader, error) {
	return newCRecordBatchReader(newTestErrorArrowStream(true, false, lastError))
}

func replaceTestReaderStreamWithNextError(r *cRecordBatchReader, lastError string) {
	if r.stream != nil {
		C.MilvusGoTestArrowStreamReleaseAndFree(r.stream)
	}
	r.stream = newTestErrorArrowStream(false, true, lastError)
}

func newTestErrorArrowStream(schemaErr bool, nextErr bool, lastError string) *C.struct_ArrowArrayStream {
	cStream := (*C.struct_ArrowArrayStream)(C.calloc(1, C.size_t(unsafe.Sizeof(C.struct_ArrowArrayStream{}))))
	if cStream == nil {
		return nil
	}
	if schemaErr {
		cStream.get_schema = (*[0]byte)(C.MilvusGoTestArrowStreamGetSchemaError)
	}
	if nextErr {
		cStream.get_next = (*[0]byte)(C.MilvusGoTestArrowStreamGetNextError)
	}
	cStream.get_last_error = (*[0]byte)(C.MilvusGoTestArrowStreamGetLastError)
	cStream.release = (*[0]byte)(C.MilvusGoTestArrowStreamRelease)
	if lastError != "" {
		cStream.private_data = unsafe.Pointer(C.CString(lastError))
	}
	return cStream
}
