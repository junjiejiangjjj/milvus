// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "pyudf/pyudf_c.h"

#include <memory>

#include "common/EasyAssert.h"
#include "pyudf/pyudf.h"

CStatus
NewPyUDFInvocation(int32_t num_inputs,
                   int32_t num_chunks,
                   const int64_t* chunk_sizes,
                   CPyUDFInvocation* invocation) {
    if (invocation == nullptr) {
        return milvus::FailureCStatus(
            milvus::UnexpectedError,
            "py_udf: invocation output pointer is nil");
    }
    *invocation = nullptr;
    try {
        auto value = std::make_unique<milvus::pyudf::PyUDFInvocation>(
            num_inputs, num_chunks, chunk_sizes);
        *invocation = static_cast<CPyUDFInvocation>(value.release());
        return milvus::SuccessCStatus();
    } catch (std::exception& e) {
        return milvus::FailureCStatus(&e);
    }
}

int32_t
PyUDFInvocationNumInputs(CPyUDFInvocation invocation) {
    auto value = static_cast<milvus::pyudf::PyUDFInvocation*>(invocation);
    return value == nullptr ? -1 : value->num_inputs();
}

int32_t
PyUDFInvocationNumChunks(CPyUDFInvocation invocation) {
    auto value = static_cast<milvus::pyudf::PyUDFInvocation*>(invocation);
    return value == nullptr ? -1 : value->num_chunks();
}

int64_t
PyUDFInvocationChunkSize(CPyUDFInvocation invocation, int32_t chunk_index) {
    auto value = static_cast<milvus::pyudf::PyUDFInvocation*>(invocation);
    return value == nullptr ? -1 : value->chunk_size(chunk_index);
}

ArrowArray*
PyUDFInvocationInputArray(CPyUDFInvocation invocation,
                          int32_t input_index,
                          int32_t chunk_index) {
    auto value = static_cast<milvus::pyudf::PyUDFInvocation*>(invocation);
    return value == nullptr ? nullptr
                            : value->input_array(input_index, chunk_index);
}

ArrowSchema*
PyUDFInvocationInputSchema(CPyUDFInvocation invocation,
                           int32_t input_index,
                           int32_t chunk_index) {
    auto value = static_cast<milvus::pyudf::PyUDFInvocation*>(invocation);
    return value == nullptr ? nullptr
                            : value->input_schema(input_index, chunk_index);
}

void
DeletePyUDFInvocation(CPyUDFInvocation invocation) {
    delete static_cast<milvus::pyudf::PyUDFInvocation*>(invocation);
}

CStatus
RunPyUDFIdentity(CPyUDFInvocation invocation, CPyUDFResult* result) {
    if (result == nullptr) {
        return milvus::FailureCStatus(milvus::UnexpectedError,
                                      "py_udf: result output pointer is nil");
    }
    *result = nullptr;

    auto value = static_cast<milvus::pyudf::PyUDFInvocation*>(invocation);
    if (value == nullptr) {
        return milvus::FailureCStatus(milvus::UnexpectedError,
                                      "py_udf: invocation is nil");
    }

    try {
        auto identity_result = value->RunIdentity();
        *result = static_cast<CPyUDFResult>(identity_result.release());
        return milvus::SuccessCStatus();
    } catch (const std::exception& e) {
        return milvus::FailureCStatus(&e);
    } catch (...) {
        return milvus::FailureCStatus(milvus::UnexpectedError,
                                      "py_udf: unknown native exception");
    }
}

int32_t
PyUDFResultNumOutputs(CPyUDFResult result) {
    auto value = static_cast<milvus::pyudf::PyUDFResult*>(result);
    return value == nullptr ? -1 : value->num_outputs();
}

int32_t
PyUDFResultNumChunks(CPyUDFResult result, int32_t output_index) {
    auto value = static_cast<milvus::pyudf::PyUDFResult*>(result);
    return value == nullptr ? -1 : value->num_chunks(output_index);
}

ArrowArray*
PyUDFResultArray(CPyUDFResult result,
                 int32_t output_index,
                 int32_t chunk_index) {
    auto value = static_cast<milvus::pyudf::PyUDFResult*>(result);
    return value == nullptr ? nullptr
                            : value->output_array(output_index, chunk_index);
}

ArrowSchema*
PyUDFResultSchema(CPyUDFResult result,
                  int32_t output_index,
                  int32_t chunk_index) {
    auto value = static_cast<milvus::pyudf::PyUDFResult*>(result);
    return value == nullptr ? nullptr
                            : value->output_schema(output_index, chunk_index);
}

void
DeletePyUDFResult(CPyUDFResult result) {
    delete static_cast<milvus::pyudf::PyUDFResult*>(result);
}
