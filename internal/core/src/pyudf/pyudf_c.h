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

#pragma once

#include <stdint.h>

#include "common/type_c.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void* CPyUDFInvocation;
typedef void* CPyUDFResult;

struct ArrowArray;
struct ArrowSchema;

CStatus
NewPyUDFInvocation(int32_t num_inputs,
                   int32_t num_chunks,
                   const int64_t* chunk_sizes,
                   CPyUDFInvocation* invocation);

int32_t
PyUDFInvocationNumInputs(CPyUDFInvocation invocation);

int32_t
PyUDFInvocationNumChunks(CPyUDFInvocation invocation);

int64_t
PyUDFInvocationChunkSize(CPyUDFInvocation invocation, int32_t chunk_index);

struct ArrowArray*
PyUDFInvocationInputArray(CPyUDFInvocation invocation,
                          int32_t input_index,
                          int32_t chunk_index);

struct ArrowSchema*
PyUDFInvocationInputSchema(CPyUDFInvocation invocation,
                           int32_t input_index,
                           int32_t chunk_index);

void
DeletePyUDFInvocation(CPyUDFInvocation invocation);

CStatus
RunPyUDFIdentity(CPyUDFInvocation invocation, CPyUDFResult* result);

int32_t
PyUDFResultNumOutputs(CPyUDFResult result);

int32_t
PyUDFResultNumChunks(CPyUDFResult result, int32_t output_index);

struct ArrowArray*
PyUDFResultArray(CPyUDFResult result,
                 int32_t output_index,
                 int32_t chunk_index);

struct ArrowSchema*
PyUDFResultSchema(CPyUDFResult result,
                  int32_t output_index,
                  int32_t chunk_index);

void
DeletePyUDFResult(CPyUDFResult result);

#ifdef __cplusplus
}
#endif
