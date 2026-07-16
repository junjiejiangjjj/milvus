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

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "common/arrow_c_data_c.h"
#include "pyudf/pyudf.h"

namespace {

struct ReleaseCounter {
    int arrays = 0;
    int schemas = 0;
};

void
ReleaseArray(ArrowArray* array) {
    auto counter = static_cast<ReleaseCounter*>(array->private_data);
    ++counter->arrays;
    array->release = nullptr;
}

void
ReleaseSchema(ArrowSchema* schema) {
    auto counter = static_cast<ReleaseCounter*>(schema->private_data);
    ++counter->schemas;
    schema->release = nullptr;
}

void
Populate(ArrowArray* array, ArrowSchema* schema, ReleaseCounter* counter) {
    array->release = ReleaseArray;
    array->private_data = counter;
    schema->release = ReleaseSchema;
    schema->private_data = counter;
}

void
PopulateDescriptor(ArrowArray* array,
                   ArrowSchema* schema,
                   ReleaseCounter* counter,
                   int64_t array_length,
                   int64_t array_offset,
                   const char* schema_format) {
    Populate(array, schema, counter);
    array->length = array_length;
    array->offset = array_offset;
    schema->format = schema_format;
}

void
FreeStatus(CStatus* status) {
    if (status->error_code != 0 && status->error_msg != nullptr) {
        free(const_cast<char*>(status->error_msg));
    }
}

CPyUDFInvocation
NewInvocation(int32_t num_inputs,
              int32_t num_chunks,
              const int64_t* chunk_sizes) {
    CPyUDFInvocation invocation = nullptr;
    auto status =
        NewPyUDFInvocation(num_inputs, num_chunks, chunk_sizes, &invocation);
    EXPECT_EQ(status.error_code, 0) << status.error_msg;
    EXPECT_NE(invocation, nullptr);
    FreeStatus(&status);
    return invocation;
}

TEST(PyUDFHandlesCTest, InvocationCopiesMetadataAndKeepsStableSlots) {
    int64_t chunk_sizes[] = {2, 0, 5};
    auto invocation = NewInvocation(2, 3, chunk_sizes);
    chunk_sizes[0] = 99;

    EXPECT_EQ(PyUDFInvocationNumInputs(invocation), 2);
    EXPECT_EQ(PyUDFInvocationNumChunks(invocation), 3);
    EXPECT_EQ(PyUDFInvocationChunkSize(invocation, 0), 2);
    EXPECT_EQ(PyUDFInvocationChunkSize(invocation, 1), 0);
    EXPECT_EQ(PyUDFInvocationChunkSize(invocation, 2), 5);

    auto array = PyUDFInvocationInputArray(invocation, 1, 2);
    auto schema = PyUDFInvocationInputSchema(invocation, 1, 2);
    ASSERT_NE(array, nullptr);
    ASSERT_NE(schema, nullptr);
    EXPECT_EQ(array->release, nullptr);
    EXPECT_EQ(schema->release, nullptr);
    EXPECT_EQ(PyUDFInvocationInputArray(invocation, 1, 2), array);
    EXPECT_EQ(PyUDFInvocationInputSchema(invocation, 1, 2), schema);

    EXPECT_EQ(PyUDFInvocationInputArray(invocation, -1, 0), nullptr);
    EXPECT_EQ(PyUDFInvocationInputArray(invocation, 2, 0), nullptr);
    EXPECT_EQ(PyUDFInvocationInputSchema(invocation, 0, 3), nullptr);
    EXPECT_EQ(PyUDFInvocationChunkSize(invocation, 3), -1);

    DeletePyUDFInvocation(invocation);
}

TEST(PyUDFHandlesCTest, InvocationAllowsZeroChunks) {
    auto invocation = NewInvocation(3, 0, nullptr);
    EXPECT_EQ(PyUDFInvocationNumInputs(invocation), 3);
    EXPECT_EQ(PyUDFInvocationNumChunks(invocation), 0);
    EXPECT_EQ(PyUDFInvocationInputArray(invocation, 0, 0), nullptr);
    DeletePyUDFInvocation(invocation);
}

TEST(PyUDFHandlesCTest, InvocationRejectsInvalidArguments) {
    CPyUDFInvocation invocation = reinterpret_cast<CPyUDFInvocation>(0x1);
    auto status = NewPyUDFInvocation(-1, 1, nullptr, &invocation);
    EXPECT_NE(status.error_code, 0);
    EXPECT_EQ(invocation, nullptr);
    FreeStatus(&status);

    status = NewPyUDFInvocation(1, 1, nullptr, &invocation);
    EXPECT_NE(status.error_code, 0);
    EXPECT_EQ(invocation, nullptr);
    FreeStatus(&status);

    int64_t negative_size[] = {-1};
    status = NewPyUDFInvocation(1, 1, negative_size, &invocation);
    EXPECT_NE(status.error_code, 0);
    EXPECT_EQ(invocation, nullptr);
    FreeStatus(&status);

    int64_t chunk_size[] = {1};
    status = NewPyUDFInvocation(1, 1, chunk_size, nullptr);
    EXPECT_NE(status.error_code, 0);
    FreeStatus(&status);
}

TEST(PyUDFHandlesCTest, InvocationDeletesPopulatedSlotsExactlyOnce) {
    int64_t chunk_sizes[] = {1, 1};
    auto invocation = NewInvocation(2, 2, chunk_sizes);
    ReleaseCounter counter;
    Populate(PyUDFInvocationInputArray(invocation, 0, 0),
             PyUDFInvocationInputSchema(invocation, 0, 0),
             &counter);
    Populate(PyUDFInvocationInputArray(invocation, 1, 1),
             PyUDFInvocationInputSchema(invocation, 1, 1),
             &counter);

    DeletePyUDFInvocation(invocation);
    EXPECT_EQ(counter.arrays, 2);
    EXPECT_EQ(counter.schemas, 2);
}

TEST(PyUDFHandlesCTest, IdentityMovesDescriptorsAndReleasesOnce) {
    int64_t chunk_sizes[] = {2, 0};
    auto invocation = NewInvocation(2, 2, chunk_sizes);
    ReleaseCounter counter;

    for (int32_t input = 0; input < 2; ++input) {
        for (int32_t chunk = 0; chunk < 2; ++chunk) {
            auto array = PyUDFInvocationInputArray(invocation, input, chunk);
            auto schema = PyUDFInvocationInputSchema(invocation, input, chunk);
            PopulateDescriptor(array,
                               schema,
                               &counter,
                               10 * input + chunk,
                               input + chunk,
                               input == 0 ? "l" : "u");
        }
    }

    CPyUDFResult result = nullptr;
    auto status = RunPyUDFIdentity(invocation, &result);
    ASSERT_EQ(status.error_code, 0) << status.error_msg;
    ASSERT_NE(result, nullptr);
    FreeStatus(&status);

    EXPECT_EQ(PyUDFResultNumOutputs(result), 2);
    for (int32_t output = 0; output < 2; ++output) {
        EXPECT_EQ(PyUDFResultNumChunks(result, output), 2);
        for (int32_t chunk = 0; chunk < 2; ++chunk) {
            auto input_array =
                PyUDFInvocationInputArray(invocation, output, chunk);
            auto input_schema =
                PyUDFInvocationInputSchema(invocation, output, chunk);
            EXPECT_EQ(input_array->release, nullptr);
            EXPECT_EQ(input_schema->release, nullptr);

            auto output_array = PyUDFResultArray(result, output, chunk);
            auto output_schema = PyUDFResultSchema(result, output, chunk);
            ASSERT_NE(output_array, nullptr);
            ASSERT_NE(output_schema, nullptr);
            EXPECT_EQ(output_array->length, 10 * output + chunk);
            EXPECT_EQ(output_array->offset, output + chunk);
            EXPECT_STREQ(output_schema->format, output == 0 ? "l" : "u");
            EXPECT_NE(output_array->release, nullptr);
            EXPECT_NE(output_schema->release, nullptr);
        }
    }

    DeletePyUDFInvocation(invocation);
    EXPECT_EQ(counter.arrays, 0);
    EXPECT_EQ(counter.schemas, 0);
    DeletePyUDFResult(result);
    EXPECT_EQ(counter.arrays, 4);
    EXPECT_EQ(counter.schemas, 4);
}

TEST(PyUDFHandlesCTest, IdentityRejectsUnpopulatedSlotsAtomically) {
    int64_t chunk_sizes[] = {1, 1};
    auto invocation = NewInvocation(1, 2, chunk_sizes);
    ReleaseCounter counter;
    Populate(PyUDFInvocationInputArray(invocation, 0, 0),
             PyUDFInvocationInputSchema(invocation, 0, 0),
             &counter);

    CPyUDFResult result = reinterpret_cast<CPyUDFResult>(0x1);
    auto status = RunPyUDFIdentity(invocation, &result);
    EXPECT_NE(status.error_code, 0);
    EXPECT_EQ(result, nullptr);
    FreeStatus(&status);

    auto populated_array = PyUDFInvocationInputArray(invocation, 0, 0);
    auto populated_schema = PyUDFInvocationInputSchema(invocation, 0, 0);
    EXPECT_NE(populated_array->release, nullptr);
    EXPECT_NE(populated_schema->release, nullptr);
    EXPECT_EQ(PyUDFInvocationInputArray(invocation, 0, 1)->release, nullptr);
    EXPECT_EQ(PyUDFInvocationInputSchema(invocation, 0, 1)->release, nullptr);

    DeletePyUDFInvocation(invocation);
    EXPECT_EQ(counter.arrays, 1);
    EXPECT_EQ(counter.schemas, 1);
}

TEST(PyUDFHandlesCTest, IdentityRejectsNullArguments) {
    CPyUDFResult result = reinterpret_cast<CPyUDFResult>(0x1);
    auto status = RunPyUDFIdentity(nullptr, &result);
    EXPECT_NE(status.error_code, 0);
    EXPECT_EQ(result, nullptr);
    FreeStatus(&status);

    int64_t chunk_sizes[] = {1};
    auto invocation = NewInvocation(1, 1, chunk_sizes);
    status = RunPyUDFIdentity(invocation, nullptr);
    EXPECT_NE(status.error_code, 0);
    FreeStatus(&status);
    DeletePyUDFInvocation(invocation);
}

TEST(PyUDFHandlesCTest, ResultExposesMetadataAndBounds) {
    int32_t chunk_counts[] = {2, 0, 1};
    auto result = static_cast<CPyUDFResult>(
        new milvus::pyudf::PyUDFResult(3, chunk_counts));

    EXPECT_EQ(PyUDFResultNumOutputs(result), 3);
    EXPECT_EQ(PyUDFResultNumChunks(result, 0), 2);
    EXPECT_EQ(PyUDFResultNumChunks(result, 1), 0);
    EXPECT_EQ(PyUDFResultNumChunks(result, 2), 1);
    EXPECT_EQ(PyUDFResultNumChunks(result, 3), -1);
    EXPECT_NE(PyUDFResultArray(result, 0, 0), nullptr);
    EXPECT_NE(PyUDFResultSchema(result, 0, 1), nullptr);
    EXPECT_EQ(PyUDFResultArray(result, 1, 0), nullptr);
    EXPECT_EQ(PyUDFResultSchema(result, 2, 1), nullptr);

    DeletePyUDFResult(result);
}

TEST(PyUDFHandlesCTest, ResultDeletesOnlyUnconsumedSlots) {
    int32_t chunk_counts[] = {2, 1};
    auto result = static_cast<CPyUDFResult>(
        new milvus::pyudf::PyUDFResult(2, chunk_counts));
    ReleaseCounter counter;
    for (int32_t output = 0; output < 2; ++output) {
        for (int32_t chunk = 0; chunk < PyUDFResultNumChunks(result, output);
             ++chunk) {
            Populate(PyUDFResultArray(result, output, chunk),
                     PyUDFResultSchema(result, output, chunk),
                     &counter);
        }
    }

    auto consumed_array = PyUDFResultArray(result, 0, 1);
    auto consumed_schema = PyUDFResultSchema(result, 1, 0);
    consumed_array->release = nullptr;
    consumed_schema->release = nullptr;

    DeletePyUDFResult(result);
    EXPECT_EQ(counter.arrays, 2);
    EXPECT_EQ(counter.schemas, 2);
}

TEST(PyUDFHandlesCTest, NullHandlesAreSafe) {
    EXPECT_EQ(PyUDFInvocationNumInputs(nullptr), -1);
    EXPECT_EQ(PyUDFInvocationNumChunks(nullptr), -1);
    EXPECT_EQ(PyUDFInvocationChunkSize(nullptr, 0), -1);
    EXPECT_EQ(PyUDFInvocationInputArray(nullptr, 0, 0), nullptr);
    EXPECT_EQ(PyUDFInvocationInputSchema(nullptr, 0, 0), nullptr);
    DeletePyUDFInvocation(nullptr);

    EXPECT_EQ(PyUDFResultNumOutputs(nullptr), -1);
    EXPECT_EQ(PyUDFResultNumChunks(nullptr, 0), -1);
    EXPECT_EQ(PyUDFResultArray(nullptr, 0, 0), nullptr);
    EXPECT_EQ(PyUDFResultSchema(nullptr, 0, 0), nullptr);
    DeletePyUDFResult(nullptr);
}

}  // namespace
