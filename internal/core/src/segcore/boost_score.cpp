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

#include "segcore/boost_score_c.h"

#include <arrow/array/array_primitive.h>
#include <arrow/c/bridge.h>

#include <limits>
#include <memory>

#include "common/Common.h"
#include "common/Consts.h"
#include "common/EasyAssert.h"
#include "common/OpContext.h"
#include "common/Types.h"
#include "exec/QueryContext.h"
#include "pb/plan.pb.h"
#include "query/PlanImpl.h"
#include "query/PlanProto.h"
#include "rescores/BoostScoreRunner.h"
#include "segcore/SegmentInterface.h"

namespace {

milvus::FixedVector<int32_t>
BuildScorerOffsets(const std::shared_ptr<arrow::Array>& offsets) {
    AssertInfo(offsets != nullptr, "offset array is null");
    AssertInfo(offsets->type_id() == arrow::Type::INT64,
               "offset array must be Int64, got {}",
               offsets->type()->ToString());
    AssertInfo(offsets->null_count() == 0, "offset array contains null");

    auto int64_offsets = std::static_pointer_cast<arrow::Int64Array>(offsets);
    auto count = int64_offsets->length();
    milvus::FixedVector<int32_t> scorer_offsets;
    scorer_offsets.reserve(count);
    for (auto i = 0; i < count; ++i) {
        auto offset = int64_offsets->Value(i);
        AssertInfo(offset >= 0,
                   "offset must be non-negative, offset: {}",
                   offset);
        AssertInfo(offset <= std::numeric_limits<int32_t>::max(),
                   "offset exceeds int32 range, offset: {}",
                   offset);
        scorer_offsets.push_back(static_cast<int32_t>(offset));
    }
    return scorer_offsets;
}

}  // namespace

CStatus
ComputeScorerScoresOnOffsetChunks(CSegmentInterface c_segment,
                                  CSearchPlan c_plan,
                                  const void* serialized_score_function,
                                  int64_t serialized_score_function_size,
                                  ArrowArray* offset_chunks,
                                  ArrowSchema* offset_schemas,
                                  int64_t num_chunks,
                                  uint64_t timestamp,
                                  uint64_t collection_ttl,
                                  int32_t consistency_level,
                                  uint64_t entity_ttl_physical_time_us,
                                  float* const* output_score_chunks,
                                  bool* const* output_has_score_chunks) {
    try {
        AssertInfo(c_segment != nullptr, "segment is null");
        AssertInfo(c_plan != nullptr, "search plan is null");
        AssertInfo(serialized_score_function != nullptr,
                   "serialized score function is null");
        AssertInfo(serialized_score_function_size > 0,
                   "serialized score function is empty");
        AssertInfo(num_chunks >= 0, "chunk count must be non-negative");
        if (num_chunks > 0) {
            AssertInfo(offset_chunks != nullptr, "offset chunks is null");
            AssertInfo(offset_schemas != nullptr, "offset schemas is null");
            AssertInfo(output_score_chunks != nullptr,
                       "output score chunks is null");
            AssertInfo(output_has_score_chunks != nullptr,
                       "output has score chunks is null");
        }

        auto segment =
            static_cast<milvus::segcore::SegmentInternalInterface*>(c_segment);
        auto plan = static_cast<milvus::query::Plan*>(c_plan);

        milvus::proto::plan::ScoreFunction score_function;
        auto ok = score_function.ParseFromArray(serialized_score_function,
                                                serialized_score_function_size);
        AssertInfo(ok, "failed to parse score function");

        milvus::query::ProtoParser parser(plan->schema_);
        auto scorer = parser.ParseScorer(score_function);

        auto active_count = segment->get_active_count(timestamp);
        auto query_context = std::make_shared<milvus::exec::QueryContext>(
            DEAFULT_QUERY_ID,
            segment,
            active_count,
            timestamp,
            collection_ttl,
            consistency_level,
            plan->plan_node_->plan_options_,
            std::make_shared<milvus::exec::QueryConfig>(),
            nullptr,
            std::unordered_map<std::string,
                               std::shared_ptr<milvus::exec::BaseConfig>>(),
            entity_ttl_physical_time_us);
        query_context->set_search_info(plan->plan_node_->search_info_);

        auto op_context = milvus::OpContext();
        query_context->set_op_context(&op_context);
        auto exec_context = milvus::exec::ExecContext(query_context.get());

        for (auto chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
            auto offset_array_result =
                arrow::ImportArray(&offset_chunks[chunk_idx],
                                   &offset_schemas[chunk_idx]);
            AssertInfo(offset_array_result.ok(),
                       "failed to import offset chunk {}: {}",
                       chunk_idx,
                       offset_array_result.status().ToString());
            auto offset_array = offset_array_result.ValueOrDie();
            if (offset_array->length() == 0) {
                continue;
            }
            AssertInfo(output_score_chunks[chunk_idx] != nullptr,
                       "output score chunk {} is null",
                       chunk_idx);
            AssertInfo(output_has_score_chunks[chunk_idx] != nullptr,
                       "output has score chunk {} is null",
                       chunk_idx);

            auto scorer_offsets = BuildScorerOffsets(offset_array);
            milvus::rescores::ComputeScorerScores(
                &exec_context,
                &op_context,
                segment,
                scorer,
                scorer_offsets,
                output_score_chunks[chunk_idx],
                output_has_score_chunks[chunk_idx]);
        }
        return milvus::SuccessCStatus();
    } catch (std::exception& e) {
        return milvus::FailureCStatus(&e);
    }
}
