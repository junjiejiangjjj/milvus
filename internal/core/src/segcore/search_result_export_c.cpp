// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#include "segcore/search_result_export_c.h"

#include <arrow/api.h>
#include <arrow/c/bridge.h>
#include <arrow/c/abi.h>

#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "common/EasyAssert.h"
#include "log/Log.h"
#include "common/QueryResult.h"
#include "common/Types.h"
#include "monitor/scope_metric.h"
#include "query/PlanImpl.h"
#include "segcore/SegmentInterface.h"
#include "segcore/Utils.h"

using SearchResult = milvus::SearchResult;

namespace {

// FilterInvalidSearchResult compacts valid rows (offset != -1) in place,
// and builds topk_per_nq (per-NQ valid row counts).
// Returns the per-NQ valid counts.
std::vector<int64_t>
FilterAndGetTopkPerNQ(SearchResult* search_result) {
    auto nq = search_result->total_nq_;
    auto topK = search_result->unity_topK_;
    auto& offsets = search_result->seg_offsets_;
    auto& distances = search_result->distances_;

    std::vector<int64_t> topk_per_nq(nq, 0);
    uint32_t valid_index = 0;

    for (int64_t i = 0; i < nq; ++i) {
        for (int64_t j = 0; j < topK; ++j) {
            auto index = i * topK + j;
            if (offsets[index] != INVALID_SEG_OFFSET) {
                topk_per_nq[i]++;
                offsets[valid_index] = offsets[index];
                distances[valid_index] = distances[index];
                valid_index++;
            }
        }
    }
    offsets.resize(valid_index);
    distances.resize(valid_index);

    // Also build the prefix sum that FillPrimaryKeys expects
    search_result->topk_per_nq_prefix_sum_.resize(nq + 1);
    search_result->topk_per_nq_prefix_sum_[0] = 0;
    for (int64_t i = 0; i < nq; ++i) {
        search_result->topk_per_nq_prefix_sum_[i + 1] =
            search_result->topk_per_nq_prefix_sum_[i] + topk_per_nq[i];
    }

    return topk_per_nq;
}

// Build the topk_per_nq metadata string: "3,2,5" means NQ=0 has 3 rows, etc.
std::string
TopkPerNQToString(const std::vector<int64_t>& topk_per_nq) {
    std::ostringstream oss;
    for (size_t i = 0; i < topk_per_nq.size(); ++i) {
        if (i > 0)
            oss << ",";
        oss << topk_per_nq[i];
    }
    return oss.str();
}

// Build an empty RecordBatch (0 rows) for empty search results.
arrow::Result<std::shared_ptr<arrow::RecordBatch>>
BuildEmptyBatch() {
    arrow::Int64Builder id_builder;
    arrow::FloatBuilder score_builder;
    arrow::Int64Builder offset_builder;

    std::shared_ptr<arrow::Array> id_arr, score_arr, offset_arr;
    ARROW_RETURN_NOT_OK(id_builder.Finish(&id_arr));
    ARROW_RETURN_NOT_OK(score_builder.Finish(&score_arr));
    ARROW_RETURN_NOT_OK(offset_builder.Finish(&offset_arr));

    auto schema = arrow::schema(
        {arrow::field("$id", arrow::int64()),
         arrow::field("$score", arrow::float32()),
         arrow::field("$seg_offset", arrow::int64())});
    auto metadata =
        arrow::KeyValueMetadata::Make({"topk_per_nq"}, {""});
    schema = schema->WithMetadata(metadata);
    return arrow::RecordBatch::Make(
        schema, 0, {id_arr, score_arr, offset_arr});
}

// Convert a protobuf FieldData (scalar) to an Arrow Array + Field.
// Returns nullopt for unsupported types.
arrow::Result<std::pair<std::shared_ptr<arrow::Field>, std::shared_ptr<arrow::Array>>>
FieldDataToArrow(const milvus::DataArray& field_data, size_t total_valid) {
    auto field_name = std::to_string(field_data.field_id());
    if (!field_data.has_scalars()) {
        return arrow::Status::NotImplemented("non-scalar output field not supported in Arrow export");
    }
    const auto& scalars = field_data.scalars();

    if (scalars.has_bool_data()) {
        const auto& data = scalars.bool_data().data();
        arrow::BooleanBuilder builder;
        ARROW_RETURN_NOT_OK(builder.Reserve(total_valid));
        for (size_t i = 0; i < total_valid; ++i) {
            builder.UnsafeAppend(data[i]);
        }
        std::shared_ptr<arrow::Array> arr;
        ARROW_RETURN_NOT_OK(builder.Finish(&arr));
        return std::make_pair(arrow::field(field_name, arrow::boolean()), arr);
    }
    if (scalars.has_int_data()) {
        const auto& data = scalars.int_data().data();
        arrow::Int32Builder builder;
        ARROW_RETURN_NOT_OK(builder.Reserve(total_valid));
        for (size_t i = 0; i < total_valid; ++i) {
            builder.UnsafeAppend(data[i]);
        }
        std::shared_ptr<arrow::Array> arr;
        ARROW_RETURN_NOT_OK(builder.Finish(&arr));
        return std::make_pair(arrow::field(field_name, arrow::int32()), arr);
    }
    if (scalars.has_long_data()) {
        const auto& data = scalars.long_data().data();
        arrow::Int64Builder builder;
        ARROW_RETURN_NOT_OK(builder.Reserve(total_valid));
        for (size_t i = 0; i < total_valid; ++i) {
            builder.UnsafeAppend(data[i]);
        }
        std::shared_ptr<arrow::Array> arr;
        ARROW_RETURN_NOT_OK(builder.Finish(&arr));
        return std::make_pair(arrow::field(field_name, arrow::int64()), arr);
    }
    if (scalars.has_float_data()) {
        const auto& data = scalars.float_data().data();
        arrow::FloatBuilder builder;
        ARROW_RETURN_NOT_OK(builder.Reserve(total_valid));
        for (size_t i = 0; i < total_valid; ++i) {
            builder.UnsafeAppend(data[i]);
        }
        std::shared_ptr<arrow::Array> arr;
        ARROW_RETURN_NOT_OK(builder.Finish(&arr));
        return std::make_pair(arrow::field(field_name, arrow::float32()), arr);
    }
    if (scalars.has_double_data()) {
        const auto& data = scalars.double_data().data();
        arrow::DoubleBuilder builder;
        ARROW_RETURN_NOT_OK(builder.Reserve(total_valid));
        for (size_t i = 0; i < total_valid; ++i) {
            builder.UnsafeAppend(data[i]);
        }
        std::shared_ptr<arrow::Array> arr;
        ARROW_RETURN_NOT_OK(builder.Finish(&arr));
        return std::make_pair(arrow::field(field_name, arrow::float64()), arr);
    }
    if (scalars.has_string_data()) {
        const auto& data = scalars.string_data().data();
        arrow::StringBuilder builder;
        for (size_t i = 0; i < total_valid; ++i) {
            ARROW_RETURN_NOT_OK(builder.Append(data[i]));
        }
        std::shared_ptr<arrow::Array> arr;
        ARROW_RETURN_NOT_OK(builder.Finish(&arr));
        return std::make_pair(arrow::field(field_name, arrow::utf8()), arr);
    }
    if (scalars.has_json_data()) {
        const auto& data = scalars.json_data().data();
        arrow::BinaryBuilder builder;
        for (size_t i = 0; i < total_valid; ++i) {
            ARROW_RETURN_NOT_OK(builder.Append(data[i]));
        }
        std::shared_ptr<arrow::Array> arr;
        ARROW_RETURN_NOT_OK(builder.Finish(&arr));
        return std::make_pair(arrow::field(field_name, arrow::binary()), arr);
    }

    return arrow::Status::NotImplemented("unsupported scalar type in Arrow export");
}

// Build Arrow RecordBatch from a SearchResult that has been filtered and had PKs filled.
// extra_fields contains additional field data to include (e.g., for L0 rerank).
arrow::Result<std::shared_ptr<arrow::RecordBatch>>
BuildSearchResultBatch(
    SearchResult* search_result,
    const std::vector<int64_t>& topk_per_nq,
    const std::map<milvus::FieldId, std::unique_ptr<milvus::DataArray>>& extra_fields) {
    auto total_valid = search_result->seg_offsets_.size();

    // Collect fields and arrays
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> arrays;

    // Build $id column based on PK type
    if (search_result->pk_type_ == milvus::DataType::INT64) {
        arrow::Int64Builder id_builder;
        ARROW_RETURN_NOT_OK(id_builder.Reserve(total_valid));
        for (size_t i = 0; i < total_valid; ++i) {
            auto& pk = search_result->primary_keys_[i];
            id_builder.UnsafeAppend(std::get<int64_t>(pk));
        }
        std::shared_ptr<arrow::Array> id_array;
        ARROW_RETURN_NOT_OK(id_builder.Finish(&id_array));
        fields.push_back(arrow::field("$id", arrow::int64()));
        arrays.push_back(id_array);
    } else {
        arrow::StringBuilder id_builder;
        for (size_t i = 0; i < total_valid; ++i) {
            auto& pk = search_result->primary_keys_[i];
            ARROW_RETURN_NOT_OK(
                id_builder.Append(std::get<std::string>(pk)));
        }
        std::shared_ptr<arrow::Array> id_array;
        ARROW_RETURN_NOT_OK(id_builder.Finish(&id_array));
        fields.push_back(arrow::field("$id", arrow::utf8()));
        arrays.push_back(id_array);
    }

    // Build $score column (float32)
    {
        arrow::FloatBuilder score_builder;
        ARROW_RETURN_NOT_OK(score_builder.Reserve(total_valid));
        for (size_t i = 0; i < total_valid; ++i) {
            score_builder.UnsafeAppend(search_result->distances_[i]);
        }
        std::shared_ptr<arrow::Array> score_array;
        ARROW_RETURN_NOT_OK(score_builder.Finish(&score_array));
        fields.push_back(arrow::field("$score", arrow::float32()));
        arrays.push_back(score_array);
    }

    // Build $seg_offset column (int64)
    {
        arrow::Int64Builder seg_offset_builder;
        ARROW_RETURN_NOT_OK(seg_offset_builder.Reserve(total_valid));
        for (size_t i = 0; i < total_valid; ++i) {
            seg_offset_builder.UnsafeAppend(search_result->seg_offsets_[i]);
        }
        std::shared_ptr<arrow::Array> seg_offset_array;
        ARROW_RETURN_NOT_OK(seg_offset_builder.Finish(&seg_offset_array));
        fields.push_back(arrow::field("$seg_offset", arrow::int64()));
        arrays.push_back(seg_offset_array);
    }

    // Build extra field columns (e.g., for L0 rerank)
    for (auto& [field_id, field_data] : extra_fields) {
        auto result = FieldDataToArrow(*field_data, total_valid);
        if (!result.ok()) {
            continue;
        }
        auto [field, arr] = *result;
        fields.push_back(field);
        arrays.push_back(arr);
    }

    // Build schema with metadata
    auto schema = arrow::schema(fields);
    auto metadata = arrow::KeyValueMetadata::Make(
        {"topk_per_nq"}, {TopkPerNQToString(topk_per_nq)});
    schema = schema->WithMetadata(metadata);

    return arrow::RecordBatch::Make(
        schema, total_valid, arrays);
}

}  // namespace

CStatus
ExportSearchResultAsArrow(CSearchResult c_search_result,
                          CSearchPlan c_plan,
                          const int64_t* extra_field_ids,
                          int64_t num_extra_fields,
                          struct ArrowArray* out_array,
                          struct ArrowSchema* out_schema) {
    SCOPE_CGO_CALL_METRIC();

    try {
        auto search_result = static_cast<SearchResult*>(c_search_result);
        auto plan = static_cast<milvus::query::Plan*>(c_plan);

        // Handle empty result
        if (search_result->unity_topK_ == 0 ||
            search_result->seg_offsets_.empty()) {
            auto empty_batch_result = BuildEmptyBatch();
            if (!empty_batch_result.ok()) {
                return milvus::FailureCStatus(
                    milvus::ErrorCode::UnexpectedError,
                    empty_batch_result.status().ToString());
            }
            auto status = arrow::ExportRecordBatch(
                *empty_batch_result.ValueUnsafe(), out_array, out_schema);
            if (!status.ok()) {
                return milvus::FailureCStatus(
                    milvus::ErrorCode::UnexpectedError, status.ToString());
            }
            return milvus::SuccessCStatus();
        }

        // Step 1: Filter invalid rows (offset == -1) and get per-NQ counts
        auto topk_per_nq = FilterAndGetTopkPerNQ(search_result);

        auto segment = static_cast<milvus::segcore::SegmentInternalInterface*>(
            search_result->segment_);

        // Step 2: Fill primary keys from segment
        if (search_result->get_total_result_count() > 0) {
            segment->FillPrimaryKeys(plan, *search_result);
        }

        // Step 3: Read extra fields (e.g., for L0 rerank) via bulk_subscript
        std::map<milvus::FieldId, std::unique_ptr<milvus::DataArray>> extra_fields;
        if (num_extra_fields > 0 && extra_field_ids != nullptr &&
            search_result->get_total_result_count() > 0) {
            auto size = search_result->seg_offsets_.size();
            milvus::OpContext op_ctx;
            for (int64_t i = 0; i < num_extra_fields; i++) {
                auto field_id = milvus::FieldId(extra_field_ids[i]);
                auto field_data = segment->bulk_subscript(
                    &op_ctx, field_id, search_result->seg_offsets_.data(), size);
                extra_fields[field_id] = std::move(field_data);
            }
            search_result->search_storage_cost_.scanned_remote_bytes +=
                op_ctx.storage_usage.scanned_cold_bytes.load();
            search_result->search_storage_cost_.scanned_total_bytes +=
                op_ctx.storage_usage.scanned_total_bytes.load();
        }

        // Step 4: Build Arrow RecordBatch (includes extra fields if present)
        auto batch_result =
            BuildSearchResultBatch(search_result, topk_per_nq, extra_fields);
        if (!batch_result.ok()) {
            return milvus::FailureCStatus(
                milvus::ErrorCode::UnexpectedError,
                batch_result.status().ToString());
        }

        // Step 5: Export via Arrow C Data Interface
        auto status = arrow::ExportRecordBatch(
            *batch_result.ValueUnsafe(), out_array, out_schema);
        if (!status.ok()) {
            return milvus::FailureCStatus(
                milvus::ErrorCode::UnexpectedError, status.ToString());
        }

        return milvus::SuccessCStatus();
    } catch (std::exception& e) {
        return milvus::FailureCStatus(&e);
    }
}

CStatus
FillOutputFieldsOrdered(CSearchResult* search_results,
                        int64_t num_search_results,
                        CSearchPlan c_plan,
                        const int32_t* result_seg_indices,
                        const int64_t* result_seg_offsets,
                        int64_t total_rows,
                        CProto* out_result) {
    SCOPE_CGO_CALL_METRIC();

    try {
        auto plan = static_cast<milvus::query::Plan*>(c_plan);

        if (plan->target_entries_.empty() || total_rows == 0) {
            out_result->proto_blob = nullptr;
            out_result->proto_size = 0;
            return milvus::SuccessCStatus();
        }

        // Step 1: Group offsets by segment index
        std::unordered_map<int32_t,
                           std::vector<std::pair<int64_t, int64_t>>>
            seg_groups;
        for (int64_t i = 0; i < total_rows; i++) {
            seg_groups[result_seg_indices[i]].emplace_back(i,
                                                           result_seg_offsets[i]);
        }

        // Step 2: FillTargetEntry per segment
        struct SegResult {
            SearchResult temp_result;
            std::vector<int64_t> result_positions;
        };
        std::unordered_map<int32_t, SegResult> seg_results;

        for (auto& [seg_idx, pairs] : seg_groups) {
            auto sr = static_cast<SearchResult*>(search_results[seg_idx]);
            auto segment =
                static_cast<milvus::segcore::SegmentInternalInterface*>(
                    sr->segment_);

            auto& seg_res = seg_results[seg_idx];
            seg_res.temp_result.segment_ = sr->segment_;
            seg_res.result_positions.reserve(pairs.size());

            for (auto& [pos, offset] : pairs) {
                seg_res.temp_result.seg_offsets_.push_back(offset);
                seg_res.result_positions.push_back(pos);
            }
            seg_res.temp_result.distances_.resize(pairs.size(), 0.0f);

            segment->FillTargetEntry(plan, seg_res.temp_result);
        }

        // Step 3: Build MergeBase array in output order
        std::vector<milvus::segcore::MergeBase> result_pairs(total_rows);
        for (auto& [seg_idx, seg_res] : seg_results) {
            for (size_t i = 0; i < seg_res.result_positions.size(); i++) {
                auto pos = seg_res.result_positions[i];
                result_pairs[pos] = {
                    &seg_res.temp_result.output_fields_data_,
                    i};
            }
        }

        // Step 4: Use MergeDataArray to assemble each field in order
        auto result_data =
            std::make_unique<milvus::proto::schema::SearchResultData>();
        for (auto field_id : plan->target_entries_) {
            auto& field_meta = plan->schema_->operator[](field_id);
            auto field_data =
                milvus::segcore::MergeDataArray(result_pairs, field_meta);
            if (field_meta.get_data_type() == milvus::DataType::ARRAY) {
                field_data->mutable_scalars()
                    ->mutable_array_data()
                    ->set_element_type(
                        milvus::proto::schema::DataType(
                            field_meta.get_element_type()));
            } else if (field_meta.get_data_type() ==
                       milvus::DataType::VECTOR_ARRAY) {
                field_data->mutable_vectors()
                    ->mutable_vector_array()
                    ->set_element_type(
                        milvus::proto::schema::DataType(
                            field_meta.get_element_type()));
            }
            result_data->mutable_fields_data()->AddAllocated(
                field_data.release());
        }

        // Step 5: Single serialization
        auto size = result_data->ByteSizeLong();
        void* buffer = malloc(size);
        if (buffer == nullptr) {
            return milvus::FailureCStatus(
                milvus::ErrorCode::UnexpectedError,
                "failed to allocate memory for proto serialization");
        }
        result_data->SerializeToArray(buffer, size);

        out_result->proto_blob = buffer;
        out_result->proto_size = size;

        return milvus::SuccessCStatus();
    } catch (std::exception& e) {
        return milvus::FailureCStatus(&e);
    }
}
