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

#include "SegmentInterface.h"

#include <cstdint>

#include "Utils.h"
#include "common/EasyAssert.h"
#include "common/SystemProperty.h"
#include "common/Tracer.h"
#include "common/Types.h"
#include "monitor/prometheus_client.h"
#include "query/ExecPlanNodeVisitor.h"

namespace milvus::segcore {

void
SegmentInternalInterface::FillPrimaryKeys(const query::Plan* plan,
                                          SearchResult& results) const {
    std::shared_lock lck(mutex_);
    AssertInfo(plan, "empty plan");
    auto size = results.distances_.size();
    AssertInfo(results.seg_offsets_.size() == size,
               "Size of result distances is not equal to size of ids");
    Assert(results.primary_keys_.size() == 0);
    results.primary_keys_.resize(size);

    auto pk_field_id_opt = get_schema().get_primary_field_id();
    AssertInfo(pk_field_id_opt.has_value(),
               "Cannot get primary key offset from schema");
    auto pk_field_id = pk_field_id_opt.value();
    AssertInfo(IsPrimaryKeyDataType(get_schema()[pk_field_id].get_data_type()),
               "Primary key field is not INT64 or VARCHAR type");

    auto field_data =
        bulk_subscript(pk_field_id, results.seg_offsets_.data(), size);
    results.pk_type_ = DataType(field_data->type());

    ParsePksFromFieldData(results.primary_keys_, *field_data.get());
}

void
SegmentInternalInterface::FillTargetEntry(const query::Plan* plan,
                                          SearchResult& results) const {
    std::shared_lock lck(mutex_);
    AssertInfo(plan, "empty plan");
    auto size = results.distances_.size();
    AssertInfo(results.seg_offsets_.size() == size,
               "Size of result distances is not equal to size of ids");

    std::unique_ptr<DataArray> field_data;
    // fill other entries except primary key by result_offset
    for (auto field_id : plan->target_entries_) {
        auto& field_meta = plan->schema_->operator[](field_id);
        if (plan->schema_->get_dynamic_field_id().has_value() &&
            plan->schema_->get_dynamic_field_id().value() == field_id &&
            !plan->target_dynamic_fields_.empty()) {
            auto& target_dynamic_fields = plan->target_dynamic_fields_;
            field_data = bulk_subscript(field_id,
                                        results.seg_offsets_.data(),
                                        size,
                                        target_dynamic_fields);
        } else if (!is_field_exist(field_id)) {
            field_data = bulk_subscript_not_exist_field(field_meta, size);
        } else {
            field_data =
                bulk_subscript(field_id, results.seg_offsets_.data(), size);
        }
        results.output_fields_data_[field_id] = std::move(field_data);
    }
}

std::unique_ptr<SearchResult>
SegmentInternalInterface::Search(
    const query::Plan* plan,
    const query::PlaceholderGroup* placeholder_group,
    Timestamp timestamp,
    int32_t consistency_level,
    Timestamp collection_ttl) const {
    std::shared_lock lck(mutex_);
    milvus::tracer::AddEvent("obtained_segment_lock_mutex");
    check_search(plan);
    query::ExecPlanNodeVisitor visitor(
        *this, timestamp, placeholder_group, consistency_level, collection_ttl);
    auto results = std::make_unique<SearchResult>();
    *results = visitor.get_moved_result(*plan->plan_node_);
    results->segment_ = (void*)this;
    return results;
}

std::unique_ptr<proto::segcore::RetrieveResults>
SegmentInternalInterface::Retrieve(tracer::TraceContext* trace_ctx,
                                   const query::RetrievePlan* plan,
                                   Timestamp timestamp,
                                   int64_t limit_size,
                                   bool ignore_non_pk,
                                   int32_t consistency_level,
                                   Timestamp collection_ttl) const {
    std::shared_lock lck(mutex_);
    tracer::AutoSpan span("Retrieve", tracer::GetRootSpan());
    auto results = std::make_unique<proto::segcore::RetrieveResults>();
    query::ExecPlanNodeVisitor visitor(
        *this, timestamp, consistency_level, collection_ttl);
    auto retrieve_results = visitor.get_retrieve_result(*plan->plan_node_);
    retrieve_results.segment_ = (void*)this;
    results->set_has_more_result(retrieve_results.has_more_result);

    auto result_rows = retrieve_results.result_offsets_.size();
    int64_t output_data_size = 0;
    for (auto field_id : plan->field_ids_) {
        output_data_size += get_field_avg_size(field_id) * result_rows;
    }
    if (output_data_size > limit_size) {
        ThrowInfo(
            RetrieveError,
            fmt::format("query results exceed the limit size ", limit_size));
    }

    results->set_all_retrieve_count(retrieve_results.total_data_cnt_);
    if (plan->plan_node_->is_count_) {
        AssertInfo(retrieve_results.field_data_.size() == 1,
                   "count result should only have one column");
        *results->add_fields_data() = retrieve_results.field_data_[0];
        return results;
    }

    results->mutable_offset()->Add(retrieve_results.result_offsets_.begin(),
                                   retrieve_results.result_offsets_.end());

    std::chrono::high_resolution_clock::time_point get_target_entry_start =
        std::chrono::high_resolution_clock::now();
    FillTargetEntry(trace_ctx,
                    plan,
                    results,
                    retrieve_results.result_offsets_.data(),
                    retrieve_results.result_offsets_.size(),
                    ignore_non_pk,
                    true);
    std::chrono::high_resolution_clock::time_point get_target_entry_end =
        std::chrono::high_resolution_clock::now();
    double get_entry_cost = std::chrono::duration<double, std::micro>(
                                get_target_entry_end - get_target_entry_start)
                                .count();
    monitor::internal_core_retrieve_get_target_entry_latency.Observe(
        get_entry_cost / 1000);
    return results;
}

void
SegmentInternalInterface::FillTargetEntry(
    tracer::TraceContext* trace_ctx,
    const query::RetrievePlan* plan,
    const std::unique_ptr<proto::segcore::RetrieveResults>& results,
    const int64_t* offsets,
    int64_t size,
    bool ignore_non_pk,
    bool fill_ids) const {
    tracer::AutoSpan span("FillTargetEntry", tracer::GetRootSpan());

    auto fields_data = results->mutable_fields_data();
    auto ids = results->mutable_ids();
    auto pk_field_id = plan->schema_->get_primary_field_id();

    auto is_pk_field = [&, pk_field_id](const FieldId& field_id) -> bool {
        return pk_field_id.has_value() && pk_field_id.value() == field_id;
    };

    for (auto field_id : plan->field_ids_) {
        if (SystemProperty::Instance().IsSystem(field_id)) {
            auto system_type =
                SystemProperty::Instance().GetSystemFieldType(field_id);

            FixedVector<int64_t> output(size);
            bulk_subscript(system_type, offsets, size, output.data());

            auto data_array = std::make_unique<DataArray>();
            data_array->set_field_id(field_id.get());
            data_array->set_type(milvus::proto::schema::DataType::Int64);

            auto scalar_array = data_array->mutable_scalars();
            auto data = reinterpret_cast<const int64_t*>(output.data());
            auto obj = scalar_array->mutable_long_data();
            obj->mutable_data()->Add(data, data + size);
            fields_data->AddAllocated(data_array.release());
            continue;
        }

        if (ignore_non_pk && !is_pk_field(field_id)) {
            continue;
        }

        if (plan->schema_->get_dynamic_field_id().has_value() &&
            plan->schema_->get_dynamic_field_id().value() == field_id &&
            !plan->target_dynamic_fields_.empty()) {
            auto& target_dynamic_fields = plan->target_dynamic_fields_;
            auto col =
                bulk_subscript(field_id, offsets, size, target_dynamic_fields);
            fields_data->AddAllocated(col.release());
            continue;
        }
        std::unique_ptr<DataArray> col;
        auto& field_meta = plan->schema_->operator[](field_id);
        if (!is_field_exist(field_id)) {
            col = std::move(bulk_subscript_not_exist_field(field_meta, size));
        } else {
            col = bulk_subscript(field_id, offsets, size);
        }
        // todo(SpadeA): consider vector array?
        if (field_meta.get_data_type() == DataType::ARRAY) {
            col->mutable_scalars()->mutable_array_data()->set_element_type(
                proto::schema::DataType(field_meta.get_element_type()));
        }
        if (fill_ids && is_pk_field(field_id)) {
            // fill_ids should be true when the first Retrieve was called. The reduce phase depends on the ids to do
            // merge-sort.
            auto col_data = col.get();
            switch (field_meta.get_data_type()) {
                case DataType::INT64: {
                    auto int_ids = ids->mutable_int_id();
                    auto& src_data = col_data->scalars().long_data();
                    int_ids->mutable_data()->Add(src_data.data().begin(),
                                                 src_data.data().end());
                    break;
                }
                case DataType::VARCHAR: {
                    auto str_ids = ids->mutable_str_id();
                    auto& src_data = col_data->scalars().string_data();
                    for (auto i = 0; i < src_data.data_size(); ++i) {
                        *(str_ids->mutable_data()->Add()) = src_data.data(i);
                    }
                    break;
                }
                default: {
                    ThrowInfo(DataTypeInvalid,
                              fmt::format("unsupported datatype {}",
                                          field_meta.get_data_type()));
                }
            }
        }
        if (!ignore_non_pk) {
            // when ignore_non_pk is false, it indicates two situations:
            //  1. No need to do the two-phase Retrieval, the target entries should be returned as the first Retrieval
            //      is done, below two cases are included:
            //       a. There is only one segment;
            //       b. No pagination is used;
            //  2. The FillTargetEntry was called by the second Retrieval (by offsets).
            fields_data->AddAllocated(col.release());
        }
    }
}

std::unique_ptr<proto::segcore::RetrieveResults>
SegmentInternalInterface::Retrieve(tracer::TraceContext* trace_ctx,
                                   const query::RetrievePlan* Plan,
                                   const int64_t* offsets,
                                   int64_t size) const {
    std::shared_lock lck(mutex_);
    tracer::AutoSpan span("RetrieveByOffsets", tracer::GetRootSpan());
    auto results = std::make_unique<proto::segcore::RetrieveResults>();
    std::chrono::high_resolution_clock::time_point get_target_entry_start =
        std::chrono::high_resolution_clock::now();
    FillTargetEntry(trace_ctx, Plan, results, offsets, size, false, false);
    std::chrono::high_resolution_clock::time_point get_target_entry_end =
        std::chrono::high_resolution_clock::now();
    double get_entry_cost = std::chrono::duration<double, std::micro>(
                                get_target_entry_end - get_target_entry_start)
                                .count();
    monitor::internal_core_retrieve_get_target_entry_latency.Observe(
        get_entry_cost / 1000);
    return results;
}

int64_t
SegmentInternalInterface::get_real_count() const {
#if 0
    auto insert_cnt = get_row_count();
    BitsetType bitset_holder;
    bitset_holder.resize(insert_cnt, false);
    mask_with_delete(bitset_holder, insert_cnt, MAX_TIMESTAMP);
    return bitset_holder.size() - bitset_holder.count();
#endif
    auto plan = std::make_unique<query::RetrievePlan>(
        std::make_shared<Schema>(get_schema()));
    plan->plan_node_ = std::make_unique<query::RetrievePlanNode>();
    milvus::plan::PlanNodePtr plannode;
    std::vector<milvus::plan::PlanNodePtr> sources;
    plannode = std::make_shared<milvus::plan::MvccNode>(
        milvus::plan::GetNextPlanNodeId());
    sources = std::vector<milvus::plan::PlanNodePtr>{plannode};
    plannode = std::make_shared<milvus::plan::CountNode>(
        milvus::plan::GetNextPlanNodeId(), sources);
    plan->plan_node_->plannodes_ = plannode;
    plan->plan_node_->is_count_ = true;
    auto res =
        Retrieve(nullptr, plan.get(), MAX_TIMESTAMP, INT64_MAX, false, 0);
    AssertInfo(res->fields_data().size() == 1,
               "count result should only have one column");
    AssertInfo(res->fields_data()[0].has_scalars(),
               "count result should match scalar");
    AssertInfo(res->fields_data()[0].scalars().has_long_data(),
               "count result should match long data");
    AssertInfo(res->fields_data()[0].scalars().long_data().data_size() == 1,
               "count result should only have one row");
    return res->fields_data()[0].scalars().long_data().data(0);
}

int64_t
SegmentInternalInterface::get_field_avg_size(FieldId field_id) const {
    AssertInfo(field_id.get() >= 0,
               "invalid field id, should be greater than or equal to 0");
    if (SystemProperty::Instance().IsSystem(field_id)) {
        if (field_id == TimestampFieldID || field_id == RowFieldID) {
            return sizeof(int64_t);
        }

        ThrowInfo(FieldIDInvalid, "unsupported system field id");
    }

    auto& schema = get_schema();
    auto& field_meta = schema[field_id];
    auto data_type = field_meta.get_data_type();

    std::shared_lock lck(mutex_);
    if (IsVariableDataType(data_type)) {
        if (variable_fields_avg_size_.find(field_id) ==
            variable_fields_avg_size_.end()) {
            return 0;
        }

        return variable_fields_avg_size_.at(field_id).second;
    } else {
        return field_meta.get_sizeof();
    }
}

void
SegmentInternalInterface::set_field_avg_size(FieldId field_id,
                                             int64_t num_rows,
                                             int64_t field_size) {
    AssertInfo(field_id.get() >= 0,
               "invalid field id, should be greater than or equal to 0");
    auto& schema = get_schema();
    auto& field_meta = schema[field_id];
    auto data_type = field_meta.get_data_type();

    std::unique_lock lck(mutex_);
    if (IsVariableDataType(data_type)) {
        AssertInfo(num_rows > 0,
                   "The num rows of field data should be greater than 0");
        if (variable_fields_avg_size_.find(field_id) ==
            variable_fields_avg_size_.end()) {
            variable_fields_avg_size_.emplace(field_id, std::make_pair(0, 0));
        }

        auto& field_info = variable_fields_avg_size_.at(field_id);
        auto size = field_info.first * field_info.second + field_size;
        field_info.first = field_info.first + num_rows;
        field_info.second = size / field_info.first;
    }
}

void
SegmentInternalInterface::timestamp_filter(BitsetType& bitset,
                                           Timestamp timestamp) const {
    auto& timestamps = get_timestamps();
    auto cnt = bitset.size();
    if (timestamps[cnt - 1] <= timestamp) {
        // no need to filter out anything.
        return;
    }

    auto pilot = upper_bound(timestamps, 0, cnt, timestamp);
    // offset bigger than pilot should be filtered out.
    auto offset = pilot;
    while (offset < cnt) {
        bitset[offset] = false;

        const auto next_offset = bitset.find_next(offset);
        if (!next_offset.has_value()) {
            return;
        }
        offset = next_offset.value();
    }
}

void
SegmentInternalInterface::timestamp_filter(BitsetType& bitset,
                                           const std::vector<int64_t>& offsets,
                                           Timestamp timestamp) const {
    auto& timestamps = get_timestamps();
    auto cnt = bitset.size();
    if (timestamps[cnt - 1] <= timestamp) {
        // no need to filter out anything.
        return;
    }

    // point query, faster than binary search.
    for (auto& offset : offsets) {
        if (timestamps[offset] > timestamp) {
            bitset.set(offset, true);
        }
    }
}

const SkipIndex&
SegmentInternalInterface::GetSkipIndex() const {
    return skip_index_;
}

index::TextMatchIndex*
SegmentInternalInterface::GetTextIndex(FieldId field_id) const {
    std::shared_lock lock(mutex_);
    auto iter = text_indexes_.find(field_id);
    if (iter == text_indexes_.end()) {
        throw SegcoreError(
            ErrorCode::TextIndexNotFound,
            fmt::format("text index not found for field {}", field_id.get()));
    }
    return iter->second.get();
}

std::unique_ptr<DataArray>
SegmentInternalInterface::bulk_subscript_not_exist_field(
    const milvus::FieldMeta& field_meta, int64_t count) const {
    auto data_type = field_meta.get_data_type();
    if (IsVectorDataType(data_type)) {
        ThrowInfo(DataTypeInvalid,
                  fmt::format("unsupported added field type {}",
                              field_meta.get_data_type()));
    }
    auto result = CreateEmptyScalarDataArray(count, field_meta);
    if (field_meta.default_value().has_value()) {
        auto res = result->mutable_valid_data()->mutable_data();
        for (int64_t i = 0; i < count; ++i) {
            res[i] = true;
        }
        switch (field_meta.get_data_type()) {
            case DataType::BOOL: {
                auto data_ptr = result->mutable_scalars()
                                    ->mutable_bool_data()
                                    ->mutable_data()
                                    ->mutable_data();

                for (int64_t i = 0; i < count; ++i) {
                    data_ptr[i] = field_meta.default_value()->bool_data();
                }
                break;
            }
            case DataType::INT8:
            case DataType::INT16:
            case DataType::INT32: {
                auto data_ptr = result->mutable_scalars()
                                    ->mutable_int_data()
                                    ->mutable_data()
                                    ->mutable_data();

                for (int64_t i = 0; i < count; ++i) {
                    data_ptr[i] = field_meta.default_value()->int_data();
                }
                break;
            }
            case DataType::INT64: {
                auto data_ptr = result->mutable_scalars()
                                    ->mutable_long_data()
                                    ->mutable_data()
                                    ->mutable_data();

                for (int64_t i = 0; i < count; ++i) {
                    data_ptr[i] = field_meta.default_value()->long_data();
                }
                break;
            }
            case DataType::FLOAT: {
                auto data_ptr = result->mutable_scalars()
                                    ->mutable_float_data()
                                    ->mutable_data()
                                    ->mutable_data();

                for (int64_t i = 0; i < count; ++i) {
                    data_ptr[i] = field_meta.default_value()->float_data();
                }
                break;
            }
            case DataType::DOUBLE: {
                auto data_ptr = result->mutable_scalars()
                                    ->mutable_double_data()
                                    ->mutable_data()
                                    ->mutable_data();

                for (int64_t i = 0; i < count; ++i) {
                    data_ptr[i] = field_meta.default_value()->double_data();
                }
                break;
            }
            case DataType::VARCHAR: {
                auto data_ptr = result->mutable_scalars()
                                    ->mutable_string_data()
                                    ->mutable_data();

                for (int64_t i = 0; i < count; ++i) {
                    data_ptr->at(i) = field_meta.default_value()->string_data();
                }
                break;
            }
            default: {
                ThrowInfo(DataTypeInvalid,
                          fmt::format("unsupported default value type {}",
                                      field_meta.get_data_type()));
            }
        }
        return result;
    };
    for (int64_t i = 0; i < count; ++i) {
        auto res = result->mutable_valid_data()->mutable_data();
        res[i] = false;
    }
    return result;
}

index::JsonKeyStatsInvertedIndex*
SegmentInternalInterface::GetJsonKeyIndex(FieldId field_id) const {
    std::shared_lock lock(mutex_);
    auto iter = json_indexes_.find(field_id);
    if (iter == json_indexes_.end()) {
        return nullptr;
    }
    return iter->second.get();
}

// Only sealed segment has ngram index
PinWrapper<index::NgramInvertedIndex*>
SegmentInternalInterface::GetNgramIndex(FieldId field_id) const {
    return PinWrapper<index::NgramInvertedIndex*>(nullptr);
}

}  // namespace milvus::segcore
