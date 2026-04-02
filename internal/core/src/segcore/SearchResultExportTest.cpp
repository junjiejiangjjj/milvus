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

#include <arrow/api.h>
#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "common/Consts.h"
#include "common/QueryResult.h"
#include "gtest/gtest.h"
#include "segcore/Utils.h"

using milvus::DataType;
using milvus::GroupByValueType;
using milvus::PkType;
using milvus::SearchResult;
using milvus::segcore::SortEqualScoresByPks;

// ---------------------------------------------------------------------------
// CompactValidRows — tested indirectly via the full export path, but we need
// a direct test to verify the compact + prefix-sum logic. Since CompactValidRows
// is in an anonymous namespace, we test it through SortEqualScoresByPks which
// requires the prefix sum to be pre-built. We replicate the compact logic here.
// ---------------------------------------------------------------------------

namespace {

// Mirror of CompactValidRows for testing (the real one is in anonymous namespace).
void
TestCompactValidRows(SearchResult* sr) {
    auto nq = sr->total_nq_;
    auto topK = sr->unity_topK_;
    auto& offsets = sr->seg_offsets_;
    auto& distances = sr->distances_;
    auto& prefix = sr->topk_per_nq_prefix_sum_;
    prefix.assign(nq + 1, 0);
    uint32_t valid_index = 0;

    const bool has_element_indices =
        sr->element_level_ && !sr->element_indices_.empty();

    for (int64_t i = 0; i < nq; ++i) {
        for (int64_t j = 0; j < topK; ++j) {
            auto index = i * topK + j;
            if (offsets[index] != INVALID_SEG_OFFSET) {
                offsets[valid_index] = offsets[index];
                distances[valid_index] = distances[index];
                if (has_element_indices) {
                    sr->element_indices_[valid_index] =
                        sr->element_indices_[index];
                }
                valid_index++;
            }
        }
        prefix[i + 1] = valid_index;
    }

    offsets.resize(valid_index);
    distances.resize(valid_index);
    if (has_element_indices) {
        sr->element_indices_.resize(valid_index);
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// SortEqualScoresByPks
// ---------------------------------------------------------------------------

TEST(SearchResultExport, SortEqualScoresByPks_Basic) {
    SearchResult sr;
    sr.total_nq_ = 1;
    sr.unity_topK_ = 4;

    // Scores: all equal → should sort by PK ASC
    sr.distances_ = {1.0f, 1.0f, 1.0f, 1.0f};
    sr.seg_offsets_ = {30, 10, 40, 20};
    sr.primary_keys_ = {PkType(int64_t(300)), PkType(int64_t(100)),
                        PkType(int64_t(400)), PkType(int64_t(200))};

    // Build prefix sum (required by SortEqualScoresByPks)
    sr.topk_per_nq_prefix_sum_ = {0, 4};

    SortEqualScoresByPks(&sr);

    // After sort: PKs should be in ASC order
    EXPECT_EQ(std::get<int64_t>(sr.primary_keys_[0]), 100);
    EXPECT_EQ(std::get<int64_t>(sr.primary_keys_[1]), 200);
    EXPECT_EQ(std::get<int64_t>(sr.primary_keys_[2]), 300);
    EXPECT_EQ(std::get<int64_t>(sr.primary_keys_[3]), 400);

    // seg_offsets should follow the same permutation
    EXPECT_EQ(sr.seg_offsets_[0], 10);
    EXPECT_EQ(sr.seg_offsets_[1], 20);
    EXPECT_EQ(sr.seg_offsets_[2], 30);
    EXPECT_EQ(sr.seg_offsets_[3], 40);
}

TEST(SearchResultExport, SortEqualScoresByPks_MixedScores) {
    SearchResult sr;
    sr.total_nq_ = 1;
    sr.unity_topK_ = 6;

    // Two equal-score groups: [5.0, 5.0, 5.0] and [3.0, 3.0, 3.0]
    sr.distances_ = {5.0f, 5.0f, 5.0f, 3.0f, 3.0f, 3.0f};
    sr.seg_offsets_ = {30, 10, 20, 60, 40, 50};
    sr.primary_keys_ = {PkType(int64_t(300)), PkType(int64_t(100)),
                        PkType(int64_t(200)), PkType(int64_t(600)),
                        PkType(int64_t(400)), PkType(int64_t(500))};
    sr.topk_per_nq_prefix_sum_ = {0, 6};

    SortEqualScoresByPks(&sr);

    // First group (score=5.0): PKs sorted ASC
    EXPECT_EQ(std::get<int64_t>(sr.primary_keys_[0]), 100);
    EXPECT_EQ(std::get<int64_t>(sr.primary_keys_[1]), 200);
    EXPECT_EQ(std::get<int64_t>(sr.primary_keys_[2]), 300);
    // Second group (score=3.0): PKs sorted ASC
    EXPECT_EQ(std::get<int64_t>(sr.primary_keys_[3]), 400);
    EXPECT_EQ(std::get<int64_t>(sr.primary_keys_[4]), 500);
    EXPECT_EQ(std::get<int64_t>(sr.primary_keys_[5]), 600);

    // Distances unchanged (all equal within groups)
    EXPECT_FLOAT_EQ(sr.distances_[0], 5.0f);
    EXPECT_FLOAT_EQ(sr.distances_[3], 3.0f);
}

TEST(SearchResultExport, SortEqualScoresByPks_WithElementIndices) {
    SearchResult sr;
    sr.total_nq_ = 1;
    sr.unity_topK_ = 3;
    sr.element_level_ = true;

    sr.distances_ = {1.0f, 1.0f, 1.0f};
    sr.seg_offsets_ = {30, 10, 20};
    sr.primary_keys_ = {PkType(int64_t(300)), PkType(int64_t(100)),
                        PkType(int64_t(200))};
    sr.element_indices_ = {33, 11, 22};
    sr.topk_per_nq_prefix_sum_ = {0, 3};

    SortEqualScoresByPks(&sr);

    // element_indices should follow the same permutation as PKs
    EXPECT_EQ(sr.element_indices_[0], 11);  // was at index 1 (PK=100)
    EXPECT_EQ(sr.element_indices_[1], 22);  // was at index 2 (PK=200)
    EXPECT_EQ(sr.element_indices_[2], 33);  // was at index 0 (PK=300)
}

TEST(SearchResultExport, SortEqualScoresByPks_WithGroupBy) {
    SearchResult sr;
    sr.total_nq_ = 1;
    sr.unity_topK_ = 3;

    sr.distances_ = {1.0f, 1.0f, 1.0f};
    sr.seg_offsets_ = {30, 10, 20};
    sr.primary_keys_ = {PkType(int64_t(300)), PkType(int64_t(100)),
                        PkType(int64_t(200))};
    sr.group_by_values_ = std::vector<GroupByValueType>{
        GroupByValueType(int64_t(3)), GroupByValueType(int64_t(1)),
        GroupByValueType(int64_t(2))};
    sr.topk_per_nq_prefix_sum_ = {0, 3};

    SortEqualScoresByPks(&sr);

    // group_by_values should follow the same permutation
    auto& gbv = sr.group_by_values_.value();
    EXPECT_EQ(std::get<int64_t>(gbv[0].value()), 1);
    EXPECT_EQ(std::get<int64_t>(gbv[1].value()), 2);
    EXPECT_EQ(std::get<int64_t>(gbv[2].value()), 3);
}

TEST(SearchResultExport, SortEqualScoresByPks_MultiNQ) {
    SearchResult sr;
    sr.total_nq_ = 2;
    sr.unity_topK_ = 3;

    // NQ0: 3 results with equal scores, unsorted PKs
    // NQ1: 3 results with equal scores, unsorted PKs
    sr.distances_ = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f};
    sr.seg_offsets_ = {30, 10, 20, 60, 40, 50};
    sr.primary_keys_ = {PkType(int64_t(30)), PkType(int64_t(10)),
                        PkType(int64_t(20)), PkType(int64_t(60)),
                        PkType(int64_t(40)), PkType(int64_t(50))};

    TestCompactValidRows(&sr);
    SortEqualScoresByPks(&sr);

    // NQ0: sorted
    EXPECT_EQ(std::get<int64_t>(sr.primary_keys_[0]), 10);
    EXPECT_EQ(std::get<int64_t>(sr.primary_keys_[1]), 20);
    EXPECT_EQ(std::get<int64_t>(sr.primary_keys_[2]), 30);
    // NQ1: sorted
    EXPECT_EQ(std::get<int64_t>(sr.primary_keys_[3]), 40);
    EXPECT_EQ(std::get<int64_t>(sr.primary_keys_[4]), 50);
    EXPECT_EQ(std::get<int64_t>(sr.primary_keys_[5]), 60);
}

TEST(SearchResultExport, SortEqualScoresByPks_EmptyElementIndices) {
    // element_level_ is true but element_indices_ is empty — should not crash
    SearchResult sr;
    sr.total_nq_ = 1;
    sr.unity_topK_ = 2;
    sr.element_level_ = true;

    sr.distances_ = {1.0f, 1.0f};
    sr.seg_offsets_ = {20, 10};
    sr.primary_keys_ = {PkType(int64_t(200)), PkType(int64_t(100))};
    // element_indices_ intentionally left empty
    sr.topk_per_nq_prefix_sum_ = {0, 2};

    // Should not crash (the fix checks !element_indices_.empty())
    SortEqualScoresByPks(&sr);

    EXPECT_EQ(std::get<int64_t>(sr.primary_keys_[0]), 100);
    EXPECT_EQ(std::get<int64_t>(sr.primary_keys_[1]), 200);
}

TEST(SearchResultExport, SortEqualScoresByPks_SingleElement) {
    SearchResult sr;
    sr.total_nq_ = 1;
    sr.unity_topK_ = 1;

    sr.distances_ = {1.0f};
    sr.seg_offsets_ = {10};
    sr.primary_keys_ = {PkType(int64_t(100))};
    sr.topk_per_nq_prefix_sum_ = {0, 1};

    // Single element — should be a no-op
    SortEqualScoresByPks(&sr);
    EXPECT_EQ(std::get<int64_t>(sr.primary_keys_[0]), 100);
}

// ---------------------------------------------------------------------------
// CompactValidRows (via TestCompactValidRows mirror)
// ---------------------------------------------------------------------------

TEST(SearchResultExport, CompactValidRows_AllValid) {
    SearchResult sr;
    sr.total_nq_ = 2;
    sr.unity_topK_ = 2;
    sr.seg_offsets_ = {10, 20, 30, 40};
    sr.distances_ = {1.0f, 2.0f, 3.0f, 4.0f};

    TestCompactValidRows(&sr);

    EXPECT_EQ(sr.seg_offsets_.size(), 4u);
    EXPECT_EQ(sr.topk_per_nq_prefix_sum_[0], 0u);
    EXPECT_EQ(sr.topk_per_nq_prefix_sum_[1], 2u);
    EXPECT_EQ(sr.topk_per_nq_prefix_sum_[2], 4u);
}

TEST(SearchResultExport, CompactValidRows_AllInvalid) {
    SearchResult sr;
    sr.total_nq_ = 2;
    sr.unity_topK_ = 2;
    sr.seg_offsets_ = {INVALID_SEG_OFFSET, INVALID_SEG_OFFSET,
                       INVALID_SEG_OFFSET, INVALID_SEG_OFFSET};
    sr.distances_ = {1.0f, 2.0f, 3.0f, 4.0f};

    TestCompactValidRows(&sr);

    EXPECT_EQ(sr.seg_offsets_.size(), 0u);
    EXPECT_EQ(sr.topk_per_nq_prefix_sum_[0], 0u);
    EXPECT_EQ(sr.topk_per_nq_prefix_sum_[1], 0u);
    EXPECT_EQ(sr.topk_per_nq_prefix_sum_[2], 0u);
}

TEST(SearchResultExport, CompactValidRows_Mixed) {
    SearchResult sr;
    sr.total_nq_ = 2;
    sr.unity_topK_ = 3;
    // NQ0: valid, invalid, valid → 2 valid
    // NQ1: invalid, valid, invalid → 1 valid
    sr.seg_offsets_ = {10, INVALID_SEG_OFFSET, 30,
                       INVALID_SEG_OFFSET, 50, INVALID_SEG_OFFSET};
    sr.distances_ = {1.0f, 0.0f, 3.0f, 0.0f, 5.0f, 0.0f};

    TestCompactValidRows(&sr);

    EXPECT_EQ(sr.seg_offsets_.size(), 3u);
    EXPECT_EQ(sr.seg_offsets_[0], 10);
    EXPECT_EQ(sr.seg_offsets_[1], 30);
    EXPECT_EQ(sr.seg_offsets_[2], 50);

    EXPECT_EQ(sr.topk_per_nq_prefix_sum_[0], 0u);
    EXPECT_EQ(sr.topk_per_nq_prefix_sum_[1], 2u);
    EXPECT_EQ(sr.topk_per_nq_prefix_sum_[2], 3u);
}

TEST(SearchResultExport, CompactValidRows_WithElementIndices) {
    SearchResult sr;
    sr.total_nq_ = 1;
    sr.unity_topK_ = 3;
    sr.element_level_ = true;
    sr.seg_offsets_ = {10, INVALID_SEG_OFFSET, 30};
    sr.distances_ = {1.0f, 0.0f, 3.0f};
    sr.element_indices_ = {11, 0, 33};

    TestCompactValidRows(&sr);

    EXPECT_EQ(sr.seg_offsets_.size(), 2u);
    EXPECT_EQ(sr.element_indices_.size(), 2u);
    EXPECT_EQ(sr.element_indices_[0], 11);
    EXPECT_EQ(sr.element_indices_[1], 33);
}
