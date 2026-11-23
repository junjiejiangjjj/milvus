// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <cstddef>
#include <cstdint>
#include <variant>

#include <nlohmann/json_fwd.hpp>

namespace milvus::grep {

class Result;

// Borrows one selected input from Result. Serialize synchronously before the
// native result is moved or destroyed; no text or span buffers are copied here.
struct GrepDetailsView {
    GrepDetailsView(const Result& result, size_t text_index)
        : result(result), text_index(text_index) {
    }
    GrepDetailsView(const Result&&, size_t) = delete;

    const Result& result;
    size_t text_index;

    nlohmann::json
    ToJson() const;
};

enum class GrepCountUnit {
    MatchingLines,
    Matches,
};

struct GrepCount {
    uint64_t count;
    GrepCountUnit unit;
    bool complete;

    nlohmann::json
    ToJson() const;
};

// Represents a selected input for which no details are requested, not a miss.
struct GrepExists {
    nlohmann::json
    ToJson() const;
};

// C++ output model; the execution mode determines which payload is populated.
struct GrepResult {
    using Payload = std::variant<GrepDetailsView, GrepCount, GrepExists>;
    Payload payload;

    nlohmann::json
    ToJson() const;
};

}  // namespace milvus::grep
