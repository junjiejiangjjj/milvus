/*
 * Licensed to the LF AI & Data foundation under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package tasks

import (
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/util/function/chain"
	"github.com/milvus-io/milvus/pkg/v2/util/merr"
)

// marshalReduceResult converts a mergeResult into a SearchResultData proto.
// It exports $id, $score, and (when present) the $group_by column from the
// merged DataFrame produced by heapMergeReduce. Output field data is filled in
// later by lateMaterializeOutputFields.
func marshalReduceResult(result *mergeResult) (*schemapb.SearchResultData, error) {
	if result == nil || result.DF == nil {
		return nil, merr.WrapErrServiceInternal("nil reduce result")
	}
	data, err := chain.ToSearchResultDataWithOptions(result.DF, &chain.ExportOptions{
		GroupByField: groupByCol,
	})
	if err != nil {
		return nil, err
	}
	// chain.ExportOptions populates GroupByFieldValue.FieldName from the column
	// name ("$group_by"). Clear it so proxy.fillFieldNames resolves the real
	// schema field name from FieldId — matching the legacy C++ reduce path.
	if data.GroupByFieldValue != nil {
		data.GroupByFieldValue.FieldName = ""
	}
	return data, nil
}
