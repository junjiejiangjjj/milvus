/*
 * # Licensed to the LF AI & Data foundation under one
 * # or more contributor license agreements. See the NOTICE file
 * # distributed with this work for additional information
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

package types

import "github.com/apache/arrow/go/v17/arrow"

var highlightArrowType = arrow.ListOfNonNullable(arrow.StructOf(
	arrow.Field{Name: "field_name", Type: arrow.BinaryTypes.String, Nullable: false},
	arrow.Field{Name: "fragments", Type: arrow.ListOfNonNullable(arrow.BinaryTypes.String), Nullable: false},
	arrow.Field{Name: "scores", Type: arrow.ListOfNonNullable(arrow.PrimitiveTypes.Float32), Nullable: false},
))

// HighlightArrowType returns the canonical row-level Highlight result type.
// Empty results and empty fragments/scores are represented by empty lists,
// not NULL values.
func HighlightArrowType() arrow.DataType {
	return highlightArrowType
}

// IsHighlightArrowType reports whether dataType exactly matches the canonical
// Highlight result type, including child field names, order, and nullability.
func IsHighlightArrowType(dataType arrow.DataType) bool {
	return dataType != nil && arrow.TypeEqual(highlightArrowType, dataType)
}
