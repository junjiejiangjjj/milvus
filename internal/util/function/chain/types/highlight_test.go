/*
 * # Licensed to the LF AI & Data foundation under one
 * # or more contributor license agreements. See the NOTICE file
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

package types

import (
	"testing"

	"github.com/apache/arrow/go/v17/arrow"
	"github.com/stretchr/testify/assert"
)

func TestHighlightArrowType(t *testing.T) {
	expected := arrow.ListOfNonNullable(arrow.StructOf(
		arrow.Field{Name: "field_name", Type: arrow.BinaryTypes.String, Nullable: false},
		arrow.Field{Name: "fragments", Type: arrow.ListOfNonNullable(arrow.BinaryTypes.String), Nullable: false},
		arrow.Field{Name: "scores", Type: arrow.ListOfNonNullable(arrow.PrimitiveTypes.Float32), Nullable: false},
	))
	assert.Equal(t, "$highlight", HighlightFieldName)
	assert.True(t, arrow.TypeEqual(expected, HighlightArrowType()))
	assert.True(t, IsHighlightArrowType(expected))
	assert.False(t, IsHighlightArrowType(nil))

	assert.False(t, IsHighlightArrowType(arrow.ListOf(arrow.StructOf(
		arrow.Field{Name: "field_name", Type: arrow.BinaryTypes.String, Nullable: false},
		arrow.Field{Name: "fragments", Type: arrow.ListOfNonNullable(arrow.BinaryTypes.String), Nullable: false},
		arrow.Field{Name: "scores", Type: arrow.ListOfNonNullable(arrow.PrimitiveTypes.Float32), Nullable: false},
	))))
	assert.False(t, IsHighlightArrowType(arrow.ListOfNonNullable(arrow.StructOf(
		arrow.Field{Name: "field_name", Type: arrow.BinaryTypes.LargeString, Nullable: false},
		arrow.Field{Name: "fragments", Type: arrow.ListOfNonNullable(arrow.BinaryTypes.String), Nullable: false},
		arrow.Field{Name: "scores", Type: arrow.ListOfNonNullable(arrow.PrimitiveTypes.Float32), Nullable: false},
	))))
	assert.False(t, IsHighlightArrowType(arrow.ListOfNonNullable(arrow.StructOf(
		arrow.Field{Name: "field_name", Type: arrow.BinaryTypes.String, Nullable: true},
		arrow.Field{Name: "fragments", Type: arrow.ListOfNonNullable(arrow.BinaryTypes.String), Nullable: false},
		arrow.Field{Name: "scores", Type: arrow.ListOfNonNullable(arrow.PrimitiveTypes.Float32), Nullable: false},
	))))
}
