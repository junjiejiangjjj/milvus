/*
 * # Licensed to the LF AI & Data foundation under one
 * # or more contributor license agreements. See the NOTICE file
 * # distributed with this work for additional information
 * # regarding copyright ownership. The ASF licenses this file
 * # to you under the Apache License, Version 2.0 (the
 * # "License"); you may not use this file except in compliance
 * # with the License. You may obtain a copy of the License at
 * #
 * #     http://www.apache.org/licenses/LICENSE-2.0
 * #
 * # Unless required by applicable law or agreed to in writing, software
 * # distributed under the License is distributed on an "AS IS" BASIS,
 * # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * # See the License for the specific language governing permissions and
 * # limitations under the License.
 */

package chain

/*

import (
	"testing"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/stretchr/testify/assert"
)

func TestFromSearchResultProto(t *testing.T) {
	// Create a simple search result
	searchResults := &milvuspb.SearchResults{
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
		},
		Results: &schemapb.SearchResultData{
			NumQueries: 2,
			TopK:       3,
			Topks:      []int64{2, 3}, // First query has 2 results, second query has 3 results
			Ids: &schemapb.IDs{
				IdField: &schemapb.IDs_IntId{
					IntId: &schemapb.LongArray{
						Data: []int64{1, 2, 3, 4, 5},
					},
				},
			},
			Scores: []float32{0.9, 0.8, 0.95, 0.85, 0.75},
			FieldsData: []*schemapb.FieldData{
				{
					Type:      schemapb.DataType_Int64,
					FieldName: "age",
					Field: &schemapb.FieldData_Scalars{
						Scalars: &schemapb.ScalarField{
							Data: &schemapb.ScalarField_LongData{
								LongData: &schemapb.LongArray{
									Data: []int64{25, 30, 35, 40, 45},
								},
							},
						},
					},
				},
				{
					Type:      schemapb.DataType_VarChar,
					FieldName: "name",
					Field: &schemapb.FieldData_Scalars{
						Scalars: &schemapb.ScalarField{
							Data: &schemapb.ScalarField_StringData{
								StringData: &schemapb.StringArray{
									Data: []string{"Alice", "Bob", "Charlie", "David", "Eve"},
								},
							},
						},
					},
				},
			},
		},
	}

	df, err := FromSearchResultProto(searchResults)
	assert.NoError(t, err)
	assert.NotNil(t, df)
	assert.NotNil(t, df.table)

	// Verify the table has correct number of rows and columns
	table := *df.table
	assert.Equal(t, int64(5), table.NumRows())
	assert.Equal(t, int64(4), table.NumCols()) // id, score, age, name

	// Verify schema
	schema := table.Schema()
	assert.Equal(t, 4, len(schema.Fields()))
	assert.Equal(t, "id", schema.Field(0).Name)
	assert.Equal(t, "score", schema.Field(1).Name)
	assert.Equal(t, "age", schema.Field(2).Name)
	assert.Equal(t, "name", schema.Field(3).Name)
}

func TestFromSearchResultProtoWithStringIDs(t *testing.T) {
	// Create a search result with string IDs
	searchResults := &milvuspb.SearchResults{
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
		},
		Results: &schemapb.SearchResultData{
			NumQueries: 1,
			TopK:       3,
			Topks:      []int64{3},
			Ids: &schemapb.IDs{
				IdField: &schemapb.IDs_StrId{
					StrId: &schemapb.StringArray{
						Data: []string{"id1", "id2", "id3"},
					},
				},
			},
			Scores: []float32{0.9, 0.8, 0.7},
			FieldsData: []*schemapb.FieldData{
				{
					Type:      schemapb.DataType_Float,
					FieldName: "price",
					Field: &schemapb.FieldData_Scalars{
						Scalars: &schemapb.ScalarField{
							Data: &schemapb.ScalarField_FloatData{
								FloatData: &schemapb.FloatArray{
									Data: []float32{19.99, 29.99, 39.99},
								},
							},
						},
					},
				},
			},
		},
	}

	df, err := FromSearchResultProto(searchResults)
	assert.NoError(t, err)
	assert.NotNil(t, df)
	assert.NotNil(t, df.table)

	// Verify the table
	table := *df.table
	assert.Equal(t, int64(3), table.NumRows())
	assert.Equal(t, int64(3), table.NumCols()) // id, score, price
}

func TestFromSearchResultProtoNilInput(t *testing.T) {
	// Test nil input
	df, err := FromSearchResultProto(nil)
	assert.Error(t, err)
	assert.Nil(t, df)
	assert.Contains(t, err.Error(), "search data is nil")
}

func TestFromSearchResultProtoEmptyTopks(t *testing.T) {
	// Test empty topks
	searchResults := &milvuspb.SearchResults{
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
		},
		Results: &schemapb.SearchResultData{
			NumQueries: 0,
			TopK:       0,
			Topks:      []int64{},
		},
	}

	df, err := FromSearchResultProto(searchResults)
	assert.Error(t, err)
	assert.Nil(t, df)
	assert.Contains(t, err.Error(), "topks is empty")
}

*/
