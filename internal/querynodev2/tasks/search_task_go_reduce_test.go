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

package tasks

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/apache/arrow/go/v17/arrow/memory"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	mock_segcore "github.com/milvus-io/milvus/internal/mocks/util/mock_segcore"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/util/function/chain"
	"github.com/milvus-io/milvus/internal/util/initcore"
	"github.com/milvus-io/milvus/internal/util/segcore"
	"github.com/milvus-io/milvus/pkg/v2/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v2/util/typeutil"
)

// testSegments holds pre-created segments and search results for testing.
type testSegments struct {
	collection    *segcore.CCollection
	segments      []segcore.CSegment
	searchResults []*segcore.SearchResult
	searchReq     *segcore.SearchRequest
	chunkManager  storage.ChunkManager
	rootPath      string
}

func (ts *testSegments) cleanup() {
	for _, r := range ts.searchResults {
		r.Release()
	}
	ts.searchReq.Delete()
	for _, seg := range ts.segments {
		seg.Release()
	}
	ts.collection.Release()
	ts.chunkManager.RemoveWithPrefix(context.Background(), ts.rootPath)
}

func setupTestSegments(t *testing.T, numSegments int, msgLength int, nq, topK int64, outputFieldIDs []int64) *testSegments {
	t.Helper()

	paramtable.Init()

	localDataRootPath := filepath.Join(paramtable.Get().LocalStorageCfg.Path.GetValue(), typeutil.QueryNodeRole)
	initcore.InitLocalChunkManager(localDataRootPath)
	if err := initcore.InitMmapManager(paramtable.Get(), 1); err != nil {
		t.Fatal(err)
	}
	if err := initcore.InitTieredStorage(paramtable.Get()); err != nil {
		t.Fatal(err)
	}

	ctx := context.Background()
	rootPath := t.Name()
	chunkManagerFactory := storage.NewTestChunkManagerFactory(paramtable.Get(), rootPath)
	chunkManager, _ := chunkManagerFactory.NewPersistentStorageChunkManager(ctx)
	initcore.InitRemoteChunkManager(paramtable.Get())

	collectionID := int64(100)
	partitionID := int64(10)

	schema := mock_segcore.GenTestCollectionSchema("test-late-mat", schemapb.DataType_Int64, false)
	collection, err := segcore.CreateCCollection(&segcore.CreateCCollectionRequest{
		CollectionID: collectionID,
		Schema:       schema,
		IndexMeta:    mock_segcore.GenTestIndexMeta(collectionID, schema),
	})
	if err != nil {
		t.Fatal(err)
	}

	ts := &testSegments{
		collection:   collection,
		chunkManager: chunkManager,
		rootPath:     rootPath,
	}

	segIDs := make([]int64, numSegments)
	for i := 0; i < numSegments; i++ {
		segmentID := int64(i + 1)
		segIDs[i] = segmentID

		seg, err := segcore.CreateCSegment(&segcore.CreateCSegmentRequest{
			Collection:  collection,
			SegmentID:   segmentID,
			SegmentType: segcore.SegmentTypeSealed,
			IsSorted:    false,
		})
		if err != nil {
			t.Fatal(err)
		}

		binlogs, _, err := mock_segcore.SaveBinLog(ctx,
			collectionID, partitionID, segmentID, msgLength, schema, chunkManager)
		if err != nil {
			t.Fatal(err)
		}

		req := &segcore.LoadFieldDataRequest{RowCount: int64(msgLength)}
		for _, binlog := range binlogs {
			req.Fields = append(req.Fields, segcore.LoadFieldDataInfo{Field: binlog})
		}
		if _, err := seg.LoadFieldData(ctx, req); err != nil {
			t.Fatal(err)
		}

		ts.segments = append(ts.segments, seg)
	}

	var searchReq *segcore.SearchRequest
	if len(outputFieldIDs) > 0 {
		searchReq, err = mock_segcore.GenSearchPlanAndRequestsWithOutputFields(collection, segIDs, nq, topK, outputFieldIDs)
	} else {
		searchReq, err = mock_segcore.GenSearchPlanAndRequestsWithTopK(collection, segIDs, nq, topK)
	}
	if err != nil {
		t.Fatal(err)
	}
	ts.searchReq = searchReq

	for _, seg := range ts.segments {
		result, err := seg.Search(ctx, searchReq)
		if err != nil {
			t.Fatal(err)
		}
		ts.searchResults = append(ts.searchResults, result)
	}

	return ts
}

// runGoReducePipeline runs the same pipeline as executeGoReduce Steps 1-2
// and returns the merged ReduceResult and DataFrames (caller must release).
func runGoReducePipeline(t *testing.T, ts *testSegments) (*ReduceResult, []*chain.DataFrame) {
	t.Helper()

	pool := memory.NewGoAllocator()
	plan := ts.searchReq.Plan()
	topK := plan.GetTopK()

	segDFs := make([]*chain.DataFrame, 0, len(ts.searchResults))
	for _, res := range ts.searchResults {
		record, err := segcore.ExportSearchResultAsArrow(res, plan, nil)
		require.NoError(t, err)

		topkPerNQ, err := segcore.ParseTopkPerNQ(record.Schema())
		require.NoError(t, err)

		df, err := DataFrameFromArrowRecord(record, topkPerNQ)
		record.Release()
		require.NoError(t, err)
		segDFs = append(segDFs, df)
	}

	reduceResult, err := HeapMergeReduce(pool, segDFs, topK, nil)
	require.NoError(t, err)

	return reduceResult, segDFs
}

func TestLateMaterializeOutputFields(t *testing.T) {
	// Field IDs from GenTestCollectionSchema (100+i):
	// 103=Int32, 104=Float (scalar fields easy to verify)
	outputFieldIDs := []int64{103, 104}

	ts := setupTestSegments(t, 2, 2000, 2, 10, outputFieldIDs)
	defer ts.cleanup()

	reduceResult, segDFs := runGoReducePipeline(t, ts)
	defer func() {
		reduceResult.DF.Release()
		for _, df := range segDFs {
			df.Release()
		}
	}()

	// Marshal base result (ids + scores only)
	searchResultData, err := MarshalReduceResult(reduceResult)
	require.NoError(t, err)

	// Before Late Mat: no field data
	assert.Empty(t, searchResultData.FieldsData)

	// Run Late Materialization
	plan := ts.searchReq.Plan()
	err = lateMaterializeOutputFields(ts.searchResults, plan, reduceResult.Sources, searchResultData)
	require.NoError(t, err)

	// After Late Mat: should have field data for each output field
	require.Len(t, searchResultData.FieldsData, len(outputFieldIDs),
		"should have one FieldData entry per output field")

	totalRows := int(reduceResult.DF.NumRows())
	require.Greater(t, totalRows, 0, "should have some results")

	// Verify each field has correct number of rows
	for _, fd := range searchResultData.FieldsData {
		switch f := fd.Field.(type) {
		case *schemapb.FieldData_Scalars:
			switch d := f.Scalars.Data.(type) {
			case *schemapb.ScalarField_IntData:
				assert.Len(t, d.IntData.Data, totalRows,
					"int field %s row count mismatch", fd.FieldName)
			case *schemapb.ScalarField_FloatData:
				assert.Len(t, d.FloatData.Data, totalRows,
					"float field %s row count mismatch", fd.FieldName)
			default:
				t.Errorf("unexpected scalar type for field %s", fd.FieldName)
			}
		default:
			t.Errorf("expected scalar field, got %T for field %s", fd.Field, fd.FieldName)
		}
	}

	// Verify IDs and scores also match total rows
	assert.Len(t, searchResultData.Scores, totalRows)

	t.Logf("Late Mat OK: %d output fields, %d total rows", len(searchResultData.FieldsData), totalRows)
}

func TestLateMaterializeOutputFields_NoOutputFields(t *testing.T) {
	// No output fields in the plan
	ts := setupTestSegments(t, 2, 2000, 1, 5, nil)
	defer ts.cleanup()

	reduceResult, segDFs := runGoReducePipeline(t, ts)
	defer func() {
		reduceResult.DF.Release()
		for _, df := range segDFs {
			df.Release()
		}
	}()

	searchResultData, err := MarshalReduceResult(reduceResult)
	require.NoError(t, err)

	plan := ts.searchReq.Plan()
	err = lateMaterializeOutputFields(ts.searchResults, plan, reduceResult.Sources, searchResultData)
	require.NoError(t, err)

	// No output fields → FieldsData remains empty
	assert.Empty(t, searchResultData.FieldsData)
}

func TestLateMaterializeOutputFields_EmptySources(t *testing.T) {
	// Empty sources should be a no-op
	searchResultData := &schemapb.SearchResultData{}
	err := lateMaterializeOutputFields(nil, nil, nil, searchResultData)
	require.NoError(t, err)
	assert.Empty(t, searchResultData.FieldsData)

	err = lateMaterializeOutputFields(nil, nil, [][]SegmentSource{}, searchResultData)
	require.NoError(t, err)
	assert.Empty(t, searchResultData.FieldsData)
}
