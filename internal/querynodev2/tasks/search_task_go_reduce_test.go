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
	"fmt"
	"path/filepath"
	"testing"

	"github.com/apache/arrow/go/v17/arrow/array"
	"github.com/apache/arrow/go/v17/arrow/memory"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	mock_segcore "github.com/milvus-io/milvus/internal/mocks/util/mock_segcore"
	"github.com/milvus-io/milvus/internal/querynodev2/segments"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/util/function/chain"
	"github.com/milvus-io/milvus/internal/util/initcore"
	"github.com/milvus-io/milvus/internal/util/segcore"
	"github.com/milvus-io/milvus/pkg/v2/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v2/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v2/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v2/util/typeutil"
)

// testSegments holds pre-created segments and search results for testing.
type testSegments struct {
	manager       *segments.Manager
	collection    *segments.Collection
	segs          []segments.Segment
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
	for _, seg := range ts.segs {
		seg.Release(context.Background())
	}
	ts.manager.Collection.Unref(ts.collection.ID(), 1)
	ts.chunkManager.RemoveWithPrefix(context.Background(), ts.rootPath)
}

// setupOpts varies the search request shape across test cases.
// At most one of OutputFieldIDs / GroupByFieldID should be set.
type setupOpts struct {
	OutputFieldIDs []int64
	GroupByFieldID int64
	GroupSize      int64
}

func setupTestSegments(t *testing.T, numSegments int, msgLength int, nq, topK int64, opts setupOpts) *testSegments {
	t.Helper()
	require.False(t, opts.GroupByFieldID > 0 && len(opts.OutputFieldIDs) > 0,
		"setupOpts: OutputFieldIDs and GroupByFieldID are mutually exclusive")

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

	schema := mock_segcore.GenTestCollectionSchema("test-late-mat", schemapb.DataType_Int64, true)
	indexMeta := mock_segcore.GenTestIndexMeta(collectionID, schema)

	manager := segments.NewManager()
	manager.Collection.PutOrRef(collectionID, schema, indexMeta, &querypb.LoadMetaInfo{
		LoadType:     querypb.LoadType_LoadCollection,
		CollectionID: collectionID,
		PartitionIDs: []int64{partitionID},
	})
	collection := manager.Collection.Get(collectionID)

	ts := &testSegments{
		manager:      manager,
		collection:   collection,
		chunkManager: chunkManager,
		rootPath:     rootPath,
	}

	segIDs := make([]int64, numSegments)
	for i := 0; i < numSegments; i++ {
		segmentID := int64(i + 1)
		segIDs[i] = segmentID

		seg, err := segments.NewSegment(ctx, collection, manager.Segment,
			segments.SegmentTypeSealed, 0,
			&querypb.SegmentLoadInfo{
				SegmentID:     segmentID,
				CollectionID:  collectionID,
				PartitionID:   partitionID,
				NumOfRows:     int64(msgLength),
				InsertChannel: fmt.Sprintf("by-dev-rootcoord-dml_0_%dv0", collectionID),
				Level:         datapb.SegmentLevel_Legacy,
			})
		if err != nil {
			t.Fatal(err)
		}

		binlogs, _, err := mock_segcore.SaveBinLog(ctx,
			collectionID, partitionID, segmentID, msgLength, schema, chunkManager)
		if err != nil {
			t.Fatal(err)
		}

		// Transition segment state from OnlyMeta → DataLoaded so subsequent
		// operations see it as "loaded". LoadFieldData itself does not drive
		// this transition; production code goes through Loader.loadSegment,
		// which wraps the loads in StartLoadData() / guard.Done().
		guard, err := seg.(*segments.LocalSegment).StartLoadData()
		if err != nil {
			t.Fatal(err)
		}
		for _, binlog := range binlogs {
			if err := seg.(*segments.LocalSegment).LoadFieldData(ctx, binlog.FieldID, int64(msgLength), binlog); err != nil {
				t.Fatal(err)
			}
		}
		guard.Done(nil)

		manager.Segment.Put(ctx, segments.SegmentTypeSealed, seg)
		ts.segs = append(ts.segs, seg)
	}

	var searchReq *segcore.SearchRequest
	var err error
	switch {
	case opts.GroupByFieldID > 0:
		searchReq, err = mock_segcore.GenSearchPlanAndRequestsWithGroupBy(
			collection.GetCCollection(), segIDs, nq, topK, opts.GroupByFieldID, opts.GroupSize)
	case len(opts.OutputFieldIDs) > 0:
		searchReq, err = mock_segcore.GenSearchPlanAndRequestsWithOutputFields(
			collection.GetCCollection(), segIDs, nq, topK, opts.OutputFieldIDs)
	default:
		searchReq, err = mock_segcore.GenSearchPlanAndRequestsWithTopK(
			collection.GetCCollection(), segIDs, nq, topK)
	}
	if err != nil {
		t.Fatal(err)
	}
	ts.searchReq = searchReq

	for _, seg := range ts.segs {
		result, err := seg.Search(ctx, searchReq)
		if err != nil {
			t.Fatal(err)
		}
		ts.searchResults = append(ts.searchResults, result)
	}

	return ts
}

// runGoReducePipeline runs the same pipeline as executeGoReduce
// and returns the merged mergeResult and DataFrames (caller must release).
// Group-by is auto-detected from the first SearchResult, mirroring production.
func runGoReducePipeline(t *testing.T, ts *testSegments) (*mergeResult, []*chain.DataFrame) {
	t.Helper()

	pool := memory.NewGoAllocator()
	plan := ts.searchReq.Plan()
	topK := plan.GetTopK()

	segDFs := make([]*chain.DataFrame, 0, len(ts.searchResults))
	for _, res := range ts.searchResults {
		reader, err := segcore.ExportSearchResultAsArrowStream(res, plan, nil)
		require.NoError(t, err)

		df, err := dataFrameFromArrowReader(reader)
		require.NoError(t, err)
		segDFs = append(segDFs, df)
	}

	var groupByOpts *groupByOptions
	if len(segDFs) > 0 && segDFs[0].HasColumn(groupByCol) && len(ts.searchResults) > 0 {
		groupByOpts = &groupByOptions{
			GroupByFieldName: groupByCol,
			GroupSize:        ts.searchResults[0].GetMetadata().GroupSize,
		}
	}

	reduceResult, err := heapMergeReduce(pool, segDFs, topK, groupByOpts)
	require.NoError(t, err)

	return reduceResult, segDFs
}

func TestLateMaterializeOutputFields(t *testing.T) {
	// Field IDs from GenTestCollectionSchema (100+i):
	// 103=Int32, 104=Float (scalar fields easy to verify)
	outputFieldIDs := []int64{103, 104}

	ts := setupTestSegments(t, 2, 2000, 2, 10, setupOpts{OutputFieldIDs: outputFieldIDs})
	defer ts.cleanup()

	reduceResult, segDFs := runGoReducePipeline(t, ts)
	defer func() {
		reduceResult.DF.Release()
		for _, df := range segDFs {
			df.Release()
		}
	}()

	// Marshal base result (ids + scores only)
	searchResultData, err := marshalReduceResult(reduceResult)
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
	ts := setupTestSegments(t, 2, 2000, 1, 5, setupOpts{})
	defer ts.cleanup()

	reduceResult, segDFs := runGoReducePipeline(t, ts)
	defer func() {
		reduceResult.DF.Release()
		for _, df := range segDFs {
			df.Release()
		}
	}()

	searchResultData, err := marshalReduceResult(reduceResult)
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

	err = lateMaterializeOutputFields(nil, nil, [][]segmentSource{}, searchResultData)
	require.NoError(t, err)
	assert.Empty(t, searchResultData.FieldsData)
}

// TestGoReduceGroupBy verifies the end-to-end group-by reduce path: each NQ's
// results respect group_size across two segments.
//
// Uses the bool field (id 100) because mock data alternates true/false, giving
// exactly two well-populated buckets. With topK >= 2*group_size every bucket
// fills exactly to group_size, so the test asserts that strict invariant.
func TestGoReduceGroupBy(t *testing.T) {
	const (
		groupByFieldID int64 = 100 // Bool: 2 buckets, each well-populated
		groupSize      int64 = 3
		nq             int64 = 2
		topK           int64 = 10
	)

	ts := setupTestSegments(t, 2, 100, nq, topK,
		setupOpts{GroupByFieldID: groupByFieldID, GroupSize: groupSize})
	defer ts.cleanup()

	// Sanity: the C++ side should report group-by enabled on every result.
	for i, r := range ts.searchResults {
		md := r.GetMetadata()
		require.True(t, md.HasGroupBy, "result[%d] missing group_by_values_", i)
		require.Equal(t, groupSize, md.GroupSize, "result[%d] group_size mismatch", i)
	}

	reduceResult, segDFs := runGoReducePipeline(t, ts)
	defer func() {
		reduceResult.DF.Release()
		for _, df := range segDFs {
			df.Release()
		}
	}()

	// The merged DataFrame should carry the $group_by column propagated by
	// pickGroupByValues, with one chunk per NQ.
	gbCol := reduceResult.DF.Column(groupByCol)
	require.NotNil(t, gbCol, "merged DataFrame missing %s column", groupByCol)
	require.Equal(t, int(nq), reduceResult.DF.NumChunks(),
		"expected one chunk per NQ")

	for nqIdx := 0; nqIdx < int(nq); nqIdx++ {
		chunk := gbCol.Chunk(nqIdx)
		boolArr, ok := chunk.(*array.Boolean)
		require.True(t, ok, "expected $group_by chunk[%d] to be Boolean, got %T", nqIdx, chunk)

		counts := make(map[bool]int)
		for i := 0; i < boolArr.Len(); i++ {
			require.False(t, boolArr.IsNull(i),
				"NQ %d row %d: bool group_by should not be null", nqIdx, i)
			v := boolArr.Value(i)
			counts[v]++
			require.LessOrEqual(t, int64(counts[v]), groupSize,
				"NQ %d: group %v exceeded group_size (%d > %d)",
				nqIdx, v, counts[v], groupSize)
		}

		// Both buckets are well-populated and group_size×buckets (6) ≤ topK (10),
		// so each bucket should fill to exactly group_size after merge.
		require.Equal(t, 2, len(counts),
			"NQ %d: expected both true and false buckets", nqIdx)
		require.Equal(t, int(groupSize), counts[true],
			"NQ %d: true bucket should fill to group_size", nqIdx)
		require.Equal(t, int(groupSize), counts[false],
			"NQ %d: false bucket should fill to group_size", nqIdx)
		t.Logf("NQ %d: %d rows, true=%d false=%d",
			nqIdx, boolArr.Len(), counts[true], counts[false])
	}
}
