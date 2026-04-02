package tasks

import (
	"fmt"

	"github.com/apache/arrow/go/v17/arrow/memory"
	"go.uber.org/zap"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/querynodev2/segments"
	"github.com/milvus-io/milvus/internal/util/function/chain"
	"github.com/milvus-io/milvus/internal/util/segcore"
	"github.com/milvus-io/milvus/pkg/v2/log"
	"github.com/milvus-io/milvus/pkg/v2/metrics"
	"github.com/milvus-io/milvus/pkg/v2/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v2/util/merr"
	"github.com/milvus-io/milvus/pkg/v2/util/timerecord"
)

// exportSearchResultsAsArrow exports per-segment SearchResults as Arrow DataFrames
// via the Arrow C Stream Interface (one RecordBatch per NQ).
// Each DataFrame contains $id, $score, $seg_offset columns plus any extra fields,
// with one chunk per NQ query.
// extraFieldIDs specifies additional fields to export (e.g., fields needed by L0 rerank).
// The caller is responsible for releasing the returned DataFrames.
func (t *SearchTask) exportSearchResultsAsArrow(
	results []*segments.SearchResult,
	plan *segcore.SearchPlan,
	extraFieldIDs []int64,
) (segDFs []*chain.DataFrame, retErr error) {
	segDFs = make([]*chain.DataFrame, 0, len(results))
	defer func() {
		if retErr != nil {
			for _, df := range segDFs {
				df.Release()
			}
		}
	}()

	for _, res := range results {
		reader, err := segcore.ExportSearchResultAsArrowStream(res, plan, extraFieldIDs)
		if err != nil {
			log.Ctx(t.ctx).Warn("failed to export search result as Arrow", zap.Error(err))
			return nil, err
		}

		df, err := dataFrameFromArrowReader(reader)
		if err != nil {
			return nil, err
		}
		segDFs = append(segDFs, df)
	}
	return segDFs, nil
}

// executeGoReduce performs the search reduce pipeline entirely in Go:
//  1. heapMergeReduce (k-way merge with PK dedup, optionally GroupBy-aware)
//  2. Late Materialization (read output fields from segments)
//  3. Marshal to SearchResultData proto
//
// segDFs are the per-segment DataFrames from exportSearchResultsAsArrow.
func (t *SearchTask) executeGoReduce(
	segDFs []*chain.DataFrame,
	results []*segments.SearchResult,
	searchReq *segcore.SearchRequest,
	metricType string,
	tr *timerecord.TimeRecorder,
	relatedDataSize int64,
) error {
	pool := memory.NewGoAllocator()
	plan := searchReq.Plan()
	topK := plan.GetTopK()

	// Group-by is enabled iff the C++ Arrow exporter emitted a $group_by column.
	var groupByOpts *groupByOptions
	if len(segDFs) > 0 && segDFs[0].HasColumn(groupByCol) && len(results) > 0 {
		groupByOpts = &groupByOptions{
			GroupByFieldName: groupByCol,
			GroupSize:        results[0].GetMetadata().GroupSize,
		}
	}

	reduceResult, err := heapMergeReduce(pool, segDFs, topK, groupByOpts)
	if err != nil {
		log.Ctx(t.ctx).Warn("failed to heapMergeReduce", zap.Error(err))
		return err
	}
	defer reduceResult.DF.Release()

	reduceLatency := tr.RecordSpan()
	metrics.QueryNodeReduceLatency.WithLabelValues(
		fmt.Sprint(t.GetNodeID()),
		metrics.SearchLabel,
		metrics.ReduceSegments,
		metrics.BatchReduce).
		Observe(float64(reduceLatency.Milliseconds()))

	nqOffset := 0
	for i := range t.originNqs {
		nq := int(t.originNqs[i])
		if err := t.buildSlicedResult(i, nqOffset, nq, reduceResult, results, plan, metricType, tr, relatedDataSize); err != nil {
			return err
		}
		nqOffset += nq
	}

	t.attributeStorageCost(results)
	return nil
}

// attributeStorageCost splits the total storage cost across sub-tasks
// proportionally to NQ. Must run AFTER every slice's Late Mat finishes —
// FillOutputFieldsOrdered accumulates bytes on the C++ SearchResult, so
// GetMetadata().StorageCost is only final after late mat completes.
func (t *SearchTask) attributeStorageCost(results []*segments.SearchResult) {
	var totalNq int64
	for _, n := range t.originNqs {
		totalNq += n
	}
	if totalNq == 0 {
		return
	}
	var totalCost segcore.StorageCost
	for _, r := range results {
		c := r.GetMetadata().StorageCost
		totalCost.ScannedRemoteBytes += c.ScannedRemoteBytes
		totalCost.ScannedTotalBytes += c.ScannedTotalBytes
	}
	for i, sliceNq := range t.originNqs {
		task := t.subTaskAt(i)
		ratio := float64(sliceNq) / float64(totalNq)
		task.result.ScannedRemoteBytes = int64(float64(totalCost.ScannedRemoteBytes) * ratio)
		task.result.ScannedTotalBytes = int64(float64(totalCost.ScannedTotalBytes) * ratio)
	}
}

// buildSlicedResult extracts the i-th sub-task's slice from the merged reduce result,
// runs Late Materialization, and assigns the serialized blob to that task.
// The sliced DataFrame is released here so peak memory stays at one slice.
func (t *SearchTask) buildSlicedResult(
	i, nqOffset, nq int,
	reduceResult *mergeResult,
	results []*segments.SearchResult,
	plan *segcore.SearchPlan,
	metricType string,
	tr *timerecord.TimeRecorder,
	relatedDataSize int64,
) error {
	sliceResult, err := extractSlice(reduceResult, nqOffset, nq)
	if err != nil {
		return err
	}
	if sliceResult != reduceResult && sliceResult.DF != nil {
		defer sliceResult.DF.Release()
	}

	searchResultData, err := marshalReduceResult(sliceResult)
	if err != nil {
		return err
	}

	if err := lateMaterializeOutputFields(results, plan, sliceResult.Sources, searchResultData); err != nil {
		return err
	}

	slicedBlob, err := proto.Marshal(searchResultData)
	if err != nil {
		return err
	}

	task := t.subTaskAt(i)
	task.result = &internalpb.SearchResults{
		Base: &commonpb.MsgBase{
			SourceID: t.GetNodeID(),
		},
		Status:         merr.Success(),
		MetricType:     metricType,
		NumQueries:     t.originNqs[i],
		TopK:           t.originTopks[i],
		SlicedBlob:     slicedBlob,
		SlicedOffset:   1,
		SlicedNumCount: 1,
		CostAggregation: &internalpb.CostAggregation{
			ServiceTime:          tr.ElapseSpan().Milliseconds(),
			TotalRelatedDataSize: relatedDataSize,
		},
	}
	return nil
}

// lateMaterializeOutputFields reads output fields from C++ segments in a single
// CGO call and assembles them into the final SearchResultData. C++ does the
// per-segment FillTargetEntry + MergeDataArray scatter + serialize.
func lateMaterializeOutputFields(
	results []*segments.SearchResult,
	plan *segcore.SearchPlan,
	sources [][]segmentSource,
	searchResultData *schemapb.SearchResultData,
) error {
	totalRows := 0
	for _, chunk := range sources {
		totalRows += len(chunk)
	}
	if totalRows == 0 {
		return nil
	}

	segIndices := make([]int32, totalRows)
	segOffsets := make([]int64, totalRows)
	pos := 0
	for _, chunk := range sources {
		for _, src := range chunk {
			segIndices[pos] = int32(src.InputIdx)
			segOffsets[pos] = src.SegOffset
			pos++
		}
	}

	protoBytes, err := segcore.FillOutputFieldsOrdered(results, plan, segIndices, segOffsets)
	if err != nil {
		return err
	}
	if len(protoBytes) == 0 {
		return nil
	}

	var fieldResult schemapb.SearchResultData
	if err := proto.Unmarshal(protoBytes, &fieldResult); err != nil {
		return err
	}
	searchResultData.FieldsData = fieldResult.FieldsData
	return nil
}

// extractSlice extracts a sub-range of NQ chunks from a mergeResult.
func extractSlice(result *mergeResult, nqOffset, nqCount int) (*mergeResult, error) {
	if nqCount == 0 {
		return &mergeResult{
			DF:      emptyDF(),
			Sources: nil,
		}, nil
	}

	totalChunks := result.DF.NumChunks()
	if nqOffset+nqCount > totalChunks {
		return nil, merr.WrapErrServiceInternal(
			fmt.Sprintf("extractSlice: nqOffset(%d)+nqCount(%d) > totalChunks(%d)",
				nqOffset, nqCount, totalChunks))
	}

	// Fast path for the single sub-task case: return the original mergeResult,
	// which the caller already releases. Caller checks identity to avoid double-release.
	if nqOffset == 0 && nqCount == totalChunks {
		return result, nil
	}

	allChunkSizes := result.DF.ChunkSizes()
	sliceChunkSizes := allChunkSizes[nqOffset : nqOffset+nqCount]

	builder := chain.NewDataFrameBuilder()
	defer builder.Release()
	builder.SetChunkSizes(sliceChunkSizes)

	for _, colName := range result.DF.ColumnNames() {
		col := result.DF.Column(colName)
		chunks := col.Chunks()
		sliceChunks := chunks[nqOffset : nqOffset+nqCount]
		for _, chunk := range sliceChunks {
			chunk.Retain()
		}
		if err := builder.AddColumnFromChunks(colName, sliceChunks); err != nil {
			return nil, err
		}
		builder.CopyFieldMetadata(result.DF, colName)
	}
	builder.CopyAllMetadata(result.DF)

	sliceSources := result.Sources[nqOffset : nqOffset+nqCount]

	return &mergeResult{
		DF:      builder.Build(),
		Sources: sliceSources,
	}, nil
}

// emptyDF creates an empty DataFrame for empty slices.
func emptyDF() *chain.DataFrame {
	builder := chain.NewDataFrameBuilder()
	builder.SetChunkSizes(nil)
	return builder.Build()
}
