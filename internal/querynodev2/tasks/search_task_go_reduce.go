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

// exportSearchResultsAsArrow exports per-segment SearchResults as Arrow DataFrames.
// Each DataFrame contains $id, $score, and $seg_offset columns, split into NQ chunks.
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
		record, err := segcore.ExportSearchResultAsArrow(res, plan, extraFieldIDs)
		if err != nil {
			log.Ctx(t.ctx).Warn("failed to export search result as Arrow", zap.Error(err))
			return nil, err
		}

		topkPerNQ, err := segcore.ParseTopkPerNQ(record.Schema())
		if err != nil {
			record.Release()
			return nil, err
		}

		df, err := DataFrameFromArrowRecord(record, topkPerNQ)
		// DataFrame slices retain their own Arrow references, release the record now
		record.Release()
		if err != nil {
			return nil, err
		}
		segDFs = append(segDFs, df)
	}
	return segDFs, nil
}

// executeGoReduce performs the search reduce pipeline entirely in Go:
//  1. HeapMergeReduce (k-way merge with PK dedup)
//  2. Late Materialization (read output fields from segments)
//  3. Marshal to SearchResultData proto
//
// segDFs are the per-segment DataFrames from exportSearchResultsAsArrow.
// This replaces the C++ ReduceSearchResultsAndFillData path.
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

	// Step 1: HeapMergeReduce
	reduceResult, err := HeapMergeReduce(pool, segDFs, topK, nil)
	if err != nil {
		log.Ctx(t.ctx).Warn("failed to HeapMergeReduce", zap.Error(err))
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

	// Step 3: Marshal reduce result into SearchResultData slices
	// For each originNq/originTopk pair, produce a SlicedBlob
	nqOffset := 0
	for i := range t.originNqs {
		nq := int(t.originNqs[i])

		// Extract the slice's portion of the reduce result
		sliceResult, err := extractSlice(reduceResult, nqOffset, nq)
		if err != nil {
			return err
		}

		searchResultData, err := MarshalReduceResult(sliceResult)
		if err != nil {
			return err
		}

		// Late Materialization: read output fields from segments
		if err := lateMaterializeOutputFields(results, plan, sliceResult.Sources, searchResultData); err != nil {
			return err
		}

		var task *SearchTask
		if i == 0 {
			task = t
		} else {
			task = t.others[i-1]
		}

		slicedBlob, err := proto.Marshal(searchResultData)
		if err != nil {
			return err
		}

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

		nqOffset += nq
	}

	return nil
}

// lateMaterializeOutputFields reads output fields from C++ segments and
// assembles them into the final SearchResultData in a single CGO call.
// C++ handles the per-segment FillTargetEntry + MergeDataArray scatter
// + serialization, avoiding N separate CGO calls and Go-side scatter.
func lateMaterializeOutputFields(
	results []*segments.SearchResult,
	plan *segcore.SearchPlan,
	sources [][]SegmentSource,
	searchResultData *schemapb.SearchResultData,
) error {
	totalRows := 0
	for _, chunk := range sources {
		totalRows += len(chunk)
	}
	if totalRows == 0 {
		return nil
	}

	// Flatten sources into ordered arrays for the C++ API
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

	// Single CGO call: C++ does FillTargetEntry + MergeDataArray + serialize
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

// extractSlice extracts a sub-range of NQ chunks from a ReduceResult.
func extractSlice(result *ReduceResult, nqOffset, nqCount int) (*ReduceResult, error) {
	if nqCount == 0 {
		return &ReduceResult{
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

	// If the full result is just this one slice, return as-is
	if nqOffset == 0 && nqCount == totalChunks {
		return result, nil
	}

	// Build a new DataFrame from the chunk range [nqOffset, nqOffset+nqCount)
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
	}

	sliceSources := result.Sources[nqOffset : nqOffset+nqCount]

	return &ReduceResult{
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
