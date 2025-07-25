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

package datacoord

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/cockroachdb/errors"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus/internal/datacoord/allocator"
	"github.com/milvus-io/milvus/internal/datacoord/task"
	"github.com/milvus-io/milvus/pkg/v2/log"
	"github.com/milvus-io/milvus/pkg/v2/metrics"
	"github.com/milvus-io/milvus/pkg/v2/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v2/util/conc"
	"github.com/milvus-io/milvus/pkg/v2/util/lock"
	"github.com/milvus-io/milvus/pkg/v2/util/merr"
	"github.com/milvus-io/milvus/pkg/v2/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v2/util/typeutil"
)

// TODO: we just warn about the long executing/queuing tasks
// need to get rid of long queuing tasks because the compaction tasks are local optimum.
var maxCompactionTaskExecutionDuration = map[datapb.CompactionType]time.Duration{
	datapb.CompactionType_MixCompaction:          30 * time.Minute,
	datapb.CompactionType_Level0DeleteCompaction: 30 * time.Minute,
	datapb.CompactionType_ClusteringCompaction:   60 * time.Minute,
	datapb.CompactionType_SortCompaction:         20 * time.Minute,
}

type CompactionInspector interface {
	start()
	stop()
	// enqueueCompaction start to enqueue compaction task and return immediately
	enqueueCompaction(task *datapb.CompactionTask) error
	// isFull return true if the task pool is full
	isFull() bool
	// get compaction tasks by signal id
	getCompactionTasksNumBySignalID(signalID int64) int
	getCompactionInfo(ctx context.Context, signalID int64) *compactionInfo
	removeTasksByChannel(channel string)
	getCompactionTasksNum(filters ...compactionTaskFilter) int
}

var (
	errChannelNotWatched = errors.New("channel is not watched")
	errChannelInBuffer   = errors.New("channel is in buffer")
)

var _ CompactionInspector = (*compactionInspector)(nil)

type compactionInfo struct {
	state        commonpb.CompactionState
	executingCnt int
	completedCnt int
	failedCnt    int
	timeoutCnt   int
	mergeInfos   map[int64]*milvuspb.CompactionMergeInfo
}

type compactionInspector struct {
	queueTasks *CompactionQueue

	executingGuard lock.RWMutex
	executingTasks map[int64]CompactionTask // planID -> task

	cleaningGuard lock.RWMutex
	cleaningTasks map[int64]CompactionTask // planID -> task

	meta             CompactionMeta
	allocator        allocator.Allocator
	cluster          Cluster
	analyzeScheduler task.GlobalScheduler
	handler          Handler
	scheduler        task.GlobalScheduler
	ievm             IndexEngineVersionManager

	stopCh   chan struct{}
	stopOnce sync.Once
	stopWg   sync.WaitGroup
}

func (c *compactionInspector) getCompactionInfo(ctx context.Context, triggerID int64) *compactionInfo {
	tasks := c.meta.GetCompactionTasksByTriggerID(ctx, triggerID)
	return summaryCompactionState(triggerID, tasks)
}

func summaryCompactionState(triggerID int64, tasks []*datapb.CompactionTask) *compactionInfo {
	ret := &compactionInfo{}
	var executingCnt, pipeliningCnt, completedCnt, failedCnt, timeoutCnt, analyzingCnt, indexingCnt, cleanedCnt, metaSavedCnt, stats int
	mergeInfos := make(map[int64]*milvuspb.CompactionMergeInfo)

	for _, task := range tasks {
		if task == nil {
			continue
		}
		switch task.GetState() {
		case datapb.CompactionTaskState_executing:
			executingCnt++
		case datapb.CompactionTaskState_pipelining:
			pipeliningCnt++
		case datapb.CompactionTaskState_completed:
			completedCnt++
		case datapb.CompactionTaskState_failed:
			failedCnt++
		case datapb.CompactionTaskState_timeout:
			timeoutCnt++
		case datapb.CompactionTaskState_analyzing:
			analyzingCnt++
		case datapb.CompactionTaskState_indexing:
			indexingCnt++
		case datapb.CompactionTaskState_cleaned:
			cleanedCnt++
		case datapb.CompactionTaskState_meta_saved:
			metaSavedCnt++
		case datapb.CompactionTaskState_statistic:
			stats++
		default:
		}
		mergeInfos[task.GetPlanID()] = getCompactionMergeInfo(task)
	}

	ret.executingCnt = executingCnt + pipeliningCnt + analyzingCnt + indexingCnt + metaSavedCnt + stats
	ret.completedCnt = completedCnt
	ret.timeoutCnt = timeoutCnt
	ret.failedCnt = failedCnt
	ret.mergeInfos = mergeInfos

	if ret.executingCnt != 0 {
		ret.state = commonpb.CompactionState_Executing
	} else {
		ret.state = commonpb.CompactionState_Completed
	}

	log.Info("compaction states",
		zap.Int64("triggerID", triggerID),
		zap.String("state", ret.state.String()),
		zap.Int("executingCnt", executingCnt),
		zap.Int("pipeliningCnt", pipeliningCnt),
		zap.Int("completedCnt", completedCnt),
		zap.Int("failedCnt", failedCnt),
		zap.Int("timeoutCnt", timeoutCnt),
		zap.Int("analyzingCnt", analyzingCnt),
		zap.Int("indexingCnt", indexingCnt),
		zap.Int("cleanedCnt", cleanedCnt),
		zap.Int("metaSavedCnt", metaSavedCnt))
	return ret
}

func (c *compactionInspector) getCompactionTasksNumBySignalID(triggerID int64) int {
	cnt := 0
	c.queueTasks.ForEach(func(ct CompactionTask) {
		if ct.GetTaskProto().GetTriggerID() == triggerID {
			cnt += 1
		}
	})
	c.executingGuard.RLock()
	for _, t := range c.executingTasks {
		if t.GetTaskProto().GetTriggerID() == triggerID {
			cnt += 1
		}
	}
	c.executingGuard.RUnlock()
	return cnt
}

func newCompactionInspector(meta CompactionMeta,
	allocator allocator.Allocator, handler Handler, scheduler task.GlobalScheduler, ievm IndexEngineVersionManager,
) *compactionInspector {
	// Higher capacity will have better ordering in priority, but consumes more memory.
	// TODO[GOOSE]: Higher capacity makes tasks waiting longer, which need to be get rid of.
	capacity := paramtable.Get().DataCoordCfg.CompactionTaskQueueCapacity.GetAsInt()
	return &compactionInspector{
		queueTasks:     NewCompactionQueue(capacity, getPrioritizer()),
		meta:           meta,
		allocator:      allocator,
		stopCh:         make(chan struct{}),
		executingTasks: make(map[int64]CompactionTask),
		cleaningTasks:  make(map[int64]CompactionTask),
		handler:        handler,
		scheduler:      scheduler,
		ievm:           ievm,
	}
}

func (c *compactionInspector) checkSchedule() {
	err := c.checkCompaction()
	if err != nil {
		log.Info("fail to update compaction", zap.Error(err))
	}
	c.cleanFailedTasks()
	c.schedule()
}

func (c *compactionInspector) schedule() []CompactionTask {
	selected := make([]CompactionTask, 0)
	if c.queueTasks.Len() == 0 {
		return selected
	}

	l0ChannelExcludes := typeutil.NewSet[string]()
	mixChannelExcludes := typeutil.NewSet[string]()
	clusterChannelExcludes := typeutil.NewSet[string]()
	mixLabelExcludes := typeutil.NewSet[string]()
	clusterLabelExcludes := typeutil.NewSet[string]()

	c.executingGuard.RLock()
	for _, t := range c.executingTasks {
		switch t.GetTaskProto().GetType() {
		case datapb.CompactionType_Level0DeleteCompaction:
			l0ChannelExcludes.Insert(t.GetTaskProto().GetChannel())
		case datapb.CompactionType_MixCompaction, datapb.CompactionType_SortCompaction:
			mixChannelExcludes.Insert(t.GetTaskProto().GetChannel())
			mixLabelExcludes.Insert(t.GetLabel())
		case datapb.CompactionType_ClusteringCompaction:
			clusterChannelExcludes.Insert(t.GetTaskProto().GetChannel())
			clusterLabelExcludes.Insert(t.GetLabel())
		}
	}
	c.executingGuard.RUnlock()

	excluded := make([]CompactionTask, 0)
	defer func() {
		// Add back the excluded tasks
		for _, t := range excluded {
			c.queueTasks.Enqueue(t)
		}
	}()

	p := getPrioritizer()
	if &c.queueTasks.prioritizer != &p {
		c.queueTasks.UpdatePrioritizer(p)
	}

	// The schedule loop will stop if either:
	// 1. no more task to schedule (the task queue is empty)
	// 2. no avaiable slots
	for {
		t, err := c.queueTasks.Dequeue()
		if err != nil {
			break // 1. no more task to schedule
		}

		switch t.GetTaskProto().GetType() {
		case datapb.CompactionType_Level0DeleteCompaction:
			if mixChannelExcludes.Contain(t.GetTaskProto().GetChannel()) ||
				clusterChannelExcludes.Contain(t.GetTaskProto().GetChannel()) {
				excluded = append(excluded, t)
				continue
			}
			l0ChannelExcludes.Insert(t.GetTaskProto().GetChannel())
			selected = append(selected, t)
		case datapb.CompactionType_MixCompaction, datapb.CompactionType_SortCompaction:
			if l0ChannelExcludes.Contain(t.GetTaskProto().GetChannel()) {
				excluded = append(excluded, t)
				continue
			}
			mixChannelExcludes.Insert(t.GetTaskProto().GetChannel())
			mixLabelExcludes.Insert(t.GetLabel())
			selected = append(selected, t)
		case datapb.CompactionType_ClusteringCompaction:
			if l0ChannelExcludes.Contain(t.GetTaskProto().GetChannel()) ||
				mixLabelExcludes.Contain(t.GetLabel()) ||
				clusterLabelExcludes.Contain(t.GetLabel()) {
				excluded = append(excluded, t)
				continue
			}
			clusterChannelExcludes.Insert(t.GetTaskProto().GetChannel())
			clusterLabelExcludes.Insert(t.GetLabel())
			selected = append(selected, t)
		}

		c.executingGuard.Lock()
		c.executingTasks[t.GetTaskProto().GetPlanID()] = t
		c.scheduler.Enqueue(t)
		c.executingGuard.Unlock()
		metrics.DataCoordCompactionTaskNum.WithLabelValues(fmt.Sprintf("%d", NullNodeID), t.GetTaskProto().GetType().String(), metrics.Pending).Dec()
		metrics.DataCoordCompactionTaskNum.WithLabelValues(fmt.Sprintf("%d", t.GetTaskProto().GetNodeID()), t.GetTaskProto().GetType().String(), metrics.Executing).Inc()
	}
	return selected
}

func (c *compactionInspector) start() {
	c.stopWg.Add(2)
	go c.loopSchedule()
	go c.loopClean()
}

func (c *compactionInspector) loadMeta() {
	// TODO: make it compatible to all types of compaction with persist meta
	triggers := c.meta.GetCompactionTasks(context.TODO())
	for _, tasks := range triggers {
		for _, task := range tasks {
			state := task.GetState()
			if state == datapb.CompactionTaskState_completed ||
				state == datapb.CompactionTaskState_cleaned ||
				state == datapb.CompactionTaskState_timeout ||
				state == datapb.CompactionTaskState_unknown {
				log.Info("compactionInspector loadMeta abandon compactionTask",
					zap.Int64("planID", task.GetPlanID()),
					zap.String("type", task.GetType().String()),
					zap.String("state", task.GetState().String()))
				continue
			} else {
				t, err := c.createCompactTask(task)
				if err != nil {
					log.Info("compactionInspector loadMeta create compactionTask failed, try to clean it",
						zap.Int64("planID", task.GetPlanID()),
						zap.String("type", task.GetType().String()),
						zap.String("state", task.GetState().String()),
						zap.Error(err),
					)
					// ignore the drop error
					c.meta.DropCompactionTask(context.TODO(), task)
					continue
				}
				if t.NeedReAssignNodeID() {
					if err = c.submitTask(t); err != nil {
						log.Info("compactionInspector loadMeta submit task failed, try to clean it",
							zap.Int64("planID", task.GetPlanID()),
							zap.String("type", task.GetType().String()),
							zap.String("state", task.GetState().String()),
							zap.Error(err),
						)
						// ignore the drop error
						c.meta.DropCompactionTask(context.Background(), task)
						continue
					}
					log.Info("compactionInspector loadMeta submitTask",
						zap.Int64("planID", t.GetTaskProto().GetPlanID()),
						zap.Int64("triggerID", t.GetTaskProto().GetTriggerID()),
						zap.Int64("collectionID", t.GetTaskProto().GetCollectionID()),
						zap.String("type", task.GetType().String()),
						zap.String("state", t.GetTaskProto().GetState().String()))
				} else {
					c.restoreTask(t)
					log.Info("compactionInspector loadMeta restoreTask",
						zap.Int64("planID", t.GetTaskProto().GetPlanID()),
						zap.Int64("triggerID", t.GetTaskProto().GetTriggerID()),
						zap.Int64("collectionID", t.GetTaskProto().GetCollectionID()),
						zap.String("type", task.GetType().String()),
						zap.String("state", t.GetTaskProto().GetState().String()))
				}
			}
		}
	}
}

func (c *compactionInspector) loopSchedule() {
	interval := paramtable.Get().DataCoordCfg.CompactionScheduleInterval.GetAsDuration(time.Millisecond)
	log.Info("compactionInspector start loop schedule", zap.Duration("schedule interval", interval))
	defer c.stopWg.Done()

	scheduleTicker := time.NewTicker(interval)
	defer scheduleTicker.Stop()
	for {
		select {
		case <-c.stopCh:
			log.Info("compactionInspector quit loop schedule")
			return

		case <-scheduleTicker.C:
			c.checkSchedule()
		}
	}
}

func (c *compactionInspector) loopClean() {
	interval := Params.DataCoordCfg.CompactionGCIntervalInSeconds.GetAsDuration(time.Second)
	log.Info("compactionInspector start clean check loop", zap.Any("gc interval", interval))
	defer c.stopWg.Done()
	cleanTicker := time.NewTicker(interval)
	defer cleanTicker.Stop()
	for {
		select {
		case <-c.stopCh:
			log.Info("Compaction inspector quit loopClean")
			return
		case <-cleanTicker.C:
			c.Clean()
		}
	}
}

func (c *compactionInspector) Clean() {
	c.cleanCompactionTaskMeta()
	c.cleanPartitionStats()
}

func (c *compactionInspector) cleanCompactionTaskMeta() {
	// gc clustering compaction tasks
	triggers := c.meta.GetCompactionTasks(context.TODO())
	for _, tasks := range triggers {
		for _, task := range tasks {
			if task.State == datapb.CompactionTaskState_cleaned {
				duration := time.Since(time.Unix(task.StartTime, 0)).Seconds()
				if duration > float64(Params.DataCoordCfg.CompactionDropToleranceInSeconds.GetAsDuration(time.Second).Seconds()) {
					// try best to delete meta
					err := c.meta.DropCompactionTask(context.TODO(), task)
					log.Ctx(context.TODO()).Debug("drop compaction task meta", zap.Int64("planID", task.PlanID))
					if err != nil {
						log.Ctx(context.TODO()).Warn("fail to drop task", zap.Int64("planID", task.PlanID), zap.Error(err))
					}
				}
			}
		}
	}
}

func (c *compactionInspector) cleanPartitionStats() error {
	log := log.Ctx(context.TODO())
	log.Debug("start gc partitionStats meta and files")
	// gc partition stats
	channelPartitionStatsInfos := make(map[string][]*datapb.PartitionStatsInfo)
	unusedPartStats := make([]*datapb.PartitionStatsInfo, 0)
	if c.meta.GetPartitionStatsMeta() == nil {
		return nil
	}
	infos := c.meta.GetPartitionStatsMeta().ListAllPartitionStatsInfos()
	for _, info := range infos {
		collInfo := c.meta.(*meta).GetCollection(info.GetCollectionID())
		if collInfo == nil {
			unusedPartStats = append(unusedPartStats, info)
			continue
		}
		channel := fmt.Sprintf("%d/%d/%s", info.CollectionID, info.PartitionID, info.VChannel)
		if _, ok := channelPartitionStatsInfos[channel]; !ok {
			channelPartitionStatsInfos[channel] = make([]*datapb.PartitionStatsInfo, 0)
		}
		channelPartitionStatsInfos[channel] = append(channelPartitionStatsInfos[channel], info)
	}
	log.Debug("channels with PartitionStats meta", zap.Int("len", len(channelPartitionStatsInfos)))

	for _, info := range unusedPartStats {
		log.Debug("collection has been dropped, remove partition stats",
			zap.Int64("collID", info.GetCollectionID()))
		if err := c.meta.CleanPartitionStatsInfo(context.TODO(), info); err != nil {
			log.Warn("gcPartitionStatsInfo fail", zap.Error(err))
			return err
		}
	}

	for channel, infos := range channelPartitionStatsInfos {
		sort.Slice(infos, func(i, j int) bool {
			return infos[i].Version > infos[j].Version
		})
		log.Debug("PartitionStats in channel", zap.String("channel", channel), zap.Int("len", len(infos)))
		if len(infos) > 2 {
			for i := 2; i < len(infos); i++ {
				info := infos[i]
				if err := c.meta.CleanPartitionStatsInfo(context.TODO(), info); err != nil {
					log.Warn("gcPartitionStatsInfo fail", zap.Error(err))
					return err
				}
			}
		}
	}
	return nil
}

func (c *compactionInspector) stop() {
	c.stopOnce.Do(func() {
		close(c.stopCh)
	})
	c.stopWg.Wait()
}

func (c *compactionInspector) removeTasksByChannel(channel string) {
	log := log.Ctx(context.TODO())
	log.Info("removing tasks by channel", zap.String("channel", channel))
	c.queueTasks.RemoveAll(func(task CompactionTask) bool {
		if task.GetTaskProto().GetChannel() == channel {
			log.Info("Compaction inspector removing tasks by channel",
				zap.String("channel", channel),
				zap.Int64("planID", task.GetTaskProto().GetPlanID()),
				zap.Int64("node", task.GetTaskProto().GetNodeID()),
			)
			metrics.DataCoordCompactionTaskNum.WithLabelValues(fmt.Sprintf("%d", task.GetTaskProto().GetNodeID()), task.GetTaskProto().GetType().String(), metrics.Pending).Dec()
			return true
		}
		return false
	})

	c.executingGuard.Lock()
	for id, task := range c.executingTasks {
		log.Info("Compaction inspector removing tasks by channel",
			zap.String("channel", channel), zap.Int64("planID", id), zap.Any("task_channel", task.GetTaskProto().GetChannel()))
		if task.GetTaskProto().GetChannel() == channel {
			log.Info("Compaction inspector removing tasks by channel",
				zap.String("channel", channel),
				zap.Int64("planID", task.GetTaskProto().GetPlanID()),
				zap.Int64("node", task.GetTaskProto().GetNodeID()),
			)
			delete(c.executingTasks, id)
			metrics.DataCoordCompactionTaskNum.WithLabelValues(fmt.Sprintf("%d", task.GetTaskProto().GetNodeID()), task.GetTaskProto().GetType().String(), metrics.Executing).Dec()
		}
	}
	c.executingGuard.Unlock()
}

func (c *compactionInspector) submitTask(t CompactionTask) error {
	if err := c.queueTasks.Enqueue(t); err != nil {
		return err
	}
	metrics.DataCoordCompactionTaskNum.WithLabelValues(fmt.Sprintf("%d", NullNodeID), t.GetTaskProto().GetType().String(), metrics.Pending).Inc()
	return nil
}

// restoreTask used to restore Task from etcd
func (c *compactionInspector) restoreTask(t CompactionTask) {
	c.executingGuard.Lock()
	c.executingTasks[t.GetTaskProto().GetPlanID()] = t
	c.scheduler.Enqueue(t)
	c.executingGuard.Unlock()
	metrics.DataCoordCompactionTaskNum.WithLabelValues(fmt.Sprintf("%d", t.GetTaskProto().GetNodeID()), t.GetTaskProto().GetType().String(), metrics.Executing).Inc()
}

// getCompactionTask return compaction
func (c *compactionInspector) getCompactionTask(planID int64) CompactionTask {
	var t CompactionTask = nil
	c.queueTasks.ForEach(func(task CompactionTask) {
		if task.GetTaskProto().GetPlanID() == planID {
			t = task
		}
	})
	if t != nil {
		return t
	}

	c.executingGuard.RLock()
	defer c.executingGuard.RUnlock()
	t = c.executingTasks[planID]
	return t
}

func (c *compactionInspector) enqueueCompaction(task *datapb.CompactionTask) error {
	log := log.Ctx(context.TODO()).With(zap.Int64("planID", task.GetPlanID()), zap.Int64("triggerID", task.GetTriggerID()), zap.Int64("collectionID", task.GetCollectionID()), zap.String("type", task.GetType().String()))
	t, err := c.createCompactTask(task)
	if err != nil {
		// Conflict is normal
		if errors.Is(err, merr.ErrCompactionPlanConflict) {
			log.RatedInfo(60, "Failed to create compaction task, compaction plan conflict", zap.Error(err))
		} else {
			log.Warn("Failed to create compaction task, unable to create compaction task", zap.Error(err))
		}
		return err
	}

	t.SetTask(t.ShadowClone(setStartTime(time.Now().Unix())))
	err = t.SaveTaskMeta()
	if err != nil {
		c.meta.SetSegmentsCompacting(context.TODO(), t.GetTaskProto().GetInputSegments(), false)
		log.Warn("Failed to enqueue compaction task, unable to save task meta", zap.Error(err))
		return err
	}
	if err = c.submitTask(t); err != nil {
		log.Warn("submit compaction task failed", zap.Error(err))
		c.meta.SetSegmentsCompacting(context.Background(), t.GetTaskProto().GetInputSegments(), false)
		return err
	}
	log.Info("Compaction plan submitted")
	return nil
}

// set segments compacting, one segment can only participate one compactionTask
func (c *compactionInspector) createCompactTask(t *datapb.CompactionTask) (CompactionTask, error) {
	var task CompactionTask
	switch t.GetType() {
	case datapb.CompactionType_MixCompaction, datapb.CompactionType_SortCompaction:
		task = newMixCompactionTask(t, c.allocator, c.meta, c.ievm)
	case datapb.CompactionType_Level0DeleteCompaction:
		task = newL0CompactionTask(t, c.allocator, c.meta)
	case datapb.CompactionType_ClusteringCompaction:
		task = newClusteringCompactionTask(t, c.allocator, c.meta, c.handler, c.analyzeScheduler)
	default:
		return nil, merr.WrapErrIllegalCompactionPlan("illegal compaction type")
	}
	exist, succeed := c.meta.CheckAndSetSegmentsCompacting(context.TODO(), t.GetInputSegments())
	if !exist {
		return nil, merr.WrapErrIllegalCompactionPlan("segment not exist")
	}
	if !succeed {
		return nil, merr.WrapErrCompactionPlanConflict("segment is compacting")
	}
	return task, nil
}

// checkCompaction retrieves executing tasks and calls each task's Process() method
// to evaluate its state and progress through the state machine.
// Completed tasks are removed from executingTasks.
// Tasks that fail or timeout are moved from executingTasks to cleaningTasks,
// where task-specific clean logic is performed asynchronously.
func (c *compactionInspector) checkCompaction() error {
	// Get executing executingTasks before GetCompactionState from DataNode to prevent false failure,
	//  for DC might add new task while GetCompactionState.

	var finishedTasks []CompactionTask
	c.executingGuard.RLock()
	for _, t := range c.executingTasks {
		c.checkDelay(t)
		finished := t.Process()
		if finished {
			finishedTasks = append(finishedTasks, t)
		}
	}
	c.executingGuard.RUnlock()

	// delete all finished
	c.executingGuard.Lock()
	for _, t := range finishedTasks {
		delete(c.executingTasks, t.GetTaskProto().GetPlanID())
		metrics.DataCoordCompactionTaskNum.WithLabelValues(fmt.Sprintf("%d", t.GetTaskProto().GetNodeID()), t.GetTaskProto().GetType().String(), metrics.Executing).Dec()
		metrics.DataCoordCompactionTaskNum.WithLabelValues(fmt.Sprintf("%d", t.GetTaskProto().GetNodeID()), t.GetTaskProto().GetType().String(), metrics.Done).Inc()
	}
	c.executingGuard.Unlock()

	// insert task need to clean
	c.cleaningGuard.Lock()
	for _, t := range finishedTasks {
		if t.GetTaskProto().GetState() == datapb.CompactionTaskState_failed ||
			t.GetTaskProto().GetState() == datapb.CompactionTaskState_timeout ||
			t.GetTaskProto().GetState() == datapb.CompactionTaskState_completed {
			log.Ctx(context.TODO()).Info("task need to clean",
				zap.Int64("collectionID", t.GetTaskProto().GetCollectionID()),
				zap.Int64("planID", t.GetTaskProto().GetPlanID()),
				zap.String("state", t.GetTaskProto().GetState().String()))
			c.cleaningTasks[t.GetTaskProto().GetPlanID()] = t
		}
	}
	c.cleaningGuard.Unlock()

	return nil
}

// cleanFailedTasks performs task define Clean logic
// while compactionInspector.Clean is to do garbage collection for cleaned tasks
func (c *compactionInspector) cleanFailedTasks() {
	c.cleaningGuard.RLock()
	cleanedTasks := make([]CompactionTask, 0)
	for _, t := range c.cleaningTasks {
		clean := t.Clean()
		if clean {
			cleanedTasks = append(cleanedTasks, t)
		}
	}
	c.cleaningGuard.RUnlock()
	c.cleaningGuard.Lock()
	for _, t := range cleanedTasks {
		delete(c.cleaningTasks, t.GetTaskProto().GetPlanID())
	}
	c.cleaningGuard.Unlock()
}

// isFull return true if the task pool is full
func (c *compactionInspector) isFull() bool {
	return c.queueTasks.Len() >= c.queueTasks.capacity
}

func (c *compactionInspector) checkDelay(t CompactionTask) {
	log := log.Ctx(context.TODO()).WithRateGroup("compactionInspector.checkDelay", 1.0, 60.0)
	maxExecDuration := maxCompactionTaskExecutionDuration[t.GetTaskProto().GetType()]
	startTime := time.Unix(t.GetTaskProto().GetStartTime(), 0)
	execDuration := time.Since(startTime)
	if execDuration >= maxExecDuration {
		log.RatedWarn(60, "compaction task is delay",
			zap.Int64("planID", t.GetTaskProto().GetPlanID()),
			zap.String("type", t.GetTaskProto().GetType().String()),
			zap.String("state", t.GetTaskProto().GetState().String()),
			zap.String("vchannel", t.GetTaskProto().GetChannel()),
			zap.Int64("nodeID", t.GetTaskProto().GetNodeID()),
			zap.Time("startTime", startTime),
			zap.Duration("execDuration", execDuration))
	}
}

func (c *compactionInspector) getCompactionTasksNum(filters ...compactionTaskFilter) int {
	cnt := 0
	isMatch := func(task CompactionTask) bool {
		for _, f := range filters {
			if !f(task) {
				return false
			}
		}
		return true
	}
	c.queueTasks.ForEach(func(task CompactionTask) {
		if isMatch(task) {
			cnt += 1
		}
	})
	c.executingGuard.RLock()
	for _, t := range c.executingTasks {
		if isMatch(t) {
			cnt += 1
		}
	}
	c.executingGuard.RUnlock()
	return cnt
}

type compactionTaskFilter func(task CompactionTask) bool

func CollectionIDCompactionTaskFilter(collectionID int64) compactionTaskFilter {
	return func(task CompactionTask) bool {
		return task.GetTaskProto().GetCollectionID() == collectionID
	}
}

func L0CompactionCompactionTaskFilter() compactionTaskFilter {
	return func(task CompactionTask) bool {
		return task.GetTaskProto().GetType() == datapb.CompactionType_Level0DeleteCompaction
	}
}

var (
	ioPool         *conc.Pool[any]
	ioPoolInitOnce sync.Once
)

func initIOPool() {
	capacity := Params.DataNodeCfg.IOConcurrency.GetAsInt()
	if capacity > 32 {
		capacity = 32
	}
	// error only happens with negative expiry duration or with negative pre-alloc size.
	ioPool = conc.NewPool[any](capacity)
}

func getOrCreateIOPool() *conc.Pool[any] {
	ioPoolInitOnce.Do(initIOPool)
	return ioPool
}
