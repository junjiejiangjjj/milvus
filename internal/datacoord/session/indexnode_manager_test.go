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

package session

import (
	"context"
	"testing"

	"github.com/cockroachdb/errors"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"

	"github.com/milvus-io/milvus/internal/mocks"
	"github.com/milvus-io/milvus/internal/types"
	"github.com/milvus-io/milvus/pkg/v2/proto/workerpb"
	"github.com/milvus-io/milvus/pkg/v2/util/merr"
	"github.com/milvus-io/milvus/pkg/v2/util/paramtable"
	typeutil "github.com/milvus-io/milvus/pkg/v2/util/typeutil"
)

func TestIndexNodeManager_AddNode(t *testing.T) {
	paramtable.Init()
	nm := NewNodeManager(context.Background(), defaultIndexNodeCreatorFunc)

	t.Run("success", func(t *testing.T) {
		err := nm.AddNode(1, "indexnode-1")
		assert.NoError(t, err)
	})

	t.Run("fail", func(t *testing.T) {
		err := nm.AddNode(2, "")
		assert.Error(t, err)
	})
}

func TestIndexNodeManager_PickClient(t *testing.T) {
	paramtable.Init()
	getMockedGetJobStatsClient := func(resp *workerpb.GetJobStatsResponse, err error) types.DataNodeClient {
		ic := mocks.NewMockDataNodeClient(t)
		ic.EXPECT().GetJobStats(mock.Anything, mock.Anything, mock.Anything).Return(resp, err)
		return ic
	}

	err := errors.New("error")

	t.Run("multiple unavailable IndexNode", func(t *testing.T) {
		nm := &IndexNodeManager{
			ctx: context.TODO(),
			nodeClients: map[typeutil.UniqueID]types.DataNodeClient{
				1: getMockedGetJobStatsClient(&workerpb.GetJobStatsResponse{
					Status: merr.Status(err),
				}, err),
				2: getMockedGetJobStatsClient(&workerpb.GetJobStatsResponse{
					Status: merr.Status(err),
				}, err),
				3: getMockedGetJobStatsClient(&workerpb.GetJobStatsResponse{
					Status: merr.Status(err),
				}, err),
				4: getMockedGetJobStatsClient(&workerpb.GetJobStatsResponse{
					Status: merr.Status(err),
				}, err),
				5: getMockedGetJobStatsClient(&workerpb.GetJobStatsResponse{
					Status: merr.Status(err),
				}, nil),
				6: getMockedGetJobStatsClient(&workerpb.GetJobStatsResponse{
					Status: merr.Status(err),
				}, nil),
				7: getMockedGetJobStatsClient(&workerpb.GetJobStatsResponse{
					Status: merr.Status(err),
				}, nil),
				8: getMockedGetJobStatsClient(&workerpb.GetJobStatsResponse{
					AvailableSlots: 1,
					Status:         merr.Success(),
				}, nil),
				9: getMockedGetJobStatsClient(&workerpb.GetJobStatsResponse{
					AvailableSlots: 10,
					Status:         merr.Success(),
				}, nil),
			},
		}

		selectNodeID, client := nm.PickClient()
		assert.NotNil(t, client)
		assert.Contains(t, []typeutil.UniqueID{8, 9}, selectNodeID)
	})
}

func TestNodeManager_StoppingNode(t *testing.T) {
	paramtable.Init()
	nm := NewNodeManager(context.Background(), defaultIndexNodeCreatorFunc)
	err := nm.AddNode(1, "indexnode-1")
	assert.NoError(t, err)
	assert.Equal(t, 1, len(nm.GetAllClients()))

	nm.StoppingNode(1)
	assert.Equal(t, 0, len(nm.GetAllClients()))
	assert.Equal(t, 1, len(nm.stoppingNodes))

	nm.RemoveNode(1)
	assert.Equal(t, 0, len(nm.GetAllClients()))
	assert.Equal(t, 0, len(nm.stoppingNodes))
}
