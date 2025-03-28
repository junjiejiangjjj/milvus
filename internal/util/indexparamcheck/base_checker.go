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

package indexparamcheck

import (
	"github.com/cockroachdb/errors"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
)

type baseChecker struct{}

func (c baseChecker) CheckTrain(dataType schemapb.DataType, params map[string]string) error {
	return nil
}

// CheckValidDataType check whether the field data type is supported for the index type
func (c baseChecker) CheckValidDataType(indexType IndexType, field *schemapb.FieldSchema) error {
	return nil
}

func (c baseChecker) SetDefaultMetricTypeIfNotExist(dType schemapb.DataType, m map[string]string) {}

func (c baseChecker) StaticCheck(dataType schemapb.DataType, params map[string]string) error {
	return errors.New("unsupported index type")
}

func newBaseChecker() IndexChecker {
	return &baseChecker{}
}
