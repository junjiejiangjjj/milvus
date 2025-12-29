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

import (
	"github.com/apache/arrow/go/v17/arrow"
	"github.com/apache/arrow/go/v17/arrow/array"
	"github.com/apache/arrow/go/v17/arrow/memory"
)

func buildInt64Array(pool memory.Allocator, data []int64) arrow.Array {
	builder := array.NewInt64Builder(pool)
	defer builder.Release()
	builder.AppendValues(data, nil)
	return builder.NewArray()
}

func buildInt64Arrays(pool memory.Allocator, data []int64, topks []int64) []arrow.Array {
	int64Arrays := make([]arrow.Array, 0, len(topks))
	start := int64(0)
	for _, topk := range topks {
		int64Array := buildInt64Array(pool, data[start:start+topk])
		int64Arrays = append(int64Arrays, int64Array)
		start += topk
	}
	return int64Arrays
}

func buildStringArray(pool memory.Allocator, data []string) arrow.Array {
	builder := array.NewStringBuilder(pool)
	defer builder.Release()
	builder.AppendValues(data, nil)
	return builder.NewArray()
}

func buildStringArrays(pool memory.Allocator, data []string, topks []int64) []arrow.Array {
	stringArrays := make([]arrow.Array, 0, len(topks))
	for _, topk := range topks {
		stringArray := buildStringArray(pool, data[:topk])
		stringArrays = append(stringArrays, stringArray)
	}
	return stringArrays
}

func buildBoolArray(pool memory.Allocator, data []bool) arrow.Array {
	builder := array.NewBooleanBuilder(pool)
	defer builder.Release()
	builder.AppendValues(data, nil)
	return builder.NewArray()
}

func buildBoolArrays(pool memory.Allocator, data []bool, topks []int64) []arrow.Array {
	boolArrays := make([]arrow.Array, 0, len(topks))
	for _, topk := range topks {
		boolArray := buildBoolArray(pool, data[:topk])
		boolArrays = append(boolArrays, boolArray)
	}
	return boolArrays
}

func buildFloat32Array(pool memory.Allocator, data []float32) arrow.Array {
	builder := array.NewFloat32Builder(pool)
	defer builder.Release()
	builder.AppendValues(data, nil)
	return builder.NewArray()
}

func buildFloat32Arrays(pool memory.Allocator, data []float32, topks []int64) []arrow.Array {
	float32Arrays := make([]arrow.Array, 0, len(topks))
	for _, topk := range topks {
		float32Array := buildFloat32Array(pool, data[:topk])
		float32Arrays = append(float32Arrays, float32Array)
	}
	return float32Arrays
}

func buildFloat64Array(pool memory.Allocator, data []float64) arrow.Array {
	builder := array.NewFloat64Builder(pool)
	defer builder.Release()
	builder.AppendValues(data, nil)
	return builder.NewArray()
}

func buildFloat64Arrays(pool memory.Allocator, data []float64, topks []int64) []arrow.Array {
	float64Arrays := make([]arrow.Array, 0, len(topks))
	for _, topk := range topks {
		float64Array := buildFloat64Array(pool, data[:topk])
		float64Arrays = append(float64Arrays, float64Array)
	}
	return float64Arrays
}
