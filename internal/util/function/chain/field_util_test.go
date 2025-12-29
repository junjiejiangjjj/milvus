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
	"testing"

	"github.com/apache/arrow/go/v17/arrow/array"
	"github.com/apache/arrow/go/v17/arrow/memory"
	"github.com/stretchr/testify/suite"
)

func TestFieldUtil(t *testing.T) {
	suite.Run(t, new(testFieldUtilSuite))
}

type testFieldUtilSuite struct {
	suite.Suite
	pool *memory.CheckedAllocator
}

func (s *testFieldUtilSuite) SetupTest() {
	s.pool = memory.NewCheckedAllocator(memory.DefaultAllocator)
}

func (s *testFieldUtilSuite) TearDownTest() {
	s.pool.AssertSize(s.T(), 0)
}

func (s *testFieldUtilSuite) TestBuildInt64Array() {

	data := []int64{1, 2, 3, 4, 5}

	arr := buildInt64Array(s.pool, data)
	defer arr.Release()

	s.Equal(5, arr.Len())

	s.Equal(int64(1), arr.(*array.Int64).Value(0))
	s.Equal(int64(5), arr.(*array.Int64).Value(4))
}

func (s *testFieldUtilSuite) TestBuildInt64Arrays() {
	data := []int64{1, 2, 3, 4, 5}
	topks := []int64{2, 3}

	arrays := buildInt64Arrays(s.pool, data, topks)
	defer func() {
		for _, arr := range arrays {
			arr.Release()
		}
	}()

	s.Equal(2, len(arrays))
	s.Equal(2, arrays[0].Len())
	s.Equal(3, arrays[1].Len())
	s.Equal(int64(1), arrays[0].(*array.Int64).Value(0))
	s.Equal(int64(2), arrays[0].(*array.Int64).Value(1))
	s.Equal(int64(3), arrays[1].(*array.Int64).Value(0))
}

func (s *testFieldUtilSuite) TestBuildInt64ArraysWithEmpty() {
	data := []int64{1, 2, 3, 4, 5}
	topks := []int64{2, 0, 3}

	arrays := buildInt64Arrays(s.pool, data, topks)
	defer func() {
		for _, arr := range arrays {
			arr.Release()
		}
	}()

	s.Equal(3, len(arrays))
	s.Equal(2, arrays[0].Len())
	s.Equal(0, arrays[1].Len())
	s.Equal(3, arrays[2].Len())
	s.Equal(int64(1), arrays[0].(*array.Int64).Value(0))
	s.Equal(int64(2), arrays[0].(*array.Int64).Value(1))
	s.Equal(int64(3), arrays[2].(*array.Int64).Value(0))
}

func (s *testFieldUtilSuite) TestBuildStringArray() {
	data := []string{"hello", "world", "test"}
	arr := buildStringArray(s.pool, data)
	defer arr.Release()

	s.Equal(3, arr.Len())
	s.Equal("hello", arr.(*array.String).Value(0))
	s.Equal("world", arr.(*array.String).Value(1))
	s.Equal("test", arr.(*array.String).Value(2))
}

func (s *testFieldUtilSuite) TestBuildStringArrays() {
	data := []string{"hello", "world", "test", "data"}
	topks := []int64{2, 3}

	arrays := buildStringArrays(s.pool, data, topks)
	defer func() {
		for _, arr := range arrays {
			arr.Release()
		}
	}()

	s.Equal(2, len(arrays))
	s.Equal(2, arrays[0].Len())
	s.Equal(3, arrays[1].Len())
	s.Equal("hello", arrays[0].(*array.String).Value(0))
	s.Equal("world", arrays[0].(*array.String).Value(1))
	s.Equal("test", arrays[1].(*array.String).Value(2))
}

func (s *testFieldUtilSuite) TestBuildBoolArray() {
	data := []bool{true, false, true}

	arr := buildBoolArray(s.pool, data)
	defer arr.Release()

	s.Equal(3, arr.Len())
	s.Equal(true, arr.(*array.Boolean).Value(0))
	s.Equal(false, arr.(*array.Boolean).Value(1))
	s.Equal(true, arr.(*array.Boolean).Value(2))
}

func (s *testFieldUtilSuite) TestBuildBoolArrays() {
	data := []bool{true, false, true, false}
	topks := []int64{2, 3}

	arrays := buildBoolArrays(s.pool, data, topks)
	defer func() {
		for _, arr := range arrays {
			arr.Release()
		}
	}()

	s.Equal(2, len(arrays))
	s.Equal(2, arrays[0].Len())
	s.Equal(3, arrays[1].Len())
	s.Equal(true, arrays[0].(*array.Boolean).Value(0))
	s.Equal(false, arrays[0].(*array.Boolean).Value(1))
	s.Equal(true, arrays[1].(*array.Boolean).Value(2))
}

func (s *testFieldUtilSuite) TestBuildFloat32Array() {
	data := []float32{1.1, 2.2, 3.3}

	arr := buildFloat32Array(s.pool, data)
	defer arr.Release()

	s.Equal(3, arr.Len())
	s.Equal(float32(1.1), arr.(*array.Float32).Value(0))
	s.Equal(float32(2.2), arr.(*array.Float32).Value(1))
	s.Equal(float32(3.3), arr.(*array.Float32).Value(2))
}

func (s *testFieldUtilSuite) TestBuildFloat32Arrays() {
	data := []float32{1.1, 2.2, 3.3, 4.4}
	topks := []int64{2, 3}

	arrays := buildFloat32Arrays(s.pool, data, topks)
	defer func() {
		for _, arr := range arrays {
			arr.Release()
		}
	}()

	s.Equal(2, len(arrays))
	s.Equal(2, arrays[0].Len())
	s.Equal(3, arrays[1].Len())
	s.Equal(float32(1.1), arrays[0].(*array.Float32).Value(0))
	s.Equal(float32(2.2), arrays[0].(*array.Float32).Value(1))
	s.Equal(float32(3.3), arrays[1].(*array.Float32).Value(2))
}

func (s *testFieldUtilSuite) TestBuildFloat64Array() {
	data := []float64{1.1, 2.2, 3.3}

	arr := buildFloat64Array(s.pool, data)
	defer arr.Release()

	s.Equal(3, arr.Len())
	s.Equal(float64(1.1), arr.(*array.Float64).Value(0))
	s.Equal(float64(2.2), arr.(*array.Float64).Value(1))
	s.Equal(float64(3.3), arr.(*array.Float64).Value(2))
}

func (s *testFieldUtilSuite) TestBuildFloat64Arrays() {
	data := []float64{1.1, 2.2, 3.3, 4.4}
	topks := []int64{2, 3}

	arrays := buildFloat64Arrays(s.pool, data, topks)
	defer func() {
		for _, arr := range arrays {
			arr.Release()
		}
	}()

	s.Equal(2, len(arrays))
	s.Equal(2, arrays[0].Len())
	s.Equal(3, arrays[1].Len())
	s.Equal(float64(1.1), arrays[0].(*array.Float64).Value(0))
	s.Equal(float64(2.2), arrays[0].(*array.Float64).Value(1))
	s.Equal(float64(3.3), arrays[1].(*array.Float64).Value(2))
}
