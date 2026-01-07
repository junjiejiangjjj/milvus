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
	"bytes"
	"fmt"

	"github.com/apache/arrow/go/v17/arrow"
	"github.com/apache/arrow/go/v17/arrow/memory"
)

type Chain struct {
	Alloc     memory.Allocator
	Operators []Operator
	Name      string
}

func (c *Chain) String() string {
	buf := bytes.NewBufferString(fmt.Sprintf("FunctionChain: %s\n", c.Name))
	for _, op := range c.Operators {
		buf.WriteString(fmt.Sprintf("  %s -> %s", op.Inputs(), op.Outputs()))
	}
	return buf.String()
}

func (c *Chain) Run(ctx *ChainContext, input *DataFrame) (*DataFrame, error) {
	return nil, nil
}

type Expr interface {
	Eval(ctx *ChainContext, alloc memory.Allocator, inputs []arrow.Array) ([]arrow.Array, error)
}

type Operator interface {
	Run(ctx *ChainContext, input *DataFrame) (*DataFrame, error)
	Inputs() []string
	Outputs() []string
	ToString() string
}

type BaseOperator struct {
	inputs     []string
	outputs    []string
	returnType []arrow.DataType
	alloc      memory.Allocator
	impl       Expr
}

type MapOperator struct {
	BaseOperator
}

func (o *MapOperator) Run(ctx *ChainContext, input *DataFrame) (*DataFrame, error) {
	for chunkIndex := 0; chunkIndex < input.GetChunkCount(); chunkIndex++ {
		columns, err := input.GetColumns(o.inputs, chunkIndex)
		if err != nil {
			return nil, err
		}
		_, err = o.impl.Eval(ctx, o.alloc, columns)
		if err != nil {
			return nil, err
		}
	}
	return nil, nil
}

type FilterOperator struct {
	BaseOperator
}

func (o *FilterOperator) Run(ctx *ChainContext, input *DataFrame) (*DataFrame, error) {
	return nil, nil
}
