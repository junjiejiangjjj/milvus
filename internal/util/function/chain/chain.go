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

// =============================================================================
// Interfaces
// =============================================================================

// FunctionExpr is a function expression interface that processes columns.
// Functions are stateless and only focus on computation logic.
// Column mapping is handled by the Operator layer (MapOp).
type FunctionExpr interface {
	// Name returns the function name (e.g., "decay", "score_combine").
	Name() string

	// OutputDataTypes returns the data types of output columns.
	// The length determines how many output columns the function produces.
	OutputDataTypes() []arrow.DataType

	// Execute executes the function on input columns and returns output columns.
	// inputs: ChunkedArrays passed from the Operator
	// returns: ChunkedArrays, one for each OutputDataTypes()
	Execute(ctx *FuncContext, inputs []*arrow.Chunked) ([]*arrow.Chunked, error)

	// IsRunnable checks if the function can run in the given stage.
	IsRunnable(stage string) bool
}

// Operator is the operator interface.
type Operator interface {
	// Name returns the operator name.
	Name() string

	// Execute executes the operator.
	Execute(ctx *FuncContext, input *DataFrame) (*DataFrame, error)

	// Inputs returns the input column names.
	Inputs() []string

	// Outputs returns the output column names.
	Outputs() []string

	// String returns a string representation of the operator.
	String() string
}

// =============================================================================
// Context
// =============================================================================

// FuncContext is the execution context.
// The pool is immutable after creation.
type FuncContext struct {
	pool  memory.Allocator
	stage string // execution stage for filtering operators
}

// NewFuncContext creates a new FuncContext with the given allocator.
func NewFuncContext(pool memory.Allocator) *FuncContext {
	if pool == nil {
		pool = memory.DefaultAllocator
	}
	return &FuncContext{pool: pool}
}

// NewFuncContextWithStage creates a new FuncContext with allocator and stage.
func NewFuncContextWithStage(pool memory.Allocator, stage string) *FuncContext {
	ctx := NewFuncContext(pool)
	ctx.stage = stage
	return ctx
}

// Pool returns the memory allocator.
func (ctx *FuncContext) Pool() memory.Allocator {
	return ctx.pool
}

// Stage returns the execution stage.
func (ctx *FuncContext) Stage() string {
	return ctx.stage
}

// =============================================================================
// FuncChain
// =============================================================================

// FuncChain is a function chain that contains a list of operators.
type FuncChain struct {
	name       string
	operators  []Operator
	alloc      memory.Allocator
	buildError error // stores error from fluent API calls
}

// NewFuncChain creates a new FuncChain.
func NewFuncChain() *FuncChain {
	return &FuncChain{
		operators: make([]Operator, 0),
		alloc:     memory.DefaultAllocator,
	}
}

// NewFuncChainWithAllocator creates a new FuncChain with the given allocator.
func NewFuncChainWithAllocator(alloc memory.Allocator) *FuncChain {
	if alloc == nil {
		alloc = memory.DefaultAllocator
	}
	return &FuncChain{
		operators: make([]Operator, 0),
		alloc:     alloc,
	}
}

// SetName sets the name of the FuncChain.
func (fc *FuncChain) SetName(name string) *FuncChain {
	fc.name = name
	return fc
}

// Add adds an operator to the chain.
func (fc *FuncChain) Add(op Operator) *FuncChain {
	fc.operators = append(fc.operators, op)
	return fc
}

// AddWithError adds an operator to the chain, recording any error for later.
// This is used by fluent API methods to defer error handling to Execute/Validate.
func (fc *FuncChain) AddWithError(op Operator, err error) *FuncChain {
	if err != nil && fc.buildError == nil {
		fc.buildError = err
	}
	if op != nil {
		fc.operators = append(fc.operators, op)
	}
	return fc
}

// Validate validates the chain configuration before execution.
// It checks for build errors and validates each operator.
func (fc *FuncChain) Validate() error {
	// Check for errors accumulated during fluent API calls
	if fc.buildError != nil {
		return fmt.Errorf("chain build error: %w", fc.buildError)
	}

	// Validate each operator
	for i, op := range fc.operators {
		if op == nil {
			return fmt.Errorf("operator[%d] is nil", i)
		}
		// Check MapOp has valid function
		if mapOp, ok := op.(*MapOp); ok {
			if mapOp.function == nil {
				return fmt.Errorf("operator[%d] MapOp has nil function", i)
			}
		}
	}

	return nil
}

// Execute executes the chain.
func (fc *FuncChain) Execute(input *DataFrame) (*DataFrame, error) {
	return fc.ExecuteWithStage(input, "")
}

// ExecuteWithStage executes the chain with stage filtering.
// Operators with functions that don't support the given stage are skipped.
// If stage is empty, all operators are executed.
func (fc *FuncChain) ExecuteWithStage(input *DataFrame, stage string) (*DataFrame, error) {
	// Validate chain before execution
	if err := fc.Validate(); err != nil {
		return nil, err
	}

	ctx := NewFuncContext(fc.alloc)

	result := input
	for _, op := range fc.operators {
		// Check if operator should be skipped based on stage
		if stage != "" {
			if mapOp, ok := op.(*MapOp); ok {
				if mapOp.function != nil && !mapOp.function.IsRunnable(stage) {
					continue // Skip this operator
				}
			}
		}

		var err error
		newResult, err := op.Execute(ctx, result)
		if err != nil {
			return nil, fmt.Errorf("%s failed: %w", op.Name(), err)
		}
		// Release intermediate results (but not the original input)
		if result != input && result != newResult {
			result.Release()
		}
		result = newResult
	}
	return result, nil
}

// Map applies a function to the DataFrame with specified column mappings.
// inputCols: column names to read from DataFrame and pass to the function
// outputCols: column names to write the function output to
// Errors are deferred until Execute() or Validate() is called.
func (fc *FuncChain) Map(fn FunctionExpr, inputCols, outputCols []string) *FuncChain {
	op, err := NewMapOp(fn, inputCols, outputCols)
	return fc.AddWithError(op, err)
}

// MapWithError is like Map but returns an error immediately instead of deferring it.
// Use this when you want immediate error feedback rather than fluent chaining.
func (fc *FuncChain) MapWithError(fn FunctionExpr, inputCols, outputCols []string) (*FuncChain, error) {
	op, err := NewMapOp(fn, inputCols, outputCols)
	if err != nil {
		return fc, err
	}
	return fc.Add(op), nil
}

// Filter filters the DataFrame based on a boolean column.
func (fc *FuncChain) Filter(column string) *FuncChain {
	return fc.Add(&FilterOp{column: column})
}

// Select selects specific columns from the DataFrame.
func (fc *FuncChain) Select(columns ...string) *FuncChain {
	return fc.Add(&SelectOp{columns: columns})
}

// Sort sorts the DataFrame by a column.
func (fc *FuncChain) Sort(column string, desc bool) *FuncChain {
	return fc.Add(&SortOp{column: column, desc: desc})
}

// Limit limits the number of rows in the DataFrame.
func (fc *FuncChain) Limit(limit int64) *FuncChain {
	return fc.Add(&LimitOp{limit: limit, offset: 0})
}

// LimitWithOffset limits the number of rows with an offset.
func (fc *FuncChain) LimitWithOffset(limit, offset int64) *FuncChain {
	return fc.Add(&LimitOp{limit: limit, offset: offset})
}

// String returns a string representation of the FuncChain.
func (fc *FuncChain) String() string {
	buf := bytes.NewBufferString(fmt.Sprintf("FuncChain: %s\n", fc.name))
	for i, op := range fc.operators {
		fmt.Fprintf(buf, "  [%d] %s: %v -> %v\n", i, op.Name(), op.Inputs(), op.Outputs())
	}
	return buf.String()
}
