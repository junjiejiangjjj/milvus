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
	"fmt"
	"sync"
)

// =============================================================================
// Function Registry
// =============================================================================

// FunctionFactory is a factory function that creates a FunctionExpr from parameters.
type FunctionFactory func(params map[string]interface{}) (FunctionExpr, error)

// FunctionRegistry is a registry for function factories.
type FunctionRegistry struct {
	mu        sync.RWMutex
	factories map[string]FunctionFactory
}

// NewFunctionRegistry creates a new FunctionRegistry.
func NewFunctionRegistry() *FunctionRegistry {
	return &FunctionRegistry{
		factories: make(map[string]FunctionFactory),
	}
}

// Register registers a function factory with the given name.
func (r *FunctionRegistry) Register(name string, factory FunctionFactory) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.factories[name] = factory
}

// Get returns the function factory for the given name.
func (r *FunctionRegistry) Get(name string) (FunctionFactory, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	factory, ok := r.factories[name]
	return factory, ok
}

// Create creates a FunctionExpr using the factory registered with the given name.
func (r *FunctionRegistry) Create(name string, params map[string]interface{}) (FunctionExpr, error) {
	factory, ok := r.Get(name)
	if !ok {
		return nil, fmt.Errorf("unknown function: %s", name)
	}
	return factory(params)
}

// Has returns true if a function with the given name is registered.
func (r *FunctionRegistry) Has(name string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	_, ok := r.factories[name]
	return ok
}

// Names returns all registered function names.
func (r *FunctionRegistry) Names() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	names := make([]string, 0, len(r.factories))
	for name := range r.factories {
		names = append(names, name)
	}
	return names
}

// =============================================================================
// Global Registry
// =============================================================================

// globalRegistry is the global function registry.
var globalRegistry = NewFunctionRegistry()

// RegisterFunction registers a function factory in the global registry.
func RegisterFunction(name string, factory FunctionFactory) {
	globalRegistry.Register(name, factory)
}

// GetFunctionFactory returns the function factory from the global registry.
func GetFunctionFactory(name string) (FunctionFactory, bool) {
	return globalRegistry.Get(name)
}

// CreateFunction creates a FunctionExpr using the global registry.
func CreateFunction(name string, params map[string]interface{}) (FunctionExpr, error) {
	return globalRegistry.Create(name, params)
}

// HasFunction returns true if a function with the given name is registered in the global registry.
func HasFunction(name string) bool {
	return globalRegistry.Has(name)
}

// FunctionNames returns all registered function names from the global registry.
func FunctionNames() []string {
	return globalRegistry.Names()
}
