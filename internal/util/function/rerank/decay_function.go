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

package rerank

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/pkg/v2/util/typeutil"
)

const (
	originKey   string = "origin"
	scaleKey    string = "scale"
	offsetKey   string = "offset"
	decayKey    string = "decay"
	functionKey string = "function"
)

const (
	gaussFunction string = "gauss"
	linerFunction string = "liner"
	expFunction   string = "exp"
)

type DecayFunction[T int64 | string, R int32 | int64 | float32 | float64] struct {
	RerankBase

	functionName string
	origin       float64
	scale        float64
	offset       float64
	decay        float64
	reScorer     decayReScorer
}

func newDecayFunction(collSchema *schemapb.CollectionSchema, funcSchema *schemapb.FunctionSchema, pkField schemapb.DataType) (Reranker, error) {
	base, err := newRerankBase(collSchema, funcSchema, decayFunctionName, false, pkField)
	if err != nil {
		return nil, err
	}

	var inputType schemapb.DataType
	for _, field := range collSchema.Fields {
		if field.Name == base.GetInputFieldNames()[0] {
			inputType = field.DataType
		}
	}

	if pkField == schemapb.DataType_Int64 {
		switch inputType {
		case schemapb.DataType_Int8, schemapb.DataType_Int16, schemapb.DataType_Int32:
			return newFunction[int64, int32](base, funcSchema)
		case schemapb.DataType_Int64:
			return &DecayFunction[int64, int64]{RerankBase: *base, offset: 0, decay: 0.5}, nil
		case schemapb.DataType_Float:
			return &DecayFunction[int64, float32]{RerankBase: *base, offset: 0, decay: 0.5}, nil
		case schemapb.DataType_Double:
			return &DecayFunction[int64, float64]{RerankBase: *base, offset: 0, decay: 0.5}, nil
		default:
			return nil, fmt.Errorf("Decay rerank: unsupported input field type:%s, only support numberic field", inputType.String())
		}
	} else {
		switch inputType {
		case schemapb.DataType_Int8, schemapb.DataType_Int16, schemapb.DataType_Int32:
			return &DecayFunction[string, int32]{RerankBase: *base, offset: 0, decay: 0.5}, nil
		case schemapb.DataType_Int64:
			return &DecayFunction[string, int64]{RerankBase: *base, offset: 0, decay: 0.5}, nil
		case schemapb.DataType_Float:
			return &DecayFunction[string, float32]{RerankBase: *base, offset: 0, decay: 0.5}, nil
		case schemapb.DataType_Double:
			return &DecayFunction[string, float64]{RerankBase: *base, offset: 0, decay: 0.5}, nil
		default:
			return nil, fmt.Errorf("Decay rerank: unsupported input field type:%s, only support numberic field", inputType.String())
		}
	}

}

func newFunction[T int64 | string, R int32 | int64 | float32 | float64](base *RerankBase, funcSchema *schemapb.FunctionSchema) (Reranker, error) {
	var err error
	decayFunc := &DecayFunction[T, R]{RerankBase: *base, offset: 0, decay: 0.5}
	orginInit := false
	scaleInit := false
	for _, param := range funcSchema.Params {
		switch strings.ToLower(param.Key) {
		case functionKey:
			decayFunc.functionName = param.Value
		case originKey:
			if decayFunc.origin, err = strconv.ParseFloat(param.Value, 64); err != nil {
				return nil, fmt.Errorf("Param origin:%s is not a number", param.Value)
			}
			orginInit = true
		case scaleKey:
			if decayFunc.scale, err = strconv.ParseFloat(param.Value, 64); err != nil {
				return nil, fmt.Errorf("Param scale:%s is not a number", param.Value)
			}
			if decayFunc.scale <= 0 {
				return nil, fmt.Errorf("Scale must > 0, but got %s", param.Value)
			}
			scaleInit = true
		case offsetKey:
			if decayFunc.offset, err = strconv.ParseFloat(param.Value, 64); err != nil {
				return nil, fmt.Errorf("Param offset:%s is not a number", param.Value)
			}
		case decayKey:
			if decayFunc.decay, err = strconv.ParseFloat(param.Value, 64); err != nil {
				return nil, fmt.Errorf("Param decay:%s is not a number", param.Value)
			}
		default:
		}
	}

	if !orginInit {
		return nil, fmt.Errorf("Decay function lost param: origin")
	}

	if !scaleInit {
		return nil, fmt.Errorf("Decay function lost param: scale")
	}
	switch decayFunctionName {
	case gaussFunction:
		decayFunc.reScorer = gaussianDecay
	case expFunction:
		decayFunc.reScorer = expDecay
	case linerFunction:
		decayFunc.reScorer = linearDecay
	default:
		return nil, fmt.Errorf("Invaild decay function: %s, only support [%s,%s,%s]", decayFunctionName, gaussFunction, linerFunction, expFunction)
	}

	if err := decayFunc.check(); err != nil {
		return nil, err
	}
	return nil, nil
}

func (decay *DecayFunction[T, R]) check() error {
	if len(decay.funcSchema.InputFieldNames) != 1 {
		return fmt.Errorf("Decay function only supoorts single input, but gets [%s] input", decay.funcSchema.InputFieldNames)
	}

	name := decay.funcSchema.InputFieldNames[0]
	var inputField *schemapb.FieldSchema
	for _, field := range decay.coll.Fields {
		if name == field.Name {
			inputField = field
		}
	}

	if inputField == nil {
		return fmt.Errorf("Decay function can not find field:[%s] in schema", name)
	}

	// decay function only supports numeric input
	if inputField.DataType != schemapb.DataType_Int8 &&
		inputField.DataType != schemapb.DataType_Int16 &&
		inputField.DataType != schemapb.DataType_Int32 &&
		inputField.DataType != schemapb.DataType_Int64 &&
		inputField.DataType != schemapb.DataType_Float &&
		inputField.DataType != schemapb.DataType_Double {
		return fmt.Errorf("Decay function supports numeric input, but the input field %s is %s", name, inputField.DataType.String())
	}
	return nil
}

func (decay *DecayFunction[T, R]) reScore(ctx context.Context, multipSearchResultData []*schemapb.SearchResultData) (*idSocres[T], error) {
	newScores := idSocres[T]{}
	for _, data := range multipSearchResultData {
		var inputField *schemapb.FieldData
		for _, field := range data.FieldsData {
			if field.FieldName == decay.GetInputFieldNames()[0] {
				inputField = field
			}
		}
		if inputField == nil {
			return nil, fmt.Errorf("Rerank decay function can not find input field, name: %s", decay.GetInputFieldNames()[0])
		}
		var inputValues numberField[R]
		if tmp, err := getNumberic(inputField); err != nil {
			return nil, err
		} else {
			inputValues = tmp.(numberField[R])
		}

		ids := newMilvusIDs(data.Ids, decay.pkType).(milvusIDs[T])
		for idx, id := range ids.data {
			if !newScores.exist(id) {
				if v, err := inputValues.GetFloat64(idx); err != nil {
					return nil, err
				} else {
					newScores.set(id, float32(decay.reScorer(decay.origin, decay.scale, decay.decay, decay.offset, v)))
				}

			}
		}

	}
	return &newScores, nil
}

func (decay *DecayFunction[T, R]) orgnizeNqScores(searchParams *SearchParams, multipSearchResultData []*schemapb.SearchResultData, idScoreData *idSocres[T]) []map[T]float32 {
	nqScores := make([]map[T]float32, searchParams.nq)
	for i := int64(0); i < searchParams.nq; i++ {
		nqScores[i] = make(map[T]float32)
	}

	for _, data := range multipSearchResultData {
		start := int64(0)
		for nqIdx := int64(0); nqIdx < searchParams.nq; nqIdx++ {
			realTopk := data.Topks[nqIdx]
			for j := start; j < start+realTopk; j++ {
				id := typeutil.GetPK(data.GetIds(), j).(T)
				if _, exists := nqScores[nqIdx][id]; !exists {
					nqScores[nqIdx][id] = idScoreData.get(id)
				}
			}
			start += realTopk
		}
	}
	return nqScores
}

func (decay *DecayFunction[T, R]) Process(ctx context.Context, searchParams *SearchParams, multipSearchResultData []*schemapb.SearchResultData) (*schemapb.SearchResultData, error) {

	idScoreData, err := decay.reScore(ctx, multipSearchResultData)
	if err != nil {
		return nil, err
	}

	nqScores := decay.orgnizeNqScores(searchParams, multipSearchResultData, idScoreData)

	ret := &schemapb.SearchResultData{
		NumQueries: searchParams.nq,
		TopK:       searchParams.limit,
		FieldsData: make([]*schemapb.FieldData, 0),
		Scores:     []float32{},
		Ids:        &schemapb.IDs{},
		Topks:      []int64{},
	}

	for i := int64(0); i < searchParams.nq; i++ {
		idScoreMap := nqScores[i]
		ids := make([]T, 0)
		for id := range idScoreMap {
			ids = append(ids, id)
		}

		big := func(i, j int) bool {
			if idScoreMap[ids[i]] == idScoreMap[ids[j]] {
				return ids[i] < ids[j]
			}
			return idScoreMap[ids[i]] > idScoreMap[ids[j]]
		}
		sort.Slice(ids, big)

		if int64(len(ids)) > searchParams.limit {
			ids = ids[:searchParams.limit]
		}

		// set real topk
		ret.Topks = append(ret.Topks, int64(len(ids)))
		// append id and score
		for index := 0; index < len(ids); index++ {
			typeutil.AppendPKs(ret.Ids, ids[index])
			score := idScoreMap[ids[index]]
			if searchParams.roundDecimal != -1 {
				multiplier := math.Pow(10.0, float64(searchParams.roundDecimal))
				score = float32(math.Floor(float64(score)*multiplier+0.5) / multiplier)
			}
			ret.Scores = append(ret.Scores, score)
		}
	}

	return ret, nil
}

type decayReScorer func(float64, float64, float64, float64, float64) float64

func gaussianDecay(origin, scale, decay, offset, distance float64) float64 {
	adjustedDist := math.Max(0, math.Abs(distance-origin)-offset)
	sigmaSquare := 0.5 * math.Pow(scale, 2.0) / math.Log(decay)
	exponent := math.Pow(adjustedDist, 2.0) / sigmaSquare
	return math.Exp(exponent)
}

func expDecay(origin, scale, decay, offset, distance float64) float64 {
	adjustedDist := math.Max(0, math.Abs(distance-origin)-offset)
	lambda := math.Log(decay) / scale
	return math.Exp(lambda * adjustedDist)
}

func linearDecay(origin, scale, decay, offset, distance float64) float64 {
	adjustedDist := math.Max(0, math.Abs(distance-origin)-offset)
	slope := (1 - decay) / scale
	return math.Max(decay, 1-slope*adjustedDist)
}
