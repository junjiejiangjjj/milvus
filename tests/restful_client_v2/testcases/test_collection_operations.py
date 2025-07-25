import datetime
import logging
import time
from utils.util_log import test_log as logger
from utils.utils import gen_collection_name, gen_vector
import pytest
from api.milvus import CollectionClient
from base.testbase import TestBase
import threading
from utils.utils import get_data_by_payload
from pymilvus import (
    FieldSchema, CollectionSchema, DataType,
    Collection
)


@pytest.mark.L0
class TestCreateCollection(TestBase):

    @pytest.mark.parametrize("dim", [128])
    def test_create_collections_quick_setup(self, dim):
        """
        target: test create collection
        method: create a collection with a simple schema
        expected: create collection success
        """
        name = gen_collection_name()
        dim = 128
        client = self.collection_client
        payload = {
            "collectionName": name,
            "dimension": dim,
        }
        logging.info(f"create collection {name} with payload: {payload}")
        rsp = client.collection_create(payload)
        assert rsp['code'] == 0
        rsp = client.collection_list()

        all_collections = rsp['data']
        assert name in all_collections
        # describe collection
        rsp = client.collection_describe(name)
        assert rsp['code'] == 0
        assert rsp['data']['collectionName'] == name
        assert rsp['data']['autoId'] is False
        assert rsp['data']['enableDynamicField'] is True
        assert "COSINE" in str(rsp['data']["indexes"])

    @pytest.mark.parametrize("dim", [128])
    @pytest.mark.parametrize("metric_type", ["L2", "COSINE", "IP"])
    @pytest.mark.parametrize("id_type", ["Int64", "VarChar"])
    @pytest.mark.parametrize("primary_field", ["id", "url"])
    @pytest.mark.parametrize("vector_field", ["vector", "embedding"])
    def test_create_collection_quick_setup_with_custom(self, vector_field, primary_field, dim, id_type, metric_type):
        """
        Insert a vector with a simple payload
        """
        # create a collection
        name = gen_collection_name()
        collection_payload = {
            "collectionName": name,
            "dimension": dim,
            "metricType": metric_type,
            "primaryFieldName": primary_field,
            "vectorFieldName": vector_field,
            "idType": id_type,
        }
        if id_type == "VarChar":
            collection_payload["params"] = {"max_length": "256"}
        rsp = self.collection_client.collection_create(collection_payload)
        assert rsp['code'] == 0
        rsp = self.collection_client.collection_describe(name)
        logger.info(f"rsp: {rsp}")
        assert rsp['code'] == 0
        assert rsp['data']['collectionName'] == name
        fields = [f["name"] for f in rsp['data']['fields']]
        assert primary_field in fields
        assert vector_field in fields
        for f in rsp['data']['fields']:
            if f['name'] == primary_field:
                assert f['type'] == id_type
                assert f['primaryKey'] is True
        for index in rsp['data']['indexes']:
            assert index['metricType'] == metric_type

    @pytest.mark.parametrize("enable_dynamic_field", [False, "False", "0"])
    @pytest.mark.parametrize("request_shards_num", [2, "2"])
    @pytest.mark.parametrize("request_ttl_seconds", [360, "360"])
    def test_create_collections_without_params(self, enable_dynamic_field, request_shards_num, request_ttl_seconds):
        """
        target: test create collection
        method: create a collection with a simple schema
        expected: create collection success
        """
        name = gen_collection_name()
        dim = 128
        metric_type = "COSINE"
        client = self.collection_client
        num_shards = 2
        consistency_level = "Strong"
        ttl_seconds = 360
        payload = {
            "collectionName": name,
            "dimension": dim,
            "metricType": metric_type,
            "params": {
                "enableDynamicField": enable_dynamic_field,
                "shardsNum": request_shards_num,
                "consistencyLevel": f"{consistency_level}",
                "ttlSeconds": request_ttl_seconds,
            },
        }

        logging.info(f"create collection {name} with payload: {payload}")
        rsp = client.collection_create(payload)
        assert rsp['code'] == 0
        rsp = client.collection_list()

        all_collections = rsp['data']
        assert name in all_collections
        # describe collection by pymilvus
        c = Collection(name)
        res = c.describe()
        logger.info(f"describe collection: {res}")
        # describe collection
        time.sleep(10)
        rsp = client.collection_describe(name)
        logger.info(f"describe collection: {rsp}")

        ttl_seconds_actual = None
        for d in rsp["data"]["properties"]:
            if d["key"] == "collection.ttl.seconds":
                ttl_seconds_actual = int(d["value"])
        assert rsp['code'] == 0
        assert rsp['data']['enableDynamicField'] == False
        assert rsp['data']['collectionName'] == name
        assert rsp['data']['shardsNum'] == num_shards
        assert rsp['data']['consistencyLevel'] == consistency_level
        assert ttl_seconds_actual == ttl_seconds

    @pytest.mark.parametrize("primary_key_field", ["book_id"])
    @pytest.mark.parametrize("partition_key_field", ["word_count"])
    @pytest.mark.parametrize("clustering_key_field", ["book_category"])
    @pytest.mark.parametrize("shardsNum", [4])
    @pytest.mark.parametrize("partitionsNum", [16])
    @pytest.mark.parametrize("ttl_seconds", [60])
    @pytest.mark.parametrize("metric_type", ["L2", "COSINE", "IP"])
    @pytest.mark.parametrize("consistency_level", ["Strong", "Bounded"])
    @pytest.mark.parametrize("enable_dynamic_field", [True, False])
    @pytest.mark.parametrize("index_type", ["AUTOINDEX", "IVF_SQ8", "HNSW"])
    @pytest.mark.parametrize("dim", [128])
    def test_create_collections_with_all_params(self,
                                                dim,
                                                index_type,
                                                enable_dynamic_field,
                                                consistency_level,
                                                metric_type,
                                                ttl_seconds,
                                                partitionsNum,
                                                shardsNum,
                                                clustering_key_field,
                                                partition_key_field,
                                                primary_key_field,
                                                ):
        """
        target: test create collection
        method: create a collection with a simple schema
        expected: create collection success
        """
        name = gen_collection_name()
        dim = dim
        metric_type = metric_type
        client = self.collection_client
        num_shards = shardsNum
        num_partitions = partitionsNum
        consistency_level = consistency_level
        ttl_seconds = ttl_seconds
        index_param_map = {
            "FLAT": {},
            "IVF_SQ8": {"nlist": 16384},
            "HNSW": {"M": 16, "efConstruction": 500},
            "BM25_SPARSE_INVERTED_INDEX": {"bm25_k1": 0.5, "bm25_b": 0.5},
            "AUTOINDEX": {}
        }

        payload = {
            "collectionName": name,
            "params": {
                "shardsNum": f"{num_shards}",
                "partitionsNum": f"{num_partitions}",
                "consistencyLevel": f"{consistency_level}",
                "ttlSeconds": f"{ttl_seconds}",
            },
            "schema": {
                "enableDynamicField": enable_dynamic_field,
                "fields": [
                    {"fieldName": "user_id", "dataType": "Int64", "elementTypeParams": {}},
                    {"fieldName": "book_id", "dataType": "Int64",
                     "isPrimary": primary_key_field == "book_id", "elementTypeParams": {}},
                    {"fieldName": "word_count", "dataType": "Int64",
                     "isPartitionKey": partition_key_field == "word_count",
                     "isClusteringKey": clustering_key_field == "word_count", "elementTypeParams": {}},
                    {"fieldName": "book_category", "dataType": "Int64",
                     "isPartitionKey": partition_key_field == "book_category",
                     "isClusteringKey": clustering_key_field == "book_category", "elementTypeParams": {}},
                    {"fieldName": "book_describe", "dataType": "VarChar", "elementTypeParams": {"max_length": "256"}},
                    {"fieldName": "document_content", "dataType": "VarChar",
                     "elementTypeParams": {"max_length": "1000",
                                           "enable_analyzer": True,
                                           "analyzer_params": {
                                               "tokenizer": "standard"
                                           },
                                           "enable_match": True}},
                    {"fieldName": "json", "dataType": "JSON", "elementTypeParams": {}},
                    {"fieldName": "int_array", "dataType": "Array", "elementDataType": "Int64",
                     "elementTypeParams": {"max_capacity": "1024"}},
                    {"fieldName": "book_intro", "dataType": "FloatVector", "elementTypeParams": {"dim": f"{dim}"}},
                    {"fieldName": "sparse_vector", "dataType": "SparseFloatVector"}
                ],
                "functions": [
                    {
                        "name": "bm25_fn",
                        "type": "BM25",
                        "inputFieldNames": ["document_content"],
                        "outputFieldNames": ["sparse_vector"],
                        "params": {}
                    }
                ]
            },
            "indexParams": [
                {"fieldName": "book_intro",
                 "indexName": "book_intro_vector",
                 "metricType": f"{metric_type}",
                 "indexType": index_type,
                 "params": index_param_map[index_type]
                 },
                {"fieldName": "sparse_vector",
                 "indexName": "sparse_vector_index",
                 "metricType": "BM25",
                 "indexType": "SPARSE_INVERTED_INDEX",
                 "params": index_param_map["BM25_SPARSE_INVERTED_INDEX"]
                }
            ]
        }

        logging.info(f"create collection {name} with payload: {payload}")
        rsp = client.collection_create(payload)
        assert rsp['code'] == 0
        rsp = client.collection_list()

        all_collections = rsp['data']
        assert name in all_collections
        # describe collection by pymilvus
        c = Collection(name)
        res = c.describe()
        logger.info(f"pymilvus describe collection: {res}")
        # describe collection
        time.sleep(10)
        rsp = client.collection_describe(name)
        logger.info(f"restful describe collection: {rsp}")

        ttl_seconds_actual = None
        for d in rsp["data"]["properties"]:
            if d["key"] == "collection.ttl.seconds":
                ttl_seconds_actual = int(d["value"])
        assert rsp['code'] == 0
        assert rsp['data']['collectionName'] == name
        assert rsp['data']['enableDynamicField'] == enable_dynamic_field
        assert rsp['data']['shardsNum'] == num_shards
        assert rsp['data']['partitionsNum'] == num_partitions
        assert rsp['data']['consistencyLevel'] == consistency_level
        assert ttl_seconds_actual == ttl_seconds
        assert len(rsp['data']["functions"]) == len(payload["schema"]["functions"])
        #
        # # check fields properties
        fields = rsp['data']['fields']
        assert len(fields) == len(payload['schema']['fields'])
        for field in fields:
            if field['name'] == primary_key_field:
                assert field['primaryKey'] is True
            if field['name'] == partition_key_field:
                assert field['partitionKey'] is True
            if field['name'] == clustering_key_field:
                assert field['clusteringKey'] is True

        # check index
        index_info = [index.to_dict() for index in c.indexes]
        logger.info(f"index_info: {index_info}")
        assert len(index_info) == 2
        for index in index_info:
            index_param = index["index_param"]
            if index_param["index_type"] == "SPARSE_INVERTED_INDEX":
                assert index_param["metric_type"] == "BM25"
                assert index_param.get("params", {}) == index_param_map["BM25_SPARSE_INVERTED_INDEX"]
            else:
                assert index_param["metric_type"] == metric_type
                assert index_param["index_type"] == index_type
                assert index_param.get("params", {}) == index_param_map[index_type]

    @pytest.mark.parametrize("auto_id", [True, False])
    @pytest.mark.parametrize("enable_dynamic_field", [True, False])
    @pytest.mark.parametrize("enable_partition_key", [True, False])
    @pytest.mark.parametrize("dim", [128])
    def test_create_collections_custom_without_index(self, dim, auto_id, enable_dynamic_field, enable_partition_key):
        """
        target: test create collection
        method: create a collection with a simple schema
        expected: create collection success
        """
        name = gen_collection_name()
        dim = 128
        client = self.collection_client
        payload = {
            "collectionName": name,
            "schema": {
                "autoId": auto_id,
                "enableDynamicField": enable_dynamic_field,
                "fields": [
                    {"fieldName": "book_id", "dataType": "Int64", "isPrimary": True, "elementTypeParams": {}},
                    {"fieldName": "user_id", "dataType": "Int64", "isPartitionKey": enable_partition_key,
                     "elementTypeParams": {}},
                    {"fieldName": "word_count", "dataType": "Int64", "elementTypeParams": {}},
                    {"fieldName": "book_describe", "dataType": "VarChar", "elementTypeParams": {"max_length": "256"}},
                    {"fieldName": "book_intro", "dataType": "FloatVector", "elementTypeParams": {"dim": f"{dim}"}},
                    {"fieldName": "image_intro", "dataType": "FloatVector", "elementTypeParams": {"dim": f"{dim}"}},
                ]
            }
        }
        logging.info(f"create collection {name} with payload: {payload}")
        rsp = client.collection_create(payload)
        assert rsp['code'] == 0
        rsp = client.collection_list()

        all_collections = rsp['data']
        assert name in all_collections
        c = Collection(name)
        logger.info(f"schema: {c.schema}")
        # describe collection
        rsp = client.collection_describe(name)
        assert rsp['code'] == 0
        assert rsp['data']['collectionName'] == name
        assert rsp['data']['autoId'] == auto_id
        assert c.schema.auto_id == auto_id
        assert rsp['data']['enableDynamicField'] == enable_dynamic_field
        assert c.schema.enable_dynamic_field == enable_dynamic_field
        # assert no index created
        indexes = rsp['data']['indexes']
        assert len(indexes) == 0
        # assert not loaded
        assert rsp['data']['load'] == "LoadStateNotLoad"
        for field in rsp['data']['fields']:
            if field['name'] == "user_id":
                assert field['partitionKey'] == enable_partition_key
        for field in c.schema.fields:
            if field.name == "user_id":
                assert field.is_partition_key == enable_partition_key

    @pytest.mark.parametrize("metric_type", ["L2", "IP", "COSINE"])
    @pytest.mark.parametrize("dim", [128])
    def test_create_collections_one_float_vector_with_index(self, dim, metric_type):
        """
        target: test create collection
        method: create a collection with a simple schema
        expected: create collection success
        """
        name = gen_collection_name()
        dim = 128
        client = self.collection_client
        payload = {
            "collectionName": name,
            "schema": {
                "fields": [
                    {"fieldName": "book_id", "dataType": "Int64", "isPrimary": True, "elementTypeParams": {}},
                    {"fieldName": "word_count", "dataType": "Int64", "elementTypeParams": {}},
                    {"fieldName": "book_describe", "dataType": "VarChar", "elementTypeParams": {"max_length": "256"}},
                    {"fieldName": "book_intro", "dataType": "FloatVector", "elementTypeParams": {"dim": f"{dim}"}}
                ]
            },
            "indexParams": [
                {"fieldName": "book_intro", "indexName": "book_intro_vector", "metricType": f"{metric_type}"}]
        }
        logging.info(f"create collection {name} with payload: {payload}")
        rsp = client.collection_create(payload)
        assert rsp['code'] == 0
        rsp = client.collection_list()

        all_collections = rsp['data']
        assert name in all_collections
        # describe collection
        time.sleep(10)
        rsp = client.collection_describe(name)
        assert rsp['code'] == 0
        assert rsp['data']['collectionName'] == name
        # assert index created
        indexes = rsp['data']['indexes']
        assert len(indexes) == len(payload['indexParams'])
        # assert load success
        assert rsp['data']['load'] == "LoadStateLoaded"

    @pytest.mark.parametrize("metric_type", ["L2", "IP", "COSINE"])
    @pytest.mark.parametrize("dim", [128])
    def test_create_collections_multi_float_vector_with_one_index(self, dim, metric_type):
        """
        target: test create collection
        method: create a collection with a simple schema
        expected: create collection success
        """
        name = gen_collection_name()
        dim = 128
        client = self.collection_client
        payload = {
            "collectionName": name,
            "schema": {
                "fields": [
                    {"fieldName": "book_id", "dataType": "Int64", "isPrimary": True, "elementTypeParams": {}},
                    {"fieldName": "word_count", "dataType": "Int64", "elementTypeParams": {}},
                    {"fieldName": "book_describe", "dataType": "VarChar", "elementTypeParams": {"max_length": "256"}},
                    {"fieldName": "book_intro", "dataType": "FloatVector", "elementTypeParams": {"dim": f"{dim}"}},
                    {"fieldName": "image_intro", "dataType": "FloatVector", "elementTypeParams": {"dim": f"{dim}"}}
                ]
            },
            "indexParams": [
                {"fieldName": "book_intro", "indexName": "book_intro_vector", "metricType": f"{metric_type}"}]
        }
        logging.info(f"create collection {name} with payload: {payload}")
        rsp = client.collection_create(payload)
        assert rsp['code'] == 65535
        rsp = client.collection_list()

        all_collections = rsp['data']
        assert name in all_collections
        # describe collection
        time.sleep(10)
        rsp = client.collection_describe(name)
        assert rsp['code'] == 0
        assert rsp['data']['collectionName'] == name
        # assert index created
        indexes = rsp['data']['indexes']
        assert len(indexes) == len(payload['indexParams'])
        # assert load success
        assert rsp['data']['load'] == "LoadStateNotLoad"

    @pytest.mark.parametrize("metric_type", ["L2", "IP", "COSINE"])
    @pytest.mark.parametrize("dim", [128])
    def test_create_collections_multi_float_vector_with_all_index(self, dim, metric_type):
        """
        target: test create collection
        method: create a collection with a simple schema
        expected: create collection success
        """
        name = gen_collection_name()
        dim = 128
        client = self.collection_client
        payload = {
            "collectionName": name,
            "schema": {
                "fields": [
                    {"fieldName": "book_id", "dataType": "Int64", "isPrimary": True, "elementTypeParams": {}},
                    {"fieldName": "word_count", "dataType": "Int64", "elementTypeParams": {}},
                    {"fieldName": "book_describe", "dataType": "VarChar", "elementTypeParams": {"max_length": "256"}},
                    {"fieldName": "book_intro", "dataType": "FloatVector", "elementTypeParams": {"dim": f"{dim}"}},
                    {"fieldName": "image_intro", "dataType": "FloatVector", "elementTypeParams": {"dim": f"{dim}"}}
                ]
            },
            "indexParams": [
                {"fieldName": "book_intro", "indexName": "book_intro_vector", "metricType": f"{metric_type}"},
                {"fieldName": "image_intro", "indexName": "image_intro_vector", "metricType": f"{metric_type}"}]
        }
        logging.info(f"create collection {name} with payload: {payload}")
        rsp = client.collection_create(payload)
        assert rsp['code'] == 0
        rsp = client.collection_list()

        all_collections = rsp['data']
        assert name in all_collections
        # describe collection
        time.sleep(10)
        rsp = client.collection_describe(name)
        assert rsp['code'] == 0
        assert rsp['data']['collectionName'] == name
        # assert index created
        indexes = rsp['data']['indexes']
        assert len(indexes) == len(payload['indexParams'])
        # assert load success
        assert rsp['data']['load'] in ["LoadStateLoaded", "LoadStateLoading"]

    @pytest.mark.parametrize("auto_id", [True])
    @pytest.mark.parametrize("enable_dynamic_field", [True])
    @pytest.mark.parametrize("enable_partition_key", [True])
    @pytest.mark.parametrize("dim", [128])
    @pytest.mark.parametrize("metric_type", ["L2", "IP", "COSINE"])
    def test_create_collections_float16_vector_datatype(self, dim, auto_id, enable_dynamic_field, enable_partition_key,
                                                        metric_type):
        """
        target: test create collection
        method: create a collection with a simple schema
        expected: create collection success
        """
        name = gen_collection_name()
        dim = 128
        client = self.collection_client
        payload = {
            "collectionName": name,
            "schema": {
                "autoId": auto_id,
                "enableDynamicField": enable_dynamic_field,
                "fields": [
                    {"fieldName": "book_id", "dataType": "Int64", "isPrimary": True, "elementTypeParams": {}},
                    {"fieldName": "float16_vector", "dataType": "Float16Vector",
                     "elementTypeParams": {"dim": f"{dim}"}},
                    {"fieldName": "bfloat16_vector", "dataType": "BFloat16Vector",
                     "elementTypeParams": {"dim": f"{dim}"}},
                ]
            },
            "indexParams": [
                {"fieldName": "float16_vector", "indexName": "float16_vector_index", "metricType": f"{metric_type}"},
                {"fieldName": "bfloat16_vector", "indexName": "bfloat16_vector_index", "metricType": f"{metric_type}"}]

        }
        logging.info(f"create collection {name} with payload: {payload}")
        rsp = client.collection_create(payload)
        assert rsp['code'] == 0
        rsp = client.collection_list()

        all_collections = rsp['data']
        assert name in all_collections
        c = Collection(name)
        logger.info(f"schema: {c.schema}")
        # describe collection
        rsp = client.collection_describe(name)
        assert rsp['code'] == 0
        assert rsp['data']['collectionName'] == name
        assert len(rsp['data']['fields']) == len(c.schema.fields)

    @pytest.mark.parametrize("auto_id", [True])
    @pytest.mark.parametrize("enable_dynamic_field", [True])
    @pytest.mark.parametrize("enable_partition_key", [True])
    @pytest.mark.parametrize("dim", [128])
    @pytest.mark.parametrize("metric_type", ["JACCARD", "HAMMING"])
    def test_create_collections_binary_vector_datatype(self, dim, auto_id, enable_dynamic_field, enable_partition_key,
                                                       metric_type):
        """
        target: test create collection
        method: create a collection with a simple schema
        expected: create collection success
        """
        name = gen_collection_name()
        dim = 128
        client = self.collection_client
        payload = {
            "collectionName": name,
            "schema": {
                "autoId": auto_id,
                "enableDynamicField": enable_dynamic_field,
                "fields": [
                    {"fieldName": "book_id", "dataType": "Int64", "isPrimary": True, "elementTypeParams": {}},
                    {"fieldName": "binary_vector", "dataType": "BinaryVector", "elementTypeParams": {"dim": f"{dim}"}},
                ]
            },
            "indexParams": [
                {"fieldName": "binary_vector", "indexName": "binary_vector_index", "metricType": f"{metric_type}"}
            ]

        }
        logging.info(f"create collection {name} with payload: {payload}")
        rsp = client.collection_create(payload)
        assert rsp['code'] == 0
        rsp = client.collection_list()

        all_collections = rsp['data']
        assert name in all_collections
        c = Collection(name)
        logger.info(f"schema: {c.schema}")
        # describe collection
        rsp = client.collection_describe(name)
        assert rsp['code'] == 0
        assert rsp['data']['collectionName'] == name
        assert len(rsp['data']['fields']) == len(c.schema.fields)

    def test_create_collections_concurrent_with_same_param(self):
        """
        target: test create collection with same param
        method: concurrent create collections with same param with multi thread
        expected: create collections all success
        """
        concurrent_rsp = []

        def create_collection(c_name, vector_dim, c_metric_type):
            collection_payload = {
                "collectionName": c_name,
                "dimension": vector_dim,
                "metricType": c_metric_type,
            }
            rsp = client.collection_create(collection_payload)
            concurrent_rsp.append(rsp)
            logger.info(rsp)

        name = gen_collection_name()
        dim = 128
        metric_type = "L2"
        client = self.collection_client
        threads = []
        for i in range(10):
            t = threading.Thread(target=create_collection, args=(name, dim, metric_type,))
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        time.sleep(10)
        success_cnt = 0
        for rsp in concurrent_rsp:
            if rsp['code'] == 0:
                success_cnt += 1
        logger.info(concurrent_rsp)
        assert success_cnt == 10
        rsp = client.collection_list()
        all_collections = rsp['data']
        assert name in all_collections
        # describe collection
        rsp = client.collection_describe(name)
        assert rsp['code'] == 0
        assert rsp['data']['collectionName'] == name

    def test_create_collections_concurrent_with_different_param(self):
        """
        target: test create collection with different param
        method: concurrent create collections with different param with multi thread
        expected: only one collection can success
        """
        concurrent_rsp = []

        def create_collection(c_name, vector_dim, c_metric_type):
            collection_payload = {
                "collectionName": c_name,
                "dimension": vector_dim,
                "metricType": c_metric_type,
            }
            rsp = client.collection_create(collection_payload)
            concurrent_rsp.append(rsp)
            logger.info(rsp)

        name = gen_collection_name()
        dim = 128
        client = self.collection_client
        threads = []
        for i in range(0, 5):
            t = threading.Thread(target=create_collection, args=(name, dim + i, "L2",))
            threads.append(t)
        for i in range(5, 10):
            t = threading.Thread(target=create_collection, args=(name, dim + i, "IP",))
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        time.sleep(10)
        success_cnt = 0
        for rsp in concurrent_rsp:
            if rsp['code'] == 0:
                success_cnt += 1
        logger.info(concurrent_rsp)
        assert success_cnt == 1
        rsp = client.collection_list()
        all_collections = rsp['data']
        assert name in all_collections
        # describe collection
        rsp = client.collection_describe(name)
        assert rsp['code'] == 0
        assert rsp['data']['collectionName'] == name

    def test_create_collections_with_nullable_default(self):
        """
        target: test create collection
        method: create a collection with default none
        expected: create collection success
        """
        name = gen_collection_name()
        dim = 128
        client = self.collection_client
        payload = {
            "collectionName": name,
            "schema": {
                "fields": [
                    {"fieldName": "book_id", "dataType": "Int64", "isPrimary": True, "elementTypeParams": {}},
                    {"fieldName": "word_count", "dataType": "Int64", "elementTypeParams": {}, "defaultValue": 100},
                    {"fieldName": "book_describe", "dataType": "VarChar", "elementTypeParams": {"max_length": "256"},
                     "nullable": True, "defaultValue": "123"},
                    {"fieldName": "book_intro", "dataType": "FloatVector", "elementTypeParams": {"dim": f"{dim}"}}
                ]
            }
        }
        logging.info(f"create collection {name} with payload: {payload}")
        rsp = client.collection_create(payload)
        assert rsp['code'] == 0
        rsp = client.collection_list()

        all_collections = rsp['data']
        assert name in all_collections
        # describe collection
        rsp = client.collection_describe(name)
        assert rsp['code'] == 0
        assert rsp['data']['fields'][2]['defaultValue'] == {'Data': {'StringData': '123'}}
        assert rsp['data']['fields'][2]['nullable'] is True


@pytest.mark.L1
class TestCreateCollectionNegative(TestBase):

    def test_create_collections_custom_with_invalid_datatype(self):
        """
        VARCHAR is not a valid data type, it should be VarChar
        """
        name = gen_collection_name()
        dim = 128
        client = self.collection_client
        payload = {
            "collectionName": name,
            "schema": {
                "fields": [
                    {"fieldName": "book_id", "dataType": "Int64", "isPrimary": True, "elementTypeParams": {}},
                    {"fieldName": "word_count", "dataType": "Int64", "elementTypeParams": {}},
                    {"fieldName": "book_describe", "dataType": "VARCHAR", "elementTypeParams": {"max_length": "256"}},
                    {"fieldName": "book_intro", "dataType": "FloatVector", "elementTypeParams": {"dim": f"{dim}"}}
                ]
            }
        }
        logging.info(f"create collection {name} with payload: {payload}")
        rsp = client.collection_create(payload)
        assert rsp['code'] == 1100

    def test_create_collections_custom_with_invalid_params(self):
        """
        """
        name = gen_collection_name()
        dim = 128
        client = self.collection_client
        payload = {
            "collectionName": name,
            "schema": {
                "enableDynamicField": 1,
                "fields": [
                    {"fieldName": "book_id", "dataType": "Int64", "isPrimary": True, "elementTypeParams": {}},
                    {"fieldName": "word_count", "dataType": "Int64", "elementTypeParams": {}},
                    {"fieldName": "book_describe", "dataType": "VarChar", "elementTypeParams": {"max_length": "256"}},
                    {"fieldName": "book_intro", "dataType": "FloatVector", "elementTypeParams": {"dim": f"{dim}"}}
                ]
            }
        }
        logging.info(f"create collection {name} with payload: {payload}")
        rsp = client.collection_create(payload)
        assert rsp['code'] == 1801

    @pytest.mark.parametrize("name",
                             [" ", "test_collection_" * 100, "test collection", "test/collection", "test\collection"])
    def test_create_collections_with_invalid_collection_name(self, name):
        """
        target: test create collection with invalid collection name
        method: create collections with invalid collection name
        expected: create collection failed with right error message
        """
        dim = 128
        client = self.collection_client
        payload = {
            "collectionName": name,
            "dimension": dim,
        }
        rsp = client.collection_create(payload)
        assert rsp['code'] == 1100
        assert "Invalid collection name" in rsp['message'] or "invalid parameter" in rsp['message']

    def test_create_collections_with_partition_key_nullable(self):
        """
        partition key field not support nullable
        """
        name = gen_collection_name()
        dim = 128
        client = self.collection_client
        payload = {
            "collectionName": name,
            "schema": {
                "fields": [
                    {"fieldName": "book_id", "dataType": "Int64", "isPrimary": True, "elementTypeParams": {}},
                    {"fieldName": "word_count", "dataType": "Int64", "elementTypeParams": {}, "isPartitionKey": True,
                     "nullable": True},
                    {"fieldName": "book_describe", "dataType": "VarChar", "elementTypeParams": {"max_length": "256"}},
                    {"fieldName": "book_intro", "dataType": "FloatVector", "elementTypeParams": {"dim": f"{dim}"}}
                ]
            }
        }
        logging.info(f"create collection {name} with payload: {payload}")
        rsp = client.collection_create(payload)
        assert rsp['code'] == 1100
        assert "partition key field not support nullable" in rsp['message']

    def test_create_collections_with_vector_nullable(self):
        """
        vector field not support nullable
        """
        name = gen_collection_name()
        dim = 128
        client = self.collection_client
        payload = {
            "collectionName": name,
            "schema": {
                "fields": [
                    {"fieldName": "book_id", "dataType": "Int64", "isPrimary": True, "elementTypeParams": {}},
                    {"fieldName": "word_count", "dataType": "Int64", "elementTypeParams": {}},
                    {"fieldName": "book_describe", "dataType": "VarChar", "elementTypeParams": {"max_length": "256"}},
                    {"fieldName": "book_intro", "dataType": "FloatVector", "elementTypeParams": {"dim": f"{dim}"},
                     "nullable": True}
                ]
            }
        }
        logging.info(f"create collection {name} with payload: {payload}")
        rsp = client.collection_create(payload)
        assert rsp['code'] == 1100
        assert "vector type not support null" in rsp['message']

    def test_create_collections_with_primary_default(self):
        """
        primary key field not support defaultValue
        """
        name = gen_collection_name()
        dim = 128
        client = self.collection_client
        payload = {
            "collectionName": name,
            "schema": {
                "fields": [
                    {"fieldName": "book_id", "dataType": "Int64", "isPrimary": True, "elementTypeParams": {},
                     "defaultValue": 123},
                    {"fieldName": "book_describe", "dataType": "VarChar", "elementTypeParams": {"max_length": "256"}},
                    {"fieldName": "book_intro", "dataType": "FloatVector", "elementTypeParams": {"dim": f"{dim}"}}
                ]
            }
        }
        logging.info(f"create collection {name} with payload: {payload}")
        rsp = client.collection_create(payload)
        assert rsp['code'] == 1100
        assert "primary field not support default_value" in rsp['message']

    def test_create_collections_with_json_field_default(self):
        """
        json field not support default value
        """
        name = gen_collection_name()
        dim = 128
        client = self.collection_client
        payload = {
            "collectionName": name,
            "schema": {
                "fields": [
                    {"fieldName": "book_id", "dataType": "Int64", "isPrimary": True, "elementTypeParams": {}},
                    {"fieldName": "word_count", "dataType": "Int64", "elementTypeParams": {}},
                    {"fieldName": "book_describe", "dataType": "VarChar", "elementTypeParams": {"max_length": "256"}},
                    {"fieldName": "json", "dataType": "JSON", "elementTypeParams": {}, "defaultValue": {"key": 1}},
                    {"fieldName": "book_intro", "dataType": "FloatVector", "elementTypeParams": {"dim": f"{dim}"}}
                ]
            }
        }
        logging.info(f"create collection {name} with payload: {payload}")
        rsp = client.collection_create(payload)
        assert rsp['code'] == 1100
        assert "convert defaultValue fail" in rsp['message']

    def test_create_collections_with_array_field_default(self):
        """
        array field not support default value
        """
        name = gen_collection_name()
        dim = 128
        client = self.collection_client
        payload = {
            "collectionName": name,
            "schema": {
                "fields": [
                    {"fieldName": "book_id", "dataType": "Int64", "isPrimary": True, "elementTypeParams": {}},
                    {"fieldName": "word_count", "dataType": "Int64", "elementTypeParams": {}},
                    {"fieldName": "int_array", "dataType": "Array", "elementDataType": "Int64", "defaultValue": [1, 2],
                     "elementTypeParams": {"max_capacity": "1024"}},
                    {"fieldName": "book_intro", "dataType": "FloatVector", "elementTypeParams": {"dim": f"{dim}"}}
                ]
            }
        }
        logging.info(f"create collection {name} with payload: {payload}")
        rsp = client.collection_create(payload)
        assert rsp['code'] == 1100
        assert "convert defaultValue fail" in rsp['message']


@pytest.mark.L0
class TestHasCollections(TestBase):

    def test_has_collections_default(self):
        """
        target: test list collection with a simple schema
        method: create collections and list them
        expected: created collections are in list
        """
        client = self.collection_client
        name_list = []
        for i in range(2):
            name = gen_collection_name()
            dim = 128
            payload = {
                "collectionName": name,
                "metricType": "L2",
                "dimension": dim,
            }
            time.sleep(1)
            rsp = client.collection_create(payload)
            assert rsp['code'] == 0
            name_list.append(name)
        rsp = client.collection_list()
        all_collections = rsp['data']
        for name in name_list:
            assert name in all_collections
            rsp = client.collection_has(collection_name=name)
            assert rsp['data']['has'] is True

    def test_has_collections_with_not_exist_name(self):
        """
        target: test list collection with a simple schema
        method: create collections and list them
        expected: created collections are in list
        """
        client = self.collection_client
        name_list = []
        for i in range(2):
            name = gen_collection_name()
            name_list.append(name)
        rsp = client.collection_list()
        all_collections = rsp['data']
        for name in name_list:
            assert name not in all_collections
            rsp = client.collection_has(collection_name=name)
            assert rsp['data']['has'] is False


@pytest.mark.L0
class TestGetCollectionStats(TestBase):

    def test_get_collections_stats(self):
        """
        target: test list collection with a simple schema
        method: create collections and list them
        expected: created collections are in list
        """
        client = self.collection_client
        name = gen_collection_name()
        dim = 128
        payload = {
            "collectionName": name,
            "metricType": "L2",
            "dimension": dim,
        }
        rsp = client.collection_create(payload)
        assert rsp['code'] == 0
        # describe collection
        client.collection_describe(collection_name=name)
        rsp = client.collection_stats(collection_name=name)
        assert rsp['code'] == 0
        assert rsp['data']['rowCount'] == 0
        # insert data
        nb = 3000
        data = get_data_by_payload(payload, nb)
        payload = {
            "collectionName": name,
            "data": data
        }
        self.vector_client.vector_insert(payload=payload)
        c = Collection(name)
        count = c.query(expr="", output_fields=["count(*)"])
        logger.info(f"count: {count}")
        c.flush()
        rsp = client.collection_stats(collection_name=name)
        assert rsp['data']['rowCount'] == nb


@pytest.mark.L0
class TestLoadReleaseCollection(TestBase):

    def test_load_and_release_collection(self):
        name = gen_collection_name()
        dim = 128
        client = self.collection_client
        payload = {
            "collectionName": name,
            "schema": {
                "fields": [
                    {"fieldName": "book_id", "dataType": "Int64", "isPrimary": True, "elementTypeParams": {}},
                    {"fieldName": "word_count", "dataType": "Int64", "elementTypeParams": {}},
                    {"fieldName": "book_describe", "dataType": "VarChar", "elementTypeParams": {"max_length": "256"}},
                    {"fieldName": "book_intro", "dataType": "FloatVector", "elementTypeParams": {"dim": f"{dim}"}}
                ]
            }
        }
        rsp = client.collection_create(payload)
        assert rsp['code'] == 0
        # create index before load
        index_params = [{"fieldName": "book_intro", "indexName": "book_intro_vector", "metricType": "L2"}]
        payload = {
            "collectionName": name,
            "indexParams": index_params
        }
        rsp = self.index_client.index_create(payload)

        # get load state before load
        rsp = client.collection_load_state(collection_name=name)
        assert rsp['data']['loadState'] == "LoadStateNotLoad"

        # describe collection
        client.collection_describe(collection_name=name)
        rsp = client.collection_load(collection_name=name)
        assert rsp['code'] == 0
        rsp = client.collection_load_state(collection_name=name)
        assert rsp['data']['loadState'] in ["LoadStateLoaded", "LoadStateLoading"]
        time.sleep(5)
        rsp = client.collection_load_state(collection_name=name)
        assert rsp['data']['loadState'] == "LoadStateLoaded"

        # release collection
        rsp = client.collection_release(collection_name=name)
        time.sleep(5)
        rsp = client.collection_load_state(collection_name=name)
        assert rsp['data']['loadState'] == "LoadStateNotLoad"


@pytest.mark.L0
class TestGetCollectionLoadState(TestBase):

    def test_get_collection_load_state(self):
        """
        target: test list collection with a simple schema
        method: create collections and list them
        expected: created collections are in list
        """
        client = self.collection_client
        name = gen_collection_name()
        dim = 128
        payload = {
            "collectionName": name,
            "metricType": "L2",
            "dimension": dim,
        }
        rsp = client.collection_create(payload)
        assert rsp['code'] == 0
        # describe collection
        client.collection_describe(collection_name=name)
        rsp = client.collection_load_state(collection_name=name)
        assert rsp['code'] == 0
        t0 = time.time()
        while time.time() - t0 < 10:
            rsp = client.collection_load_state(collection_name=name)
            if rsp['data']['loadState'] != "LoadStateNotLoad":
                break
            time.sleep(1)
        assert rsp['data']['loadState'] in ["LoadStateLoading", "LoadStateLoaded"]
        # insert data
        nb = 3000
        data = get_data_by_payload(payload, nb)
        payload = {
            "collectionName": name,
            "data": data
        }
        self.vector_client.vector_insert(payload=payload)
        rsp = client.collection_load_state(collection_name=name)
        assert rsp['data']['loadState'] in ["LoadStateLoading", "LoadStateLoaded"]
        time.sleep(10)
        rsp = client.collection_load_state(collection_name=name)
        assert rsp['data']['loadState'] == "LoadStateLoaded"


@pytest.mark.L0
class TestListCollections(TestBase):

    def test_list_collections_default(self):
        """
        target: test list collection with a simple schema
        method: create collections and list them
        expected: created collections are in list
        """
        client = self.collection_client
        name_list = []
        for i in range(2):
            name = gen_collection_name()
            dim = 128
            payload = {
                "collectionName": name,
                "metricType": "L2",
                "dimension": dim,
            }
            time.sleep(1)
            rsp = client.collection_create(payload)
            assert rsp['code'] == 0
            name_list.append(name)
        rsp = client.collection_list()
        all_collections = rsp['data']
        for name in name_list:
            assert name in all_collections


@pytest.mark.L0
class TestDescribeCollection(TestBase):

    def test_describe_collections_default(self):
        """
        target: test describe collection with a simple schema
        method: describe collection
        expected: info of description is same with param passed to create collection
        """
        name = gen_collection_name()
        dim = 128
        client = self.collection_client
        payload = {
            "collectionName": name,
            "dimension": dim,
            "metricType": "L2"
        }
        rsp = client.collection_create(payload)
        assert rsp['code'] == 0
        rsp = client.collection_list()
        all_collections = rsp['data']
        assert name in all_collections
        # describe collection
        rsp = client.collection_describe(name)
        assert rsp['code'] == 0
        assert rsp['data']['collectionName'] == name
        assert rsp['data']['autoId'] is False
        assert rsp['data']['enableDynamicField'] is True
        assert len(rsp['data']['indexes']) == 1

    def test_describe_collections_custom(self):
        """
        target: test describe collection with a simple schema
        method: describe collection
        expected: info of description is same with param passed to create collection
        """
        name = gen_collection_name()
        dim = 128
        client = self.collection_client
        fields = [
            FieldSchema(name='reviewer_id', dtype=DataType.INT64, description="", is_primary=True),
            FieldSchema(name='store_address', dtype=DataType.VARCHAR, description="", max_length=512,
                        is_partition_key=True),
            FieldSchema(name='review', dtype=DataType.VARCHAR, description="", max_length=16384),
            FieldSchema(name='vector', dtype=DataType.FLOAT_VECTOR, description="", dim=384, is_index=True),
        ]

        schema = CollectionSchema(
            fields=fields,
            description="",
            enable_dynamic_field=True,
            # The following is an alternative to setting `is_partition_key` in a field schema.
            partition_key_field="store_address"
        )

        collection = Collection(
            name=name,
            schema=schema,
        )
        logger.info(f"schema: {schema}")
        rsp = client.collection_list()
        all_collections = rsp['data']
        assert name in all_collections
        # describe collection
        rsp = client.collection_describe(name)
        assert rsp['code'] == 0
        assert rsp['data']['collectionName'] == name

        for field in rsp['data']['fields']:
            if field['name'] == "store_address":
                assert field['partitionKey'] is True
            if field['name'] == "reviewer_id":
                assert field['primaryKey'] is True
        assert rsp['data']['autoId'] is False
        assert rsp['data']['enableDynamicField'] is True


@pytest.mark.L0
class TestDescribeCollectionNegative(TestBase):

    def test_describe_collections_with_invalid_collection_name(self):
        """
        target: test describe collection with invalid collection name
        method: describe collection with invalid collection name
        expected: raise error with right error code and message
        """
        name = gen_collection_name()
        dim = 128
        client = self.collection_client
        payload = {
            "collectionName": name,
            "dimension": dim,
        }
        rsp = client.collection_create(payload)
        assert rsp['code'] == 0
        rsp = client.collection_list()
        all_collections = rsp['data']
        assert name in all_collections
        # describe collection
        invalid_name = "invalid_name"
        rsp = client.collection_describe(invalid_name)
        assert rsp['code'] == 100
        assert "can't find collection" in rsp['message']


@pytest.mark.L0
class TestDropCollection(TestBase):
    def test_drop_collections_default(self):
        """
        Drop a collection with a simple schema
        target: test drop collection with a simple schema
        method: drop collection
        expected: dropped collection was not in collection list
        """
        clo_list = []
        for i in range(5):
            time.sleep(1)
            name = 'test_collection_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f_%f")
            payload = {
                "collectionName": name,
                "dimension": 128,
                "metricType": "L2"
            }
            rsp = self.collection_client.collection_create(payload)
            assert rsp['code'] == 0
            clo_list.append(name)
        rsp = self.collection_client.collection_list()
        all_collections = rsp['data']
        for name in clo_list:
            assert name in all_collections
        for name in clo_list:
            time.sleep(0.2)
            payload = {
                "collectionName": name,
            }
            rsp = self.collection_client.collection_drop(payload)
            assert rsp['code'] == 0
        rsp = self.collection_client.collection_list()
        all_collections = rsp['data']
        for name in clo_list:
            assert name not in all_collections


@pytest.mark.L0
class TestDropCollectionNegative(TestBase):

    def test_drop_collections_with_invalid_collection_name(self):
        """
        target: test drop collection with invalid collection name
        method: drop collection with invalid collection name
        expected: raise error with right error code and message
        """
        name = gen_collection_name()
        dim = 128
        client = self.collection_client
        payload = {
            "collectionName": name,
            "dimension": dim,
        }
        rsp = client.collection_create(payload)
        assert rsp['code'] == 0
        rsp = client.collection_list()
        all_collections = rsp['data']
        assert name in all_collections
        # drop collection
        invalid_name = "invalid_name"
        payload = {
            "collectionName": invalid_name,
        }
        rsp = client.collection_drop(payload)
        assert rsp['code'] == 0


@pytest.mark.L0
class TestRenameCollection(TestBase):

    def test_rename_collection(self):
        """
        target: test rename collection
        method: rename collection
        expected: renamed collection is in collection list
        """
        name = gen_collection_name()
        dim = 128
        client = self.collection_client
        payload = {
            "collectionName": name,
            "metricType": "L2",
            "dimension": dim,
        }
        rsp = client.collection_create(payload)
        assert rsp['code'] == 0
        rsp = client.collection_list()
        all_collections = rsp['data']
        assert name in all_collections
        new_name = gen_collection_name()
        payload = {
            "collectionName": name,
            "newCollectionName": new_name,
        }
        rsp = client.collection_rename(payload)
        assert rsp['code'] == 0
        rsp = client.collection_list()
        all_collections = rsp['data']
        assert new_name in all_collections
        assert name not in all_collections


@pytest.mark.L1
class TestCollectionWithAuth(TestBase):
    def test_drop_collections_with_invalid_api_key(self):
        """
        target: test drop collection with invalid api key
        method: drop collection with invalid api key
        expected: raise error with right error code and message; collection still in collection list
        """
        name = gen_collection_name()
        dim = 128
        client = self.collection_client
        payload = {
            "collectionName": name,
            "dimension": dim,
        }
        rsp = client.collection_create(payload)
        assert rsp['code'] == 0
        rsp = client.collection_list()
        all_collections = rsp['data']
        assert name in all_collections
        # drop collection
        payload = {
            "collectionName": name,
        }
        illegal_client = CollectionClient(self.endpoint, "invalid_api_key")
        rsp = illegal_client.collection_drop(payload)
        assert rsp['code'] == 1800
        rsp = client.collection_list()
        all_collections = rsp['data']
        assert name in all_collections

    def test_describe_collections_with_invalid_api_key(self):
        """
        target: test describe collection with invalid api key
        method: describe collection with invalid api key
        expected: raise error with right error code and message
        """
        name = gen_collection_name()
        dim = 128
        client = self.collection_client
        payload = {
            "collectionName": name,
            "dimension": dim,
        }
        rsp = client.collection_create(payload)
        assert rsp['code'] == 0
        rsp = client.collection_list()
        all_collections = rsp['data']
        assert name in all_collections
        # describe collection
        illegal_client = CollectionClient(self.endpoint, "illegal_api_key")
        rsp = illegal_client.collection_describe(name)
        assert rsp['code'] == 1800

    def test_list_collections_with_invalid_api_key(self):
        """
        target: test list collection with an invalid api key
        method: list collection with invalid api key
        expected: raise error with right error code and message
        """
        client = self.collection_client
        name_list = []
        for i in range(2):
            name = gen_collection_name()
            dim = 128
            payload = {
                "collectionName": name,
                "metricType": "L2",
                "dimension": dim,
            }
            time.sleep(1)
            rsp = client.collection_create(payload)
            assert rsp['code'] == 0
            name_list.append(name)
        client = self.collection_client
        client.api_key = "illegal_api_key"
        rsp = client.collection_list()
        assert rsp['code'] == 1800

    def test_create_collections_with_invalid_api_key(self):
        """
        target: test create collection with invalid api key(wrong username and password)
        method: create collections with invalid api key
        expected: create collection failed
        """
        name = gen_collection_name()
        dim = 128
        client = self.collection_client
        client.api_key = "illegal_api_key"
        payload = {
            "collectionName": name,
            "dimension": dim,
        }
        rsp = client.collection_create(payload)
        assert rsp['code'] == 1800


@pytest.mark.L0
class TestCollectionProperties(TestBase):
    """Test collection property operations"""

    def test_refresh_load_collection(self):
        """
        target: test refresh load collection
        method: create collection, refresh load
        expected: refresh load success
        """
        name = gen_collection_name()
        dim = 128
        client = self.collection_client
        payload = {
            "collectionName": name,
            "dimension": dim,
        }
        rsp = client.collection_create(payload)
        assert rsp['code'] == 0

        # release collection
        client.collection_release(collection_name=name)
        # load collection
        client.collection_load(collection_name=name)
        client.wait_load_completed(collection_name=name)
        # refresh load
        rsp = client.refresh_load(collection_name=name)

        assert rsp['code'] == 0

    def test_alter_collection_properties(self):
        """
        target: test alter collection properties
        method: create collection, alter properties
        expected: alter properties success
        """
        name = gen_collection_name()
        dim = 128
        client = self.collection_client
        payload = {
            "collectionName": name,
            "dimension": dim,
        }
        rsp = client.collection_create(payload)
        assert rsp['code'] == 0
        client.collection_release(collection_name=name)
        # alter properties
        properties = {"mmap.enabled": "true"}
        rsp = client.alter_collection_properties(name, properties)
        assert rsp['code'] == 0
        rsp = client.collection_describe(name)
        enabled_mmap = False
        for prop in rsp['data']['properties']:
            if prop['key'] == "mmap.enabled":
                assert prop['value'] == "true"
                enabled_mmap = True
        assert enabled_mmap

    def test_drop_collection_properties(self):
        """
        target: test drop collection properties
        method: create collection, alter properties, drop properties
        expected: drop properties success
        """
        name = gen_collection_name()
        dim = 128
        client = self.collection_client
        payload = {
            "collectionName": name,
            "dimension": dim,
        }
        rsp = client.collection_create(payload)
        assert rsp['code'] == 0
        client.collection_release(collection_name=name)

        # alter properties
        properties = {"mmap.enabled": "true"}
        rsp = client.alter_collection_properties(name, properties)
        assert rsp['code'] == 0
        rsp = client.collection_describe(name)
        enabled_mmap = False
        for prop in rsp['data']['properties']:
            if prop['key'] == "mmap.enabled":
                assert prop['value'] == "true"
                enabled_mmap = True
        assert enabled_mmap

        # drop properties
        delete_keys = ["mmap.enabled"]
        rsp = client.drop_collection_properties(name, delete_keys)
        assert rsp['code'] == 0
        rsp = client.collection_describe(name)
        enabled_mmap = False
        for prop in rsp['data']['properties']:
            if prop['key'] == "mmap.enabled":
                enabled_mmap = True
        assert not enabled_mmap

    def test_alter_field_properties(self):
        """
        target: test alter field properties
        method: create collection with varchar field, alter field properties
        expected: alter field properties success
        """
        name = gen_collection_name()
        dim = 128
        client = self.collection_client
        payload = {
            "collectionName": name,
            "schema": {
                "autoId": True,
                "enableDynamicField": True,
                "fields": [
                    {"fieldName": "book_id", "dataType": "Int64", "isPrimary": True, "elementTypeParams": {}},
                    {"fieldName": "user_id", "dataType": "Int64", "isPartitionKey": True,
                     "elementTypeParams": {}},
                    {"fieldName": "word_count", "dataType": "Int64", "elementTypeParams": {}},
                    {"fieldName": "book_describe", "dataType": "VarChar", "elementTypeParams": {"max_length": "256"}},
                    {"fieldName": "book_intro", "dataType": "FloatVector", "elementTypeParams": {"dim": f"{dim}"}},
                    {"fieldName": "image_intro", "dataType": "FloatVector", "elementTypeParams": {"dim": f"{dim}"}},
                ]
            }
        }
        rsp = client.collection_create(payload)
        assert rsp['code'] == 0
        # release collection
        client.collection_release(collection_name=name)

        # describe collection
        rsp = client.collection_describe(name)
        for field in rsp['data']['fields']:
            if field['name'] == "book_describe":
                for p in field['params']:
                    if p['key'] == "max_length":
                        assert p['value'] == "256"

        # alter field properties
        field_params = {"max_length": "100"}
        rsp = client.alter_field_properties(name, "book_describe", field_params)
        assert rsp['code'] == 0

        # describe collection
        rsp = client.collection_describe(name)
        for field in rsp['data']['fields']:
            if field['name'] == "book_describe":
                for p in field['params']:
                    if p['key'] == "max_length":
                        assert p['value'] == "100"


class TestCollectionAddField(TestBase):
    """Test collection add field operations"""
    
    @pytest.mark.parametrize("field_params,test_data_generator,expected_validations", [
        # Test case 1: Int64 nullable field
        (
            {
                "fieldName": "new_int_field",
                "dataType": "Int64",
                "nullable": True,
                "elementTypeParams": {}
            },
            lambda i: i * 10,  # Generate int values
            {
                "field_type": "Int64",
                "nullable": True,
                "has_default": False,
                "data_validator": lambda item, i: item["new_int_field"] == i * 10
            }
        ),
        # Test case 2: VarChar field with max_length
        (
            {
                "fieldName": "new_varchar_field",
                "dataType": "VarChar",
                "nullable": True,
                "elementTypeParams": {"max_length": "256"}
            },
            lambda i: f"description_{i}",  # Generate string values
            {
                "field_type": "VarChar",
                "nullable": True,
                "has_default": False,
                "max_length": "256",
                "data_validator": lambda item, i: item["new_varchar_field"] == f"description_{i}"
            }
        ),
        # Test case 3: Int64 field with default value
        (
            {
                "fieldName": "new_field_with_default",
                "dataType": "Int64",
                "nullable": True,
                "defaultValue": 42,
                "elementTypeParams": {}
            },
            lambda i: i * 100,  # Generate int values when explicitly provided
            {
                "field_type": "Int64",
                "nullable": True,
                "has_default": True,
                "default_value": 42,
                "data_validator": lambda item, i: item["new_field_with_default"] == i * 100
            }
        ),
        # Test case 4: Array field
        (
            {
                "fieldName": "new_array_field",
                "dataType": "Array",
                "elementDataType": "Int64",
                "nullable": True,
                "elementTypeParams": {"max_capacity": "1024"}
            },
            lambda i: [i * 10, i * 20, i * 30],  # Generate array values
            {
                "field_type": "Array",
                "nullable": True,
                "has_default": False,
                "element_type": "Int64",
                "data_validator": lambda item, i: item["new_array_field"] == [i * 10, i * 20, i * 30]
            }
        )
    ])
    def test_add_field_parametrized(self, field_params, test_data_generator, expected_validations):
        """
        target: test add various types of fields
        method: create collection, insert data, add field, insert and query again
        expected: add field success and data operations work before and after
        """
        name = gen_collection_name()
        dim = 128
        nb = 3000  # Number of records to insert in each batch
        client = self.collection_client
        vector_client = self.vector_client
        field_name = field_params["fieldName"]
        
        # Create collection first
        payload = {
            "collectionName": name,
            "schema": {
                "fields": [
                    {"fieldName": "book_id", "dataType": "Int64", "isPrimary": True, "elementTypeParams": {}},
                    {"fieldName": "book_intro", "dataType": "FloatVector", "elementTypeParams": {"dim": f"{dim}"}}
                ]
            },
            "indexParams": [
                {"fieldName": "book_intro", "indexName": "book_intro_index", "metricType": "L2"}
            ]
        }
        rsp = client.collection_create(payload)
        assert rsp['code'] == 0
        
        # Wait for collection to be loaded
        client.wait_load_completed(collection_name=name)
        
        # Insert data before adding field
        insert_data_before = []
        for i in range(nb):
            insert_data_before.append({
                "book_id": i,
                "book_intro": gen_vector(dim=dim)
            })
        
        insert_payload = {
            "collectionName": name,
            "data": insert_data_before
        }
        rsp = vector_client.vector_insert(insert_payload)
        assert rsp['code'] == 0
        
        # Query data before adding field
        query_payload = {
            "collectionName": name,
            "expr": "book_id >= 0",
            "outputFields": ["book_id"],
            "limit": nb
        }
        rsp = vector_client.vector_query(query_payload)
        assert rsp['code'] == 0
        assert len(rsp['data']) == nb
        
        # Search data before adding field
        search_payload = {
            "collectionName": name,
            "data": [gen_vector(dim=dim)],
            "annsField": "book_intro",
            "limit": 100
        }
        rsp = vector_client.vector_search(search_payload)
        assert rsp['code'] == 0
        assert len(rsp['data']) > 0
        
        # Add field
        rsp = client.add_field(name, field_params)
        logger.info(f"add field response: {rsp}")
        assert rsp['code'] == 0
        
        # Verify field was added
        rsp = client.collection_describe(name)
        assert rsp['code'] == 0
        field_names = [field["name"] for field in rsp['data']['fields']]
        assert field_name in field_names
        
        # Check the field properties
        for field in rsp['data']['fields']:
            if field['name'] == field_name:
                assert field['type'] == expected_validations["field_type"]
                assert field['nullable'] == expected_validations["nullable"]
                
                # Check specific field type properties
                if expected_validations.get("max_length"):
                    for param in field.get('params', []):
                        if param['key'] == 'max_length':
                            assert param['value'] == expected_validations["max_length"]
                
                if expected_validations.get("element_type"):
                    assert field.get('elementType') == expected_validations["element_type"]
                
                if expected_validations.get("has_default") and expected_validations["has_default"]:
                    assert field.get('defaultValue') is not None
        
        # Insert data after adding field
        insert_data_after = []
        for i in range(nb, nb * 2):
            data_item = {
                "book_id": i,
                "book_intro": gen_vector(dim=dim)
            }
            
            # For default value test, sometimes omit the field to test default behavior
            if expected_validations.get("has_default") and expected_validations["has_default"] and i < nb + nb // 2:
                # Don't add the field for first half of records to test default value
                pass
            else:
                # Add the field with generated test data
                data_item[field_name] = test_data_generator(i)
            
            insert_data_after.append(data_item)
        
        insert_payload = {
            "collectionName": name,
            "data": insert_data_after
        }
        rsp = vector_client.vector_insert(insert_payload)
        assert rsp['code'] == 0
        
        # Query data after adding field
        query_payload = {
            "collectionName": name,
            "expr": f"book_id >= {nb}",
            "outputFields": ["book_id", field_name],
            "limit": nb
        }
        rsp = vector_client.vector_query(query_payload)
        assert rsp['code'] == 0
        assert len(rsp['data']) == nb
        
        # Validate field data for records that have explicit values
        for item in rsp['data']:
            assert field_name in item
            book_id = item["book_id"]
            # Only validate explicit values (not default values)
            if not (expected_validations.get("has_default") and expected_validations["has_default"] and book_id < nb + nb // 2):
                if expected_validations.get("data_validator"):
                    expected_validations["data_validator"](item, book_id)
        
        # Search data after adding field
        search_payload = {
            "collectionName": name,
            "data": [gen_vector(dim=dim)],
            "annsField": "book_intro",
            "limit": 100,
            "outputFields": ["book_id", field_name]
        }
        rsp = vector_client.vector_search(search_payload)
        assert rsp['code'] == 0
        assert len(rsp['data']) > 0


@pytest.mark.L1
class TestCollectionAddFieldNegative(TestBase):
    """Test collection add field negative cases"""

    def test_add_field_missing_data_type(self):
        """
        target: test add field with missing dataType
        method: create collection, add field without dataType parameter
        expected: add field failed with proper error message
        """
        name = gen_collection_name()
        dim = 128
        client = self.collection_client
        
        # Create collection first
        payload = {
            "collectionName": name,
            "schema": {
                "fields": [
                    {"fieldName": "book_id", "dataType": "Int64", "isPrimary": True, "elementTypeParams": {}},
                    {"fieldName": "book_intro", "dataType": "FloatVector", "elementTypeParams": {"dim": f"{dim}"}}
                ]
            }
        }
        rsp = client.collection_create(payload)
        assert rsp['code'] == 0
        
        # Try to add field without dataType
        field_params = {
            "fieldName": "new_field",
            "nullable": True,
            "elementTypeParams": {}
        }
        rsp = client.add_field(name, field_params)
        logger.info(f"add field response: {rsp}")
        assert rsp['code'] != 0
        assert "dataType" in rsp.get('message', '').lower() or "required" in rsp.get('message', '').lower()

    def test_add_field_invalid_default_value_type(self):
        """
        target: test add field with invalid defaultValue type
        method: create collection, add Int64 field with string defaultValue
        expected: add field failed with proper error message
        """
        name = gen_collection_name()
        dim = 128
        client = self.collection_client
        
        # Create collection first
        payload = {
            "collectionName": name,
            "schema": {
                "fields": [
                    {"fieldName": "book_id", "dataType": "Int64", "isPrimary": True, "elementTypeParams": {}},
                    {"fieldName": "book_intro", "dataType": "FloatVector", "elementTypeParams": {"dim": f"{dim}"}}
                ]
            }
        }
        rsp = client.collection_create(payload)
        assert rsp['code'] == 0
        
        # Try to add Int64 field with string defaultValue
        field_params = {
            "fieldName": "new_field",
            "dataType": "Int64",
            "nullable": True,
            "defaultValue": "aaa",  # Invalid type for Int64
            "elementTypeParams": {}
        }
        rsp = client.add_field(name, field_params)
        logger.info(f"add field response: {rsp}")
        assert rsp['code'] != 0
        assert "defaultValue" in rsp.get('message', '') or "invalid" in rsp.get('message', '').lower()

    def test_add_field_invalid_data_type(self):
        """
        target: test add field with invalid dataType
        method: create collection, add field with invalid dataType
        expected: add field failed with proper error message
        """
        name = gen_collection_name()
        dim = 128
        client = self.collection_client
        
        # Create collection first
        payload = {
            "collectionName": name,
            "schema": {
                "fields": [
                    {"fieldName": "book_id", "dataType": "Int64", "isPrimary": True, "elementTypeParams": {}},
                    {"fieldName": "book_intro", "dataType": "FloatVector", "elementTypeParams": {"dim": f"{dim}"}}
                ]
            }
        }
        rsp = client.collection_create(payload)
        assert rsp['code'] == 0
        
        # Try to add field with invalid dataType
        field_params = {
            "fieldName": "new_field",
            "dataType": "LONGLONGLONGLONGTEXT",  # Invalid dataType
            "nullable": True,
            "elementTypeParams": {}
        }
        rsp = client.add_field(name, field_params)
        logger.info(f"add field response: {rsp}")
        assert rsp['code'] != 0
        assert "invalid" in rsp.get('message', '').lower() or "data type" in rsp.get('message', '').lower()

    def test_add_field_array_missing_element_data_type(self):
        """
        target: test add Array field without elementDataType
        method: create collection, add Array field without elementDataType
        expected: add field failed with proper error message
        """
        name = gen_collection_name()
        dim = 128
        client = self.collection_client
        
        # Create collection first
        payload = {
            "collectionName": name,
            "schema": {
                "fields": [
                    {"fieldName": "book_id", "dataType": "Int64", "isPrimary": True, "elementTypeParams": {}},
                    {"fieldName": "book_intro", "dataType": "FloatVector", "elementTypeParams": {"dim": f"{dim}"}}
                ]
            }
        }
        rsp = client.collection_create(payload)
        assert rsp['code'] == 0
        
        # Try to add Array field without elementDataType
        field_params = {
            "fieldName": "new_array_field",
            "dataType": "Array",
            "nullable": True,
            "elementTypeParams": {}
        }
        rsp = client.add_field(name, field_params)
        logger.info(f"add field response: {rsp}")
        assert rsp['code'] != 0
        assert "element" in rsp.get('message', '').lower() or "invalid" in rsp.get('message', '').lower()

    def test_add_field_array_invalid_element_data_type(self):
        """
        target: test add Array field with invalid elementDataType
        method: create collection, add Array field with invalid elementDataType
        expected: add field failed with proper error message
        """
        name = gen_collection_name()
        dim = 128
        client = self.collection_client
        
        # Create collection first
        payload = {
            "collectionName": name,
            "schema": {
                "fields": [
                    {"fieldName": "book_id", "dataType": "Int64", "isPrimary": True, "elementTypeParams": {}},
                    {"fieldName": "book_intro", "dataType": "FloatVector", "elementTypeParams": {"dim": f"{dim}"}}
                ]
            }
        }
        rsp = client.collection_create(payload)
        assert rsp['code'] == 0
        
        # Try to add Array field with invalid elementDataType
        field_params = {
            "fieldName": "new_array_field",
            "dataType": "Array",
            "elementDataType": "MYBLOB",  # Invalid elementDataType
            "nullable": True,
            "elementTypeParams": {}
        }
        rsp = client.add_field(name, field_params)
        logger.info(f"add field response: {rsp}")
        assert rsp['code'] != 0
        assert "element" in rsp.get('message', '').lower() or "invalid" in rsp.get('message', '').lower()

    def test_add_field_to_nonexistent_collection(self):
        """
        target: test add field to non-existent collection
        method: add field to a collection that doesn't exist
        expected: add field failed with proper error message
        """
        name = "nonexistent_collection"
        client = self.collection_client
        
        # Try to add field to non-existent collection
        field_params = {
            "fieldName": "new_field",
            "dataType": "Int64",
            "nullable": True,
            "elementTypeParams": {}
        }
        rsp = client.add_field(name, field_params)
        logger.info(f"add field response: {rsp}")
        assert rsp['code'] != 0
        assert "collection" in rsp.get('message', '').lower() or "not found" in rsp.get('message', '').lower()


@pytest.mark.L0
class TestCollectionMaintenance(TestBase):
    """Test collection maintenance operations"""

    @pytest.mark.xfail(reason="issue: https://github.com/milvus-io/milvus/issues/39546")
    def test_collection_flush(self):
        """
        target: test collection flush
        method: create collection, insert data multiple times and flush
        expected: flush successfully
        """
        # Create collection
        name = gen_collection_name()
        client = self.collection_client
        vector_client = self.vector_client
        payload = {
            "collectionName": name,
            "schema": {
                "fields": [
                    {"fieldName": "book_id", "dataType": "Int64", "isPrimary": True, "elementTypeParams": {}},
                    {"fieldName": "my_vector", "dataType": "FloatVector", "elementTypeParams": {"dim": 128}}
                ]
            }
        }
        client.collection_create(payload)

        # Insert small batches of data multiple times
        for i in range(3):
            vectors = [gen_vector(dim=128) for _ in range(10)]
            insert_data = {
                "collectionName": name,
                "data": [
                    {
                        "book_id": i * 10 + j,
                        "my_vector": vector
                    }
                    for i, vector in enumerate(vectors)
                    for j in range(10)
                ]
            }
            response = vector_client.vector_insert(insert_data)
            assert response["code"] == 0
        c = Collection(name)
        num_entities_before_flush = c.num_entities
        # Flush collection
        response = client.flush(name)
        assert response["code"] == 0
        # check segments
        num_entities_after_flush = c.num_entities
        logger.info(f"num_entities_before_flush: {num_entities_before_flush}, num_entities_after_flush: {num_entities_after_flush}")
        assert num_entities_after_flush > num_entities_before_flush

    def test_collection_compact(self):
        """
        target: test collection compact
        method: create collection, insert data, flush multiple times, then compact
        expected: compact successfully
        """
        # Create collection
        name = gen_collection_name()
        client = self.collection_client
        vector_client = self.vector_client
        payload = {
            "collectionName": name,
            "schema": {
                "fields": [
                    {"fieldName": "book_id", "dataType": "Int64", "isPrimary": True, "elementTypeParams": {}},
                    {"fieldName": "my_vector", "dataType": "FloatVector", "elementTypeParams": {"dim": 128}}
                ]
            }
        }
        client.collection_create(payload)

        # Insert and flush multiple times
        for i in range(3):
            # Insert data
            vectors = [gen_vector(dim=128) for _ in range(10)]
            insert_data = {
                "collectionName": name,
                "data": [
                    {
                        "book_id": i * 10 + j,
                        "my_vector": vector
                    }
                    for i, vector in enumerate(vectors)
                    for j in range(10)
                ]
            }
            response = vector_client.vector_insert(insert_data)
            assert response["code"] == 0

            # Flush after each insert
            c = Collection(name)
            c.flush()
        # Compact collection
        response = client.compact(name)
        assert response["code"] == 0

        # Get compaction state
        response = client.get_compaction_state(name)
        assert response["code"] == 0
        assert "state" in response["data"]
        assert "compactionID" in response["data"]
        # TODO need verification by pymilvus

