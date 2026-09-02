import io
import time

import numpy as np
import pytest
from base.client_v2_base import TestMilvusClientV2Base
from common import common_func as cf
from common import common_type as ct
from common.common_type import CaseLabel, CheckTasks
from pymilvus import DataType, Function, FunctionChain, FunctionChainStage, FunctionScore, FunctionType
from pymilvus.function_chain import col, fn
from pymilvus.function_chain.chain import FunctionChainExpr

prefix = "function_chain"


class TestFunctionChain(TestMilvusClientV2Base):
    """Test pymilvus FunctionChain SDK integration."""

    dim = 2
    vector_field = "vector"
    scalar_field = "ts"

    def _create_function_chain_collection(self, client):
        collection_name = cf.gen_unique_str(prefix)
        schema = self.create_schema(client, auto_id=False, enable_dynamic_field=False)[0]
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field(self.scalar_field, DataType.INT64)
        schema.add_field(self.vector_field, DataType.FLOAT_VECTOR, dim=self.dim)

        return self._create_collection_with_schema_and_rows(
            client,
            collection_name,
            schema,
            [
                {"id": 1, self.scalar_field: 10, self.vector_field: [0.0, 0.0]},
                {"id": 2, self.scalar_field: 20, self.vector_field: [0.01, 0.0]},
                {"id": 3, self.scalar_field: 30, self.vector_field: [0.02, 0.0]},
            ],
        )

    def _create_json_dynamic_collection(self, client, multi_segment=False, metric_type="L2"):
        collection_name = cf.gen_unique_str(prefix)
        schema = self.create_schema(client, auto_id=False, enable_dynamic_field=True)[0]
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("base_score", DataType.INT64)
        schema.add_field("metadata", DataType.JSON, nullable=True)
        schema.add_field(self.vector_field, DataType.FLOAT_VECTOR, dim=self.dim)

        rows = [
            {
                "id": 1,
                "base_score": 1,
                self.vector_field: [0.00, 0.0],
                "metadata": {
                    "rank": 100,
                    "ratio": 0.0,
                    "mixed": 10,
                    "enabled": True,
                    "group": "A",
                    "items": [{"name": "delta"}],
                },
                "profile": {
                    "bonus": 0.0,
                    "mixed": 10,
                    "group": "A",
                    "label": "delta",
                    "enabled": True,
                },
            },
            {
                "id": 2,
                "base_score": 2,
                self.vector_field: [0.01, 0.0],
                "metadata": {
                    "rank": 0,
                    "ratio": 90.0,
                    "mixed": "100",
                    "enabled": False,
                    "group": "A",
                    "items": [{"name": "alpha"}],
                },
                "profile": {
                    "bonus": 90.0,
                    "mixed": "100",
                    "group": "A",
                    "label": "alpha",
                    "enabled": False,
                },
            },
            {
                "id": 3,
                "base_score": 3,
                self.vector_field: [0.02, 0.0],
                "metadata": {
                    "rank": 60,
                    "ratio": 20.0,
                    "mixed": None,
                    "enabled": True,
                    "group": "B",
                    "items": [{"name": "charlie"}],
                },
                "profile": {
                    "bonus": 20.0,
                    "mixed": None,
                    "group": "B",
                    "label": "charlie",
                    "enabled": True,
                },
            },
            {
                "id": 4,
                "base_score": 4,
                self.vector_field: [0.03, 0.0],
                "metadata": {
                    "rank": 50,
                    "ratio": 60.0,
                    "mixed": 20,
                    "enabled": False,
                    "group": "B",
                    "items": [{"name": "bravo"}],
                },
                "profile": {
                    "bonus": 60.0,
                    "mixed": 20,
                    "group": "B",
                    "label": "bravo",
                    "enabled": False,
                },
            },
            {
                "id": 5,
                "base_score": 5,
                self.vector_field: [0.04, 0.0],
                "metadata": {
                    "rank": 25,
                    "ratio": 10.0,
                    "enabled": True,
                    "group": "C",
                    "items": [],
                },
                "profile": {
                    "bonus": 10.0,
                    "group": "C",
                    "label": "echo",
                    "enabled": True,
                },
            },
            {
                "id": 6,
                "base_score": 6,
                self.vector_field: [0.05, 0.0],
                "metadata": None,
                "profile": {"bonus": 70.0},
            },
        ]
        if not multi_segment:
            return self._create_collection_with_schema_and_rows(
                client,
                collection_name,
                schema,
                rows,
                metric_type=metric_type,
            )

        index_params = self.prepare_index_params(client)[0]
        index_params.add_index(
            field_name=self.vector_field,
            index_type="FLAT",
            metric_type=metric_type,
        )
        self.create_collection(
            client,
            collection_name,
            schema=schema,
            index_params=index_params,
            consistency_level="Strong",
        )
        self.alter_collection_properties(
            client,
            collection_name,
            properties={"collection.autocompaction.enabled": "false"},
        )
        segment_rows = [rows[:3], rows[3:]]
        for batch in segment_rows:
            self.insert(client, collection_name, batch)
            self.flush(client, collection_name)

        assert self.wait_for_index_ready(client, collection_name, index_name=self.vector_field)
        self.load_collection(client, collection_name)

        deadline = time.time() + 60
        loaded_segments = []
        while time.time() < deadline:
            loaded_segments = client.list_loaded_segments(collection_name)
            if len(loaded_segments) == len(segment_rows):
                break
            time.sleep(1)
        assert len(loaded_segments) == len(segment_rows), loaded_segments
        assert sorted(segment.num_rows for segment in loaded_segments) == [3, 3]
        return collection_name

    def _create_l2_json_dynamic_collection(self, client):
        return self._create_json_dynamic_collection(client)

    @staticmethod
    def _set_input_data_types(chain, op_index, *data_types):
        chain.ops[op_index].params["$input_data_types"] = [int(data_type) for data_type in data_types]
        return chain

    def _create_collection_with_schema_and_rows(
        self,
        client,
        collection_name,
        schema,
        rows,
        metric_type="L2",
    ):
        index_params = self.prepare_index_params(client)[0]
        index_params.add_index(
            field_name=self.vector_field,
            index_type="FLAT",
            metric_type=metric_type,
        )
        self.create_collection(
            client,
            collection_name,
            schema=schema,
            index_params=index_params,
            consistency_level="Strong",
        )
        self.insert(client, collection_name, rows)
        self.flush(client, collection_name)
        self.load_collection(client, collection_name)
        return collection_name

    def _create_l1_function_chain_collection(self, client):
        collection_name = cf.gen_unique_str(prefix)
        schema = self.create_schema(client, auto_id=False, enable_dynamic_field=False)[0]
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field(self.scalar_field, DataType.INT64)
        schema.add_field("payload", DataType.INT64)
        schema.add_field("category", DataType.INT64)
        schema.add_field(self.vector_field, DataType.FLOAT_VECTOR, dim=self.dim)

        index_params = self.prepare_index_params(client)[0]
        index_params.add_index(field_name=self.vector_field, index_type="FLAT", metric_type="COSINE")
        self.create_collection(
            client,
            collection_name,
            schema=schema,
            index_params=index_params,
            consistency_level="Strong",
        )
        self.alter_collection_properties(
            client,
            collection_name,
            properties={"collection.autocompaction.enabled": "false"},
        )

        segment_rows = [
            [
                {
                    "id": 1,
                    self.scalar_field: 10,
                    "payload": 101,
                    "category": 1,
                    self.vector_field: [1.0, 0.0],
                },
                {
                    "id": 2,
                    self.scalar_field: 60,
                    "payload": 202,
                    "category": 2,
                    self.vector_field: [0.8, 0.6],
                },
                {
                    "id": 3,
                    self.scalar_field: 30,
                    "payload": 303,
                    "category": 3,
                    self.vector_field: [0.6, 0.8],
                },
            ],
            [
                {
                    "id": 4,
                    self.scalar_field: 50,
                    "payload": 404,
                    "category": 4,
                    self.vector_field: [0.0, 1.0],
                },
                {
                    "id": 5,
                    self.scalar_field: 20,
                    "payload": 505,
                    "category": 5,
                    self.vector_field: [-0.6, 0.8],
                },
                {
                    "id": 6,
                    self.scalar_field: 40,
                    "payload": 606,
                    "category": 6,
                    self.vector_field: [-1.0, 0.0],
                },
            ],
        ]
        for rows in segment_rows:
            self.insert(client, collection_name, rows)
            self.flush(client, collection_name)

        assert self.wait_for_index_ready(client, collection_name, index_name=self.vector_field)
        self.load_collection(client, collection_name)

        deadline = time.time() + 60
        loaded_segments = []
        while time.time() < deadline:
            loaded_segments = client.list_loaded_segments(collection_name)
            if len(loaded_segments) == len(segment_rows):
                break
            time.sleep(1)
        assert len(loaded_segments) == len(segment_rows), loaded_segments
        assert sorted(segment.num_rows for segment in loaded_segments) == [3, 3]
        return collection_name

    def _score_plus_ts_chain(self, stage):
        chain = FunctionChain(stage, name="score_plus_ts").map(
            "$score",
            fn.num_combine(col("$score"), col(self.scalar_field), mode="sum"),
        )
        if stage == FunctionChainStage.L2_RERANK:
            chain.sort(col("$score"), desc=True, tie_break_col=col("$id"))
        return chain

    def _assert_search_error(self, client, collection_name, function_chains, err_msg, **kwargs):
        self.search(
            client,
            collection_name,
            data=[[0.0, 0.0]],
            anns_field=self.vector_field,
            search_params={"metric_type": "L2"},
            limit=3,
            function_chains=function_chains,
            check_task=CheckTasks.err_res,
            check_items={ct.err_code: 1100, ct.err_msg: err_msg},
            **kwargs,
        )

    @staticmethod
    def _generate_xgboost_model(tmp_path):
        xgb = pytest.importorskip("xgboost")
        features = np.array([[0.1], [0.8], [0.2], [0.9]], dtype=np.float32)
        labels = np.array([0.2, 0.9, 0.4, 0.7], dtype=np.float32)
        dtrain = xgb.DMatrix(features, label=labels)
        booster = xgb.train(
            {
                "objective": "reg:squarederror",
                "max_depth": 2,
                "eta": 1.0,
                "lambda": 0.0,
                "alpha": 0.0,
                "base_score": 0.5,
                "tree_method": "exact",
                "seed": 7,
            },
            dtrain,
            num_boost_round=1,
        )
        model_path = tmp_path / "xgboost_l0_rerank.ubj"
        booster.save_model(model_path)
        expected = booster.predict(dtrain, output_margin=True).astype(float).tolist()
        return model_path, features[:, 0].astype(float).tolist(), expected

    @staticmethod
    def _generate_unsupported_xgboost_model(tmp_path, name, params):
        xgb = pytest.importorskip("xgboost")
        features = np.array([[0.1], [0.8], [0.2], [0.9]], dtype=np.float32)
        labels = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        dtrain = xgb.DMatrix(features, label=labels)
        train_params = {
            "objective": "reg:squarederror",
            "max_depth": 2,
            "eta": 1.0,
            "lambda": 0.0,
            "alpha": 0.0,
            "base_score": 0.5,
            "seed": 7,
        }
        train_params.update(params)
        if train_params.get("objective") == "rank:pairwise":
            dtrain.set_group([len(labels)])
        booster = xgb.train(train_params, dtrain, num_boost_round=1)
        model_path = tmp_path / f"{name}.ubj"
        booster.save_model(model_path)
        return model_path

    @staticmethod
    def _new_minio_client(minio_host):
        from minio import Minio

        return Minio(
            f"{minio_host}:9000",
            access_key="minioadmin",
            secret_key="minioadmin",
            secure=False,
        )

    def _upload_file_resource_bytes(self, client, minio_host, bucket, resource_name, remote_path, data):
        minio_client = self._new_minio_client(minio_host)
        if not minio_client.bucket_exists(bucket):
            minio_client.make_bucket(bucket)
        minio_client.put_object(bucket, remote_path, io.BytesIO(data), len(data))
        self.add_file_resource(client, resource_name, remote_path)
        return minio_client

    def _create_l0_xgboost_collection(self, client, fields, rows):
        collection_name = cf.gen_unique_str(prefix)
        schema = self.create_schema(client, auto_id=False, enable_dynamic_field=False)[0]
        schema.add_field("id", DataType.INT64, is_primary=True)
        for name, data_type, kwargs in fields:
            schema.add_field(name, data_type, **kwargs)
        schema.add_field(self.vector_field, DataType.FLOAT_VECTOR, dim=self.dim)
        return self._create_collection_with_schema_and_rows(client, collection_name, schema, rows)

    @staticmethod
    def _l0_xgboost_chain(resource_name, feature_columns, output="raw"):
        return FunctionChain(FunctionChainStage.L0_RERANK, name="l0_xgboost").map(
            "$score",
            FunctionChainExpr(
                "xgboost",
                args=tuple(col(name) for name in feature_columns),
                params={"model_resource": resource_name, "output": output},
            ),
        )

    def _assert_l0_xgboost_search_error(self, client, collection_name, chain, err_msg, limit=3):
        self.search(
            client,
            collection_name,
            data=[[0.0, 0.0]],
            anns_field=self.vector_field,
            search_params={"metric_type": "L2"},
            limit=limit,
            function_chains=chain,
            check_task=CheckTasks.err_res,
            check_items={ct.err_code: 1100, ct.err_msg: err_msg},
        )

    @staticmethod
    def _hit_field(hit, field):
        if field in hit:
            return hit[field]
        return hit.get("entity", {}).get(field)

    @staticmethod
    def _expected_l0_score(hit):
        vector = hit.get("entity", {}).get("vector", hit.get("vector"))
        l2_distance = sum(value * value for value in vector)
        return TestFunctionChain._hit_field(hit, "ts") - l2_distance

    @pytest.mark.tags(CaseLabel.L0)
    def test_search_with_l0_function_chain_xgboost_matches_local_predict(self, file_resource_env, tmp_path, minio_host):
        """
        target: test L0 function chain can rerank search results with a real XGBoost UBJ model
        method: generate a tiny XGBoost model locally, upload it as a Milvus file resource, run L0 xgboost rerank
        expected: Milvus search scores and order match local XGBoost raw predictions
        """
        from minio import Minio

        client = self._client()
        resource_name = cf.gen_unique_str("xgboost_model")
        remote_path = f"xgboost/{resource_name}.ubj"
        collection_name = cf.gen_unique_str(prefix)
        model_path, feature_values, expected_scores = self._generate_xgboost_model(tmp_path)

        minio_client = Minio(
            f"{minio_host}:9000",
            access_key="minioadmin",
            secret_key="minioadmin",
            secure=False,
        )
        bucket = file_resource_env["bucket"]
        if not minio_client.bucket_exists(bucket):
            minio_client.make_bucket(bucket)
        model_bytes = model_path.read_bytes()
        minio_client.put_object(bucket, remote_path, io.BytesIO(model_bytes), len(model_bytes))

        try:
            self.add_file_resource(client, resource_name, remote_path)

            schema = self.create_schema(client, auto_id=False, enable_dynamic_field=False)[0]
            schema.add_field("id", DataType.INT64, is_primary=True)
            schema.add_field("xgb_f0", DataType.FLOAT)
            schema.add_field(self.vector_field, DataType.FLOAT_VECTOR, dim=self.dim)
            rows = [
                {
                    "id": idx + 1,
                    "xgb_f0": value,
                    self.vector_field: [idx * 0.01, 0.0],
                }
                for idx, value in enumerate(feature_values)
            ]
            self._create_collection_with_schema_and_rows(client, collection_name, schema, rows)

            chain = FunctionChain(FunctionChainStage.L0_RERANK, name="l0_xgboost").map(
                "$score",
                FunctionChainExpr(
                    "xgboost",
                    args=(col("xgb_f0"),),
                    params={"model_resource": resource_name, "output": "raw"},
                ),
            )
            res, _ = self.search(
                client,
                collection_name,
                data=[[0.0, 0.0]],
                anns_field=self.vector_field,
                search_params={"metric_type": "L2"},
                limit=len(rows),
                output_fields=["xgb_f0"],
                function_chains=chain,
            )

            expected_by_id = {idx + 1: score for idx, score in enumerate(expected_scores)}
            expected_ids = [
                idx + 1 for idx, _ in sorted(enumerate(expected_scores), key=lambda item: item[1], reverse=True)
            ]
            assert [hit["id"] for hit in res[0]] == expected_ids
            for hit in res[0]:
                assert abs(hit["distance"]) == pytest.approx(expected_by_id[hit["id"]], rel=1e-5, abs=1e-5)
        finally:
            try:
                client.remove_file_resource(name=resource_name)
            except Exception:
                pass
            try:
                minio_client.remove_object(bucket, remote_path)
            except Exception:
                pass

    @pytest.mark.tags(CaseLabel.L0)
    def test_search_rejects_l0_function_chain_xgboost_missing_resource(self):
        """
        target: test L0 xgboost rejects a model_resource that is not registered
        method: run xgboost rerank with a missing FileResource name
        expected: search fails with file resource not found
        """
        client = self._client()
        rows = [
            {"id": 1, "xgb_f0": 0.1, self.vector_field: [0.0, 0.0]},
            {"id": 2, "xgb_f0": 0.8, self.vector_field: [0.01, 0.0]},
        ]
        collection_name = self._create_l0_xgboost_collection(
            client,
            [("xgb_f0", DataType.FLOAT, {})],
            rows,
        )
        chain = self._l0_xgboost_chain("missing_xgboost_model", ["xgb_f0"])

        self._assert_l0_xgboost_search_error(client, collection_name, chain, "file resource")

    @pytest.mark.tags(CaseLabel.L0)
    def test_search_rejects_l0_function_chain_xgboost_invalid_output_param(self):
        """
        target: test L0 xgboost rejects an invalid output parameter
        method: run xgboost rerank with output=probability
        expected: request fails because output must be default or raw
        """
        client = self._client()
        rows = [
            {"id": 1, "xgb_f0": 0.1, self.vector_field: [0.0, 0.0]},
            {"id": 2, "xgb_f0": 0.8, self.vector_field: [0.01, 0.0]},
        ]
        collection_name = self._create_l0_xgboost_collection(
            client,
            [("xgb_f0", DataType.FLOAT, {})],
            rows,
        )
        chain = self._l0_xgboost_chain("unused_xgboost_model", ["xgb_f0"], output="probability")

        self._assert_l0_xgboost_search_error(client, collection_name, chain, "output must be one of")

    @pytest.mark.tags(CaseLabel.L0)
    def test_search_rejects_l0_function_chain_xgboost_feature_count_mismatch(
        self, file_resource_env, tmp_path, minio_host
    ):
        """
        target: test L0 xgboost rejects feature count mismatches
        method: use a one-feature model with two input feature columns
        expected: search fails with feature column count mismatch
        """
        client = self._client()
        resource_name = cf.gen_unique_str("xgboost_model")
        remote_path = f"xgboost/{resource_name}.ubj"
        model_path, feature_values, _ = self._generate_xgboost_model(tmp_path)
        bucket = file_resource_env["bucket"]
        model_bytes = model_path.read_bytes()
        minio_client = self._upload_file_resource_bytes(
            client, minio_host, bucket, resource_name, remote_path, model_bytes
        )

        try:
            rows = [
                {
                    "id": idx + 1,
                    "xgb_f0": value,
                    "xgb_f1": value + 1.0,
                    self.vector_field: [idx * 0.01, 0.0],
                }
                for idx, value in enumerate(feature_values)
            ]
            collection_name = self._create_l0_xgboost_collection(
                client,
                [("xgb_f0", DataType.FLOAT, {}), ("xgb_f1", DataType.FLOAT, {})],
                rows,
            )
            chain = self._l0_xgboost_chain(resource_name, ["xgb_f0", "xgb_f1"])

            self._assert_l0_xgboost_search_error(
                client, collection_name, chain, "expected 1 feature columns, got 2", limit=len(rows)
            )
        finally:
            try:
                client.remove_file_resource(name=resource_name)
            except Exception:
                pass
            try:
                minio_client.remove_object(bucket, remote_path)
            except Exception:
                pass

    @pytest.mark.tags(CaseLabel.L0)
    def test_search_rejects_l0_function_chain_xgboost_unsupported_input_type(
        self, file_resource_env, tmp_path, minio_host
    ):
        """
        target: test L0 xgboost rejects unsupported input column types
        method: pass a varchar field as an xgboost feature
        expected: search fails with unsupported input column type
        """
        client = self._client()
        resource_name = cf.gen_unique_str("xgboost_model")
        remote_path = f"xgboost/{resource_name}.ubj"
        model_path, feature_values, _ = self._generate_xgboost_model(tmp_path)
        bucket = file_resource_env["bucket"]
        model_bytes = model_path.read_bytes()
        minio_client = self._upload_file_resource_bytes(
            client, minio_host, bucket, resource_name, remote_path, model_bytes
        )

        try:
            rows = [
                {"id": idx + 1, "xgb_text": str(value), self.vector_field: [idx * 0.01, 0.0]}
                for idx, value in enumerate(feature_values)
            ]
            collection_name = self._create_l0_xgboost_collection(
                client,
                [("xgb_text", DataType.VARCHAR, {"max_length": 64})],
                rows,
            )
            chain = self._l0_xgboost_chain(resource_name, ["xgb_text"])

            self._assert_l0_xgboost_search_error(
                client, collection_name, chain, "unsupported input column type", limit=len(rows)
            )
        finally:
            try:
                client.remove_file_resource(name=resource_name)
            except Exception:
                pass
            try:
                minio_client.remove_object(bucket, remote_path)
            except Exception:
                pass

    @pytest.mark.tags(CaseLabel.L0)
    @pytest.mark.parametrize(
        "case_name, model_data, expected_error",
        [
            ("not_ubj", b'{"learner":{}}', "failed to parse UBJ model"),
            ("unsupported_objective", None, "unsupported objective"),
            ("unsupported_booster", None, "unsupported booster"),
        ],
    )
    def test_search_rejects_l0_function_chain_xgboost_invalid_model(
        self, file_resource_env, tmp_path, minio_host, case_name, model_data, expected_error
    ):
        """
        target: test L0 xgboost rejects invalid or unsupported model artifacts
        method: register invalid UBJ content, unsupported objective, and unsupported booster models
        expected: search fails while loading the xgboost model
        """
        client = self._client()
        resource_name = cf.gen_unique_str(f"xgboost_{case_name}")
        remote_path = f"xgboost/{resource_name}.ubj"
        if model_data is None:
            if case_name == "unsupported_objective":
                model_path = self._generate_unsupported_xgboost_model(
                    tmp_path, case_name, {"objective": "rank:pairwise"}
                )
            else:
                model_path = self._generate_unsupported_xgboost_model(tmp_path, case_name, {"booster": "gblinear"})
            model_data = model_path.read_bytes()
        bucket = file_resource_env["bucket"]
        minio_client = self._upload_file_resource_bytes(
            client, minio_host, bucket, resource_name, remote_path, model_data
        )

        try:
            rows = [
                {"id": 1, "xgb_f0": 0.1, self.vector_field: [0.0, 0.0]},
                {"id": 2, "xgb_f0": 0.8, self.vector_field: [0.01, 0.0]},
            ]
            collection_name = self._create_l0_xgboost_collection(
                client,
                [("xgb_f0", DataType.FLOAT, {})],
                rows,
            )
            chain = self._l0_xgboost_chain(resource_name, ["xgb_f0"])

            self._assert_l0_xgboost_search_error(client, collection_name, chain, expected_error, limit=len(rows))
        finally:
            try:
                client.remove_file_resource(name=resource_name)
            except Exception:
                pass
            try:
                minio_client.remove_object(bucket, remote_path)
            except Exception:
                pass

    @pytest.mark.tags(CaseLabel.L0)
    @pytest.mark.parametrize(
        "stage",
        [FunctionChainStage.L0_RERANK, FunctionChainStage.L1_RERANK, FunctionChainStage.L2_RERANK],
        ids=["l0", "l1", "l2"],
    )
    @pytest.mark.parametrize("field_name", ["threshold", "interval", "iso"])
    @pytest.mark.parametrize("enable_dynamic_field", [False, True], ids=["static", "dynamic"])
    def test_search_with_function_chain_scalar_keyword_field(self, stage, field_name, enable_dynamic_field):
        """
        target: test legal scalar field names that are also expression parser keywords
        method: sum the keyword field with itself as a hidden or returned rerank input at each stage
        expected: both queries have exact rewritten scores, reordered IDs, and aligned output fields
        """
        client = self._client()
        collection_name = cf.gen_unique_str(prefix)
        schema = self.create_schema(client, auto_id=False, enable_dynamic_field=enable_dynamic_field)[0]
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field(field_name, DataType.DOUBLE)
        schema.add_field("payload", DataType.INT64)
        schema.add_field(self.vector_field, DataType.FLOAT_VECTOR, dim=self.dim)
        rows = [
            {"id": 1, field_name: 10.0, "payload": 101, self.vector_field: [1.0, 0.0]},
            {"id": 2, field_name: 30.0, "payload": 202, self.vector_field: [0.0, 1.0]},
            {"id": 3, field_name: 20.0, "payload": 303, self.vector_field: [-1.0, 0.0]},
        ]
        self._create_collection_with_schema_and_rows(client, collection_name, schema, rows, metric_type="COSINE")
        chain = FunctionChain(stage, name="scalar_keyword").map(
            "$score", fn.num_combine(col(field_name), col(field_name), mode="sum")
        )
        if stage == FunctionChainStage.L2_RERANK:
            chain.sort(col("$score"), desc=True, tie_break_col=col("$id"))

        for return_input in [False, True]:
            output_fields = ["id", "payload"]
            if return_input:
                output_fields.append(field_name)
            res, _ = self.search(
                client,
                collection_name,
                data=[[1.0, 0.0], [0.0, 1.0]],
                anns_field=self.vector_field,
                search_params={"metric_type": "COSINE"},
                limit=3,
                output_fields=output_fields,
                function_chains=chain,
            )

            assert len(res) == 2
            for hits in res:
                assert [hit["id"] for hit in hits] == [2, 3, 1]
                assert [hit["distance"] for hit in hits] == pytest.approx([60.0, 40.0, 20.0])
                assert [self._hit_field(hit, "payload") for hit in hits] == [202, 303, 101]
                if return_input:
                    assert [self._hit_field(hit, field_name) for hit in hits] == [30.0, 20.0, 10.0]
                else:
                    assert all(self._hit_field(hit, field_name) is None for hit in hits)

    @pytest.mark.tags(CaseLabel.L0)
    def test_search_with_l0_function_chain_sdk_reranks_by_scalar_field(self):
        """
        target: test pymilvus FunctionChain SDK with L0 rerank
        method: map $score = num_combine($score, ts) at L0 stage
        expected: search succeeds and result order follows rewritten score
        """
        client = self._client()
        collection_name = self._create_function_chain_collection(client)

        res, _ = self.search(
            client,
            collection_name,
            data=[[0.0, 0.0]],
            anns_field=self.vector_field,
            search_params={"metric_type": "L2"},
            limit=3,
            output_fields=[self.scalar_field, self.vector_field],
            function_chains=self._score_plus_ts_chain(FunctionChainStage.L0_RERANK),
        )

        assert [hit["id"] for hit in res[0]] == [3, 2, 1]
        assert [self._hit_field(hit, self.scalar_field) for hit in res[0]] == [30, 20, 10]
        expected_scores = [self._expected_l0_score(hit) for hit in res[0]]
        assert expected_scores == sorted(expected_scores, reverse=True)
        assert [pytest.approx(abs(hit["distance"]), rel=1e-5) for hit in res[0]] == expected_scores

    @pytest.mark.tags(CaseLabel.L0)
    def test_search_with_l0_function_chain_sdk_uses_hidden_input_field(self):
        """
        target: test L0 FunctionChain SDK can use fields that are not returned
        method: rerank by ts while only requesting primary key output
        expected: search succeeds, result order follows ts, and ts is not returned
        """
        client = self._client()
        collection_name = self._create_function_chain_collection(client)

        res, _ = self.search(
            client,
            collection_name,
            data=[[0.0, 0.0]],
            anns_field=self.vector_field,
            search_params={"metric_type": "L2"},
            limit=3,
            output_fields=["id"],
            function_chains=self._score_plus_ts_chain(FunctionChainStage.L0_RERANK),
        )

        assert [hit["id"] for hit in res[0]] == [3, 2, 1]
        assert all(self._hit_field(hit, self.scalar_field) is None for hit in res[0])

    @pytest.mark.tags(CaseLabel.L0)
    def test_search_with_l0_function_chain_sdk_can_read_id_system_input(self):
        """
        target: test L0 FunctionChain SDK can read public system input $id
        method: map $score = num_combine($score, $id) at L0 stage
        expected: search succeeds and result order follows rewritten score
        """
        client = self._client()
        collection_name = self._create_function_chain_collection(client)
        chain = FunctionChain(FunctionChainStage.L0_RERANK, name="score_plus_id").map(
            "$score",
            fn.num_combine(col("$score"), col("$id"), mode="sum"),
        )

        res, _ = self.search(
            client,
            collection_name,
            data=[[0.0, 0.0]],
            anns_field=self.vector_field,
            search_params={"metric_type": "L2"},
            limit=3,
            function_chains=chain,
        )

        assert [hit["id"] for hit in res[0]] == [3, 2, 1]

    @pytest.mark.tags(CaseLabel.L0)
    def test_search_rejects_l0_function_chain_sort_op(self):
        """
        target: test L0 FunctionChain SDK rejects non-map operators
        method: use sort op at L0 stage
        expected: request fails because public L0 currently only supports map op
        """
        client = self._client()
        collection_name = self._create_function_chain_collection(client)
        chain = FunctionChain(FunctionChainStage.L0_RERANK, name="bad_l0_sort").sort(
            col("$score"),
            desc=True,
            tie_break_col=col("$id"),
        )

        self._assert_search_error(
            client, collection_name, chain, 'type "sort" is not supported by L0 rerank function chain'
        )

    @pytest.mark.tags(CaseLabel.L0)
    def test_search_rejects_l0_function_chain_write_readonly_system_column(self):
        """
        target: test L0 FunctionChain SDK rejects writes to read-only system columns
        method: write map output to $id
        expected: request fails because only $score is writable in public L0 chains
        """
        client = self._client()
        collection_name = self._create_function_chain_collection(client)
        chain = FunctionChain(FunctionChainStage.L0_RERANK, name="bad_l0_write_id").map(
            "$id",
            fn.num_combine(col("$score"), col(self.scalar_field), mode="sum"),
        )

        self._assert_search_error(client, collection_name, chain, 'system output "$id" is not writable')

    @pytest.mark.tags(CaseLabel.L0)
    def test_search_rejects_l0_function_chain_read_internal_system_input(self):
        """
        target: test L0 FunctionChain SDK rejects internal system input columns
        method: read $seg_offset from a map expression
        expected: request fails because public L0 only exposes $id and $score as readable system inputs
        """
        client = self._client()
        collection_name = self._create_function_chain_collection(client)
        chain = FunctionChain(FunctionChainStage.L0_RERANK, name="bad_l0_seg_offset_input").map(
            "$score",
            fn.num_combine(col("$seg_offset"), col("$score"), mode="sum"),
        )

        self._assert_search_error(
            client,
            collection_name,
            chain,
            'unsupported function chain system input "$seg_offset"',
        )

    @pytest.mark.tags(CaseLabel.L0)
    def test_search_rejects_l0_function_chain_read_unknown_system_input(self):
        """
        target: test L0 FunctionChain SDK rejects unknown system input columns
        method: read $tmp_score from a map expression before it is produced
        expected: request fails because users cannot invent new $-prefixed system columns
        """
        client = self._client()
        collection_name = self._create_function_chain_collection(client)
        chain = FunctionChain(FunctionChainStage.L0_RERANK, name="bad_l0_unknown_system_input").map(
            "$score",
            fn.num_combine(col("$tmp_score"), col("$score"), mode="sum"),
        )

        self._assert_search_error(
            client,
            collection_name,
            chain,
            'unsupported function chain system input "$tmp_score"',
        )

    @pytest.mark.tags(CaseLabel.L0)
    def test_search_rejects_l0_function_chain_reserved_temp_output(self):
        """
        target: test L0 FunctionChain SDK rejects user temporary columns in system namespace
        method: write a map output named $tmp_score
        expected: request fails because $ prefix is reserved for system columns
        """
        client = self._client()
        collection_name = self._create_function_chain_collection(client)
        chain = FunctionChain(FunctionChainStage.L0_RERANK, name="bad_l0_reserved_temp_output").map(
            "$tmp_score",
            fn.num_combine(col("$score"), col(self.scalar_field), mode="sum"),
        )

        self._assert_search_error(client, collection_name, chain, 'system output "$tmp_score" is not writable')

    @pytest.mark.tags(CaseLabel.L0)
    def test_search_rejects_l0_function_chain_with_function_score(self):
        """
        target: test search rejects ambiguous L0 rerank APIs
        method: send boost FunctionScore and L0 function chain together
        expected: request fails because function_score and function_chains are mutually exclusive
        """
        client = self._client()
        collection_name = self._create_function_chain_collection(client)
        function = Function(
            name="boost_ts",
            function_type=FunctionType.RERANK,
            input_field_names=[],
            output_field_names=[],
            params={"reranker": "boost", "weight": "1.5"},
        )
        function_score = FunctionScore(functions=[function])

        self._assert_search_error(
            client,
            collection_name,
            self._score_plus_ts_chain(FunctionChainStage.L0_RERANK),
            "function_chains and ranker cannot be used together",
            ranker=function_score,
        )

    @pytest.mark.tags(CaseLabel.L0)
    def test_search_rejects_l0_function_chain_with_order_by(self):
        """
        target: test search rejects order_by with L0 function rerank
        method: send order_by_fields and L0 function chain together
        expected: request fails because they define conflicting sort criteria
        """
        client = self._client()
        collection_name = self._create_function_chain_collection(client)

        self._assert_search_error(
            client,
            collection_name,
            self._score_plus_ts_chain(FunctionChainStage.L0_RERANK),
            "order_by and function rerank cannot be used together",
            order_by_fields=[{"field": self.scalar_field, "order": "asc"}],
        )

    @pytest.mark.tags(CaseLabel.L0)
    def test_search_with_l1_function_chain_multi_query_temp_columns_offset_and_output_alignment(self):
        """
        target: test complex L1 rerank independently processes query chunks across sealed segments
        method: build a temporary score, rewrite score with two hidden fields, sort, offset, and limit two queries
        expected: each query selects exact cross-segment rows and keeps payload aligned without exposing inputs
        """
        client = self._client()
        collection_name = self._create_l1_function_chain_collection(client)
        chain = (
            FunctionChain(FunctionChainStage.L1_RERANK, name="l1_complex")
            .map(
                "boosted_score",
                fn.num_combine(col("$score"), col(self.scalar_field), mode="sum"),
            )
            .map(
                "$score",
                fn.num_combine(col("boosted_score"), col("category"), mode="sum"),
            )
            .sort(col(self.scalar_field), desc=True, tie_break_col=col("$id"))
            .limit(2, offset=1)
        )
        query_vectors = [[1.0, 0.0], [0.0, 1.0]]

        res, _ = self.search(
            client,
            collection_name,
            data=query_vectors,
            anns_field=self.vector_field,
            search_params={"metric_type": "COSINE"},
            limit=6,
            output_fields=["payload"],
            function_chains=chain,
        )

        expected_ids = [[4, 6], [4, 6]]
        payload_by_id = {4: 404, 6: 606}
        ts_by_id = {4: 50, 6: 40}
        category_by_id = {4: 4, 6: 6}
        vector_by_id = {4: [0.0, 1.0], 6: [-1.0, 0.0]}
        assert len(res) == len(query_vectors)
        for query_vector, hits, ids in zip(query_vectors, res, expected_ids):
            assert [hit["id"] for hit in hits] == ids
            expected_scores = []
            for entity_id in ids:
                cosine = sum(left * right for left, right in zip(query_vector, vector_by_id[entity_id]))
                expected_scores.append(ts_by_id[entity_id] + category_by_id[entity_id] + cosine)
            assert expected_scores == sorted(expected_scores, reverse=True)

            for hit, expected_score in zip(hits, expected_scores):
                assert self._hit_field(hit, "payload") == payload_by_id[hit["id"]]
                assert abs(hit["distance"]) == pytest.approx(expected_score, rel=1e-5, abs=1e-5)
                assert self._hit_field(hit, self.scalar_field) is None
                assert self._hit_field(hit, "category") is None
                assert self._hit_field(hit, "boosted_score") is None
                assert self._hit_field(hit, "$l1_source_index") is None

    @pytest.mark.tags(CaseLabel.L0)
    def test_search_with_l0_l1_l2_function_chains_execute_in_stage_order(self):
        """
        target: test L0, L1, and L2 function chains compose in their distributed stage order
        method: add ts at L0, category at L1 with sort/limit, then add payload and sort at L2
        expected: final IDs and scores equal exact L0 to L1 to L2 arithmetic
        """
        client = self._client()
        collection_name = self._create_l1_function_chain_collection(client)
        chains = [
            FunctionChain(FunctionChainStage.L0_RERANK, name="l0_add_ts").map(
                "$score",
                fn.num_combine(col("$score"), col(self.scalar_field), mode="sum"),
            ),
            (
                FunctionChain(FunctionChainStage.L1_RERANK, name="l1_add_category")
                .map(
                    "$score",
                    fn.num_combine(col("$score"), col("category"), mode="sum"),
                )
                .sort(col(self.scalar_field), desc=True, tie_break_col=col("$id"))
                .limit(3)
            ),
            (
                FunctionChain(FunctionChainStage.L2_RERANK, name="l2_add_payload")
                .map(
                    "$score",
                    fn.num_combine(col("$score"), col("payload"), mode="sum"),
                )
                .sort(col("$score"), desc=True, tie_break_col=col("$id"))
            ),
        ]

        res, _ = self.search(
            client,
            collection_name,
            data=[[1.0, 0.0]],
            anns_field=self.vector_field,
            search_params={"metric_type": "COSINE"},
            limit=6,
            output_fields=[self.scalar_field, "category", "payload"],
            function_chains=chains,
        )

        assert [hit["id"] for hit in res[0]] == [6, 4, 2]
        expected_by_id = {
            2: 202 + 60 + 2 + 0.8,
            4: 404 + 50 + 4 + 0.0,
            6: 606 + 40 + 6 - 1.0,
        }
        for hit in res[0]:
            entity_id = hit["id"]
            assert hit["distance"] == pytest.approx(
                expected_by_id[entity_id],
                rel=1e-5,
                abs=1e-5,
            )
            assert self._hit_field(hit, "payload") == entity_id * 101
            assert self._hit_field(hit, "category") == entity_id

    @pytest.mark.tags(CaseLabel.L0)
    @pytest.mark.parametrize(
        "case_name, chain_factory, expected_error, search_kwargs",
        [
            (
                "write_id",
                lambda: FunctionChain(FunctionChainStage.L1_RERANK, name="bad_l1_write_id").map(
                    "$id",
                    fn.num_combine(col("$score"), col("ts"), mode="sum"),
                ),
                'system output "$id" is not writable by L1',
                {},
            ),
            (
                "read_internal_system_input",
                lambda: FunctionChain(FunctionChainStage.L1_RERANK, name="bad_l1_seg_offset").map(
                    "$score",
                    fn.num_combine(col("$seg_offset"), col("$score"), mode="sum"),
                ),
                'unsupported function chain system input "$seg_offset"',
                {},
            ),
            (
                "write_reserved_provenance_column",
                lambda: FunctionChain(FunctionChainStage.L1_RERANK, name="bad_l1_provenance").map(
                    "$l1_source_index",
                    fn.num_combine(col("$score"), col("ts"), mode="sum"),
                ),
                'system output "$l1_source_index" is not writable by L1',
                {},
            ),
            (
                "order_by_conflict",
                lambda: FunctionChain(FunctionChainStage.L1_RERANK, name="bad_l1_order_by").map(
                    "$score",
                    fn.num_combine(col("$score"), col("ts"), mode="sum"),
                ),
                "order_by and function rerank cannot be used together",
                {"order_by_fields": [{"field": "ts", "order": "asc"}]},
            ),
        ],
        ids=[
            "write-id",
            "read-internal-system-input",
            "write-reserved-provenance-column",
            "order-by-conflict",
        ],
    )
    def test_search_rejects_invalid_l1_function_chain(
        self,
        case_name,
        chain_factory,
        expected_error,
        search_kwargs,
    ):
        """
        target: test L1 server validation rejects invalid outputs, inputs, functions, and search modes
        method: send malformed or incompatible L1 chains through ordinary PyMilvus search
        expected: every request fails with parameter error before any fallback execution
        """
        client = self._client()
        collection_name = self._create_function_chain_collection(client)

        self._assert_search_error(
            client,
            collection_name,
            chain_factory(),
            expected_error,
            **search_kwargs,
        )

    @pytest.mark.tags(CaseLabel.L0)
    def test_search_rejects_duplicate_l1_function_chain_stage(self):
        """
        target: test ordinary search rejects two L1 chains in one request
        method: send two independently valid L1 score maps
        expected: request fails because each function-chain stage may appear only once
        """
        client = self._client()
        collection_name = self._create_function_chain_collection(client)
        chains = [
            FunctionChain(FunctionChainStage.L1_RERANK, name="l1_first").map(
                "$score",
                fn.num_combine(col("$score"), col("ts"), mode="sum"),
            ),
            FunctionChain(FunctionChainStage.L1_RERANK, name="l1_second").map(
                "$score",
                fn.num_combine(col("$score"), col("$id"), mode="sum"),
            ),
        ]

        self._assert_search_error(
            client,
            collection_name,
            chains,
            "function chain stage FunctionChainStageL1Rerank appears more than once",
        )

    @pytest.mark.tags(CaseLabel.L0)
    @pytest.mark.parametrize(
        "stage",
        [
            FunctionChainStage.L0_RERANK,
            FunctionChainStage.L1_RERANK,
        ],
        ids=["l0", "l1"],
    )
    def test_search_with_querynode_function_chain_json_dynamic_and_scalar_inputs(self, stage):
        """
        target: test L0/L1 materialize JSON, explicit $meta, and scalar inputs across segments
        method: replace $score with values from three physical roots in a two-segment collection
        expected: both QueryNode stages produce the same global order without leaking hidden inputs
        """
        client = self._client()
        collection_name = self._create_json_dynamic_collection(client, multi_segment=True)
        chain = FunctionChain(stage, name=f"{stage.name.lower()}_json_dynamic").map(
            "$score",
            fn.num_combine(
                col('metadata["rank"]'),
                col('$meta["profile"]["bonus"]'),
                col("base_score"),
                mode="sum",
            ),
        )
        self._set_input_data_types(
            chain,
            0,
            DataType.INT64,
            DataType.DOUBLE,
            DataType.NONE,
        )

        res, _ = self.search(
            client,
            collection_name,
            data=[[0.0, 0.0], [0.04, 0.0]],
            anns_field=self.vector_field,
            search_params={"metric_type": "L2"},
            limit=6,
            output_fields=["id"],
            function_chains=chain,
        )

        expected_ids = [4, 1, 2, 3, 5, 6]
        assert [[hit["id"] for hit in hits] for hits in res] == [expected_ids, expected_ids]
        for hits in res:
            for hit in hits:
                assert self._hit_field(hit, "metadata") is None
                assert self._hit_field(hit, "profile") is None
                assert self._hit_field(hit, "base_score") is None
                assert self._hit_field(hit, 'metadata["rank"]') is None
                assert self._hit_field(hit, '$meta["profile"]["bonus"]') is None

    @pytest.mark.tags(CaseLabel.L0)
    @pytest.mark.parametrize(
        "stage",
        [
            FunctionChainStage.L0_RERANK,
            FunctionChainStage.L1_RERANK,
        ],
        ids=["l0", "l1"],
    )
    def test_search_with_querynode_json_dynamic_paths_preserves_requested_roots(self, stage):
        """
        target: test L0/L1 path inputs do not replace or leak into requested root fields
        method: rerank by JSON and $meta paths while returning the complete metadata and profile objects
        expected: roots retain their original objects and internal path/provenance columns stay hidden
        """
        client = self._client()
        collection_name = self._create_json_dynamic_collection(client, multi_segment=True)
        chain = FunctionChain(stage, name=f"{stage.name.lower()}_preserve_json_roots").map(
            "$score",
            fn.num_combine(
                col('metadata["rank"]'),
                col('$meta["profile"]["bonus"]'),
                col("base_score"),
                mode="sum",
            ),
        )
        self._set_input_data_types(
            chain,
            0,
            DataType.INT64,
            DataType.DOUBLE,
            DataType.NONE,
        )

        res, _ = self.search(
            client,
            collection_name,
            data=[[0.0, 0.0]],
            anns_field=self.vector_field,
            search_params={"metric_type": "L2"},
            limit=6,
            output_fields=["metadata", "profile"],
            function_chains=chain,
        )

        expected_ranks = {1: 100, 2: 0, 3: 60, 4: 50, 5: 25}
        expected_bonuses = {1: 0.0, 2: 90.0, 3: 20.0, 4: 60.0, 5: 10.0, 6: 70.0}
        assert [hit["id"] for hit in res[0]] == [4, 1, 2, 3, 5, 6]
        for hit in res[0]:
            entity_id = hit["id"]
            metadata = self._hit_field(hit, "metadata")
            if entity_id == 6:
                assert metadata is None
            else:
                assert metadata["rank"] == expected_ranks[entity_id]

            profile = self._hit_field(hit, "profile")
            assert profile["bonus"] == expected_bonuses[entity_id]
            assert self._hit_field(hit, "base_score") is None
            assert self._hit_field(hit, 'metadata["rank"]') is None
            assert self._hit_field(hit, '$meta["profile"]["bonus"]') is None
            assert self._hit_field(hit, "$l1_source_index") is None

    @pytest.mark.tags(CaseLabel.L0)
    @pytest.mark.parametrize(
        "stage",
        [
            FunctionChainStage.L0_RERANK,
            FunctionChainStage.L1_RERANK,
        ],
        ids=["l0", "l1"],
    )
    def test_search_with_querynode_function_chain_json_dynamic_zero_hit(self, stage):
        """
        target: test L0/L1 construct typed JSON and $meta columns for an empty candidate set
        method: execute a two-root score map with a filter that matches no rows
        expected: both QueryNode stages return one empty query result without materialization errors
        """
        client = self._client()
        collection_name = self._create_json_dynamic_collection(client, multi_segment=True)
        chain = FunctionChain(stage, name=f"{stage.name.lower()}_json_zero_hit").map(
            "$score",
            fn.num_combine(
                col('metadata["rank"]'),
                col('$meta["profile"]["bonus"]'),
                mode="sum",
            ),
        )
        self._set_input_data_types(chain, 0, DataType.INT64, DataType.DOUBLE)

        res, _ = self.search(
            client,
            collection_name,
            data=[[0.0, 0.0]],
            anns_field=self.vector_field,
            search_params={"metric_type": "L2"},
            limit=5,
            filter="id < 0",
            function_chains=chain,
        )

        assert len(res) == 1
        assert len(res[0]) == 0

    @pytest.mark.tags(CaseLabel.L0)
    @pytest.mark.parametrize(
        "stage",
        [
            FunctionChainStage.L0_RERANK,
            FunctionChainStage.L1_RERANK,
        ],
        ids=["l0", "l1"],
    )
    def test_search_with_querynode_function_chain_all_null_json_dynamic_paths(self, stage):
        """
        target: test L0/L1 preserve declared types when every projected path value is null
        method: map a missing JSON or $meta path into an internal temporary column
        expected: the numeric function accepts the typed all-null input and the temporary column is hidden
        """
        client = self._client()
        collection_name = self._create_json_dynamic_collection(client, multi_segment=True)
        cases = [
            ("json", 'metadata["not_exists"]', DataType.INT64),
            ("dynamic", '$meta["not_exists"]', DataType.DOUBLE),
            ("json_scalar_traversal", 'metadata["rank"][0]', DataType.INT64),
            ("dynamic_scalar_traversal", '$meta["profile"]["bonus"][0]', DataType.DOUBLE),
            ("json_array_out_of_range", 'metadata["items"][99]["rank"]', DataType.INT64),
        ]

        for source, path, data_type in cases:
            chain = FunctionChain(stage, name=f"{stage.name.lower()}_{source}_all_null").map(
                "all_null_projection",
                fn.num_combine(
                    col(path),
                    col("base_score"),
                    mode="sum",
                ),
            )
            self._set_input_data_types(chain, 0, data_type, DataType.NONE)

            res, _ = self.search(
                client,
                collection_name,
                data=[[0.0, 0.0]],
                anns_field=self.vector_field,
                search_params={"metric_type": "L2"},
                limit=6,
                output_fields=["id"],
                function_chains=chain,
            )

            assert [hit["id"] for hit in res[0]] == [1, 2, 3, 4, 5, 6]
            assert all(self._hit_field(hit, "all_null_projection") is None for hit in res[0])

    @pytest.mark.tags(CaseLabel.L0)
    @pytest.mark.parametrize(
        "stage",
        [
            FunctionChainStage.L0_RERANK,
            FunctionChainStage.L1_RERANK,
        ],
        ids=["l0", "l1"],
    )
    def test_search_with_querynode_function_chain_mixed_json_dynamic_path_values(self, stage):
        """
        target: test L0/L1 handle valid, mismatched, null, missing, and nullable-root path values
        method: read a mixed JSON or $meta path as INT64 and combine it with a scalar score component
        expected: only integer path values produce scores; every other representation becomes Arrow null
        """
        client = self._client()
        collection_name = self._create_json_dynamic_collection(client, multi_segment=True)
        cases = [
            ("json", 'metadata["mixed"]'),
            ("dynamic", '$meta["profile"]["mixed"]'),
        ]

        for source, path in cases:
            chain = FunctionChain(stage, name=f"{stage.name.lower()}_{source}_mixed_values").map(
                "$score",
                fn.num_combine(
                    col(path),
                    col("base_score"),
                    mode="sum",
                ),
            )
            self._set_input_data_types(chain, 0, DataType.INT64, DataType.NONE)

            res, _ = self.search(
                client,
                collection_name,
                data=[[0.0, 0.0]],
                anns_field=self.vector_field,
                search_params={"metric_type": "L2"},
                limit=6,
                output_fields=["id"],
                function_chains=chain,
            )

            assert [hit["id"] for hit in res[0]] == [4, 1, 2, 3, 5, 6]
            assert abs(res[0][0]["distance"]) == pytest.approx(24.0)
            assert abs(res[0][1]["distance"]) == pytest.approx(11.0)
            for hit in res[0]:
                assert self._hit_field(hit, path) is None
                assert self._hit_field(hit, "base_score") is None

    @pytest.mark.tags(CaseLabel.L0)
    @pytest.mark.parametrize(
        "stage",
        [
            FunctionChainStage.L0_RERANK,
            FunctionChainStage.L1_RERANK,
        ],
        ids=["l0", "l1"],
    )
    def test_search_with_querynode_function_chain_group_by_and_path_share_json_root(self, stage):
        """
        target: test L0/L1 keep GroupBy values while projecting another path from the same root
        method: group and rerank by regular JSON or $meta paths over two sealed segments
        expected: group representatives and stage-specific cross-segment rerank order remain correct
        """
        client = self._client()
        collection_name = self._create_json_dynamic_collection(client, multi_segment=True)
        # L0 runs after segment-local GroupBy. During the cross-segment merge,
        # ID 6's missing group path forms a valid null bucket and its rewritten
        # score places that bucket ahead of groups B and C.
        cases = [
            (
                "json",
                'metadata["group"]',
                'metadata["rank"]',
                DataType.INT64,
                [1, 3, 5],
            ),
            (
                "dynamic",
                '$meta["profile"]["group"]',
                '$meta["profile"]["bonus"]',
                DataType.DOUBLE,
                [6, 4, 5] if stage == FunctionChainStage.L0_RERANK else [3, 5, 1],
            ),
        ]

        for source, group_by_field, score_path, data_type, expected_ids in cases:
            chain = FunctionChain(stage, name=f"{stage.name.lower()}_{source}_group_by").map(
                "$score",
                fn.num_combine(col(score_path), col("base_score"), mode="sum"),
            )
            self._set_input_data_types(chain, 0, data_type, DataType.NONE)

            res, _ = self.search(
                client,
                collection_name,
                data=[[0.0, 0.0]],
                anns_field=self.vector_field,
                search_params={"metric_type": "L2"},
                limit=3,
                group_by_field=group_by_field,
                group_size=1,
                output_fields=["id"],
                function_chains=chain,
            )

            assert [hit["id"] for hit in res[0]] == expected_ids
            for hit in res[0]:
                assert self._hit_field(hit, group_by_field) is None
                assert self._hit_field(hit, score_path) is None
                assert self._hit_field(hit, "base_score") is None

    @pytest.mark.tags(CaseLabel.L0)
    @pytest.mark.parametrize(
        "path",
        [
            'metadata["enabled"]',
            '$meta["profile"]["enabled"]',
        ],
        ids=["json", "dynamic"],
    )
    def test_search_rejects_l1_function_chain_bool_path_sort(self, path):
        """
        target: test L1 preserves Bool path types instead of coercing them into sortable values
        method: sort a regular JSON or $meta Bool path over two sealed segments
        expected: projection succeeds, then Sort rejects the non-comparable Bool Arrow type
        """
        client = self._client()
        collection_name = self._create_json_dynamic_collection(client, multi_segment=True)
        chain = (
            FunctionChain(FunctionChainStage.L1_RERANK, name="l1_bool_json_path")
            .sort(path, desc=True, tie_break_col="$id")
            .limit(3)
        )
        self._set_input_data_types(chain, 0, DataType.BOOL, DataType.NONE)

        self._assert_search_error(
            client,
            collection_name,
            chain,
            "non-comparable type bool",
        )

    @pytest.mark.tags(CaseLabel.L0)
    @pytest.mark.parametrize(
        "stage",
        [
            FunctionChainStage.L0_RERANK,
            FunctionChainStage.L1_RERANK,
        ],
        ids=["l0", "l1"],
    )
    def test_search_rejects_invalid_querynode_json_dynamic_contracts(self, stage):
        """
        target: test L0/L1 reject malformed JSON and dynamic input/output contracts
        method: send invalid hints, roots, identifiers, repeated hints, and path outputs through MapOps
        expected: common planning rejects every request before QueryNode FunctionChain execution
        """
        client = self._client()
        collection_name = self._create_json_dynamic_collection(client)
        input_cases = [
            (
                "missing_hint",
                'metadata["rank"]',
                None,
                "JSON path input requires an explicit data_type",
            ),
            (
                "unsupported_float_hint",
                'metadata["rank"]',
                [DataType.FLOAT, DataType.NONE],
                "unsupported JSON path data type hint Float",
            ),
            (
                "json_hint",
                'metadata["rank"]',
                [DataType.JSON, DataType.NONE],
                "unsupported JSON path data type hint JSON",
            ),
            (
                "hint_count_mismatch",
                'metadata["rank"]',
                [DataType.INT64],
                "$input_data_types count 1 does not match input count 2",
            ),
            (
                "complete_json_root",
                "metadata",
                [DataType.INT64, DataType.NONE],
                "complete JSON root input is not supported",
            ),
            (
                "complete_dynamic_root",
                "$meta",
                [DataType.INT64, DataType.NONE],
                "complete JSON root input is not supported",
            ),
            (
                "bare_dynamic_path",
                'profile["bonus"]',
                [DataType.DOUBLE, DataType.NONE],
                "must use explicit $meta[...] syntax",
            ),
            (
                "nested_scalar",
                'id["nested"]',
                [DataType.INT64, DataType.NONE],
                "data type not supported accessed with []",
            ),
            (
                "negative_array_index",
                'metadata["items"][-1]["name"]',
                [DataType.VARCHAR, DataType.NONE],
                "cannot parse identifier",
            ),
            (
                "system_hint",
                "$score",
                [DataType.FLOAT, DataType.NONE],
                "system input does not accept data type hint Float",
            ),
        ]

        for case_name, path, input_data_types, err_msg in input_cases:
            chain = FunctionChain(stage, name=f"{stage.name.lower()}_{case_name}").map(
                "$score",
                fn.num_combine(col(path), col("base_score"), mode="sum"),
            )
            if input_data_types is not None:
                chain.ops[0].params["$input_data_types"] = input_data_types
            self._assert_search_error(client, collection_name, chain, err_msg)

        conflicting_chain = (
            FunctionChain(stage, name=f"{stage.name.lower()}_conflicting_json_types")
            .map(
                "temporary_score",
                fn.num_combine(col('metadata["rank"]'), col("$score"), mode="sum"),
            )
            .map(
                "$score",
                fn.num_combine(
                    col('metadata["rank"]'),
                    col("temporary_score"),
                    mode="sum",
                ),
            )
        )
        self._set_input_data_types(conflicting_chain, 0, DataType.INT64, DataType.NONE)
        self._set_input_data_types(conflicting_chain, 1, DataType.DOUBLE, DataType.NONE)
        self._assert_search_error(
            client,
            collection_name,
            conflicting_chain,
            "conflicting data type hints Int64 and Double for the same JSON path",
        )

        for source, output in [
            ("json", 'metadata["rerank_score"]'),
            ("dynamic", '$meta["rerank_score"]'),
        ]:
            output_chain = FunctionChain(
                stage,
                name=f"{stage.name.lower()}_{source}_path_output",
            ).map(
                output,
                fn.num_combine(col("$score"), col("base_score"), mode="sum"),
            )
            self._set_input_data_types(output_chain, 0, DataType.NONE, DataType.NONE)
            self._assert_search_error(
                client,
                collection_name,
                output_chain,
                "JSON root or path cannot be used as a function chain output",
            )

    @pytest.mark.tags(CaseLabel.L0)
    def test_search_with_l1_function_chain_nested_json_sort_limit_across_segments(self):
        """
        target: test L1 materializes a nested VARCHAR path and preserves source rows across segments
        method: select three rows by path order, then let L1 restore the score-order reduce contract
        expected: the selected rows survive and their final order follows the original ANN score
        """
        client = self._client()
        collection_name = self._create_json_dynamic_collection(client, multi_segment=True)
        chain = (
            FunctionChain(FunctionChainStage.L1_RERANK, name="l1_nested_json_sort_limit")
            .sort(
                'metadata["items"][0]["name"]',
                desc=False,
                tie_break_col="$id",
            )
            .limit(3)
        )
        self._set_input_data_types(chain, 0, DataType.VARCHAR, DataType.NONE)

        res, _ = self.search(
            client,
            collection_name,
            data=[[0.0, 0.0]],
            anns_field=self.vector_field,
            search_params={"metric_type": "L2"},
            limit=6,
            output_fields=["id"],
            function_chains=chain,
        )

        assert [hit["id"] for hit in res[0]] == [2, 3, 4]

    @pytest.mark.tags(CaseLabel.L0)
    def test_search_with_json_dynamic_function_chains_execute_l0_l1_l2_in_order(self):
        """
        target: test JSON and dynamic inputs compose across all three rerank stages
        method: submit stages in reverse order; set score at L0, add $meta at L1, multiply JSON at L2
        expected: exact IDs and scores follow L0 -> L1 -> L2 over two segments, regardless of request order
        """
        client = self._client()
        collection_name = self._create_json_dynamic_collection(
            client,
            multi_segment=True,
            metric_type="IP",
        )

        l0_chain = FunctionChain(FunctionChainStage.L0_RERANK, name="l0_json_rank").map(
            "$score",
            fn.num_combine(
                col('metadata["rank"]'),
                col("base_score"),
                mode="sum",
            ),
        )
        self._set_input_data_types(l0_chain, 0, DataType.INT64, DataType.NONE)

        l1_chain = FunctionChain(FunctionChainStage.L1_RERANK, name="l1_dynamic_bonus").map(
            "$score",
            fn.num_combine(
                col("$score"),
                col('$meta["profile"]["bonus"]'),
                mode="sum",
            ),
        )
        self._set_input_data_types(l1_chain, 0, DataType.NONE, DataType.DOUBLE)

        l2_chain = (
            FunctionChain(FunctionChainStage.L2_RERANK, name="l2_json_ratio")
            .map(
                "$score",
                fn.num_combine(
                    col("$score"),
                    col('metadata["ratio"]'),
                    mode="multiply",
                ),
            )
            .sort("$score", desc=True, tie_break_col="$id")
        )
        self._set_input_data_types(l2_chain, 0, DataType.NONE, DataType.DOUBLE)

        res, _ = self.search(
            client,
            collection_name,
            data=[[0.0, 0.0], [0.04, 0.0]],
            anns_field=self.vector_field,
            search_params={"metric_type": "IP"},
            limit=6,
            filter="id < 6",
            output_fields=["id"],
            function_chains=[l2_chain, l1_chain, l0_chain],
        )

        # Use the five non-null JSON roots to check every score exactly:
        # (metadata.rank + base_score + profile.bonus) * metadata.ratio.
        # Swapping L1/L2 gives ID 2 a score of 270 instead of 8280 and
        # changes the final ID order to [4, 3, 5, 2, 1].
        expected_ids = [2, 4, 3, 5, 1]
        expected_scores = {
            1: 0.0,
            2: 8280.0,
            3: 1660.0,
            4: 6840.0,
            5: 400.0,
        }
        assert [[hit["id"] for hit in hits] for hits in res] == [expected_ids, expected_ids]
        for hits in res:
            for hit in hits:
                entity_id = hit["id"]
                assert hit["distance"] == pytest.approx(expected_scores[entity_id])
                assert self._hit_field(hit, "metadata") is None
                assert self._hit_field(hit, "profile") is None
                assert self._hit_field(hit, "base_score") is None
                assert self._hit_field(hit, 'metadata["rank"]') is None
                assert self._hit_field(hit, 'metadata["ratio"]') is None
                assert self._hit_field(hit, '$meta["profile"]["bonus"]') is None

    @pytest.mark.tags(CaseLabel.L0)
    def test_search_with_l2_function_chain_json_dynamic_and_scalar_inputs(self):
        """
        target: test L2 FunctionChain mixes JSON, explicit $meta, and ordinary schema inputs
        method: replace $score with JSON INT64, $meta DOUBLE, and schema INT64 values
        expected: every query is reranked by all three inputs and hidden inputs are not returned
        """
        client = self._client()
        collection_name = self._create_l2_json_dynamic_collection(client)
        chain = (
            FunctionChain(FunctionChainStage.L2_RERANK, name="l2_json_dynamic")
            .map(
                "$score",
                fn.num_combine(
                    col('metadata["rank"]'),
                    col('$meta["profile"]["bonus"]'),
                    col("base_score"),
                    mode="sum",
                ),
            )
            .sort(col("$score"), desc=True, tie_break_col=col("$id"))
        )
        self._set_input_data_types(
            chain,
            0,
            DataType.INT64,
            DataType.DOUBLE,
            DataType.NONE,
        )

        res, _ = self.search(
            client,
            collection_name,
            data=[[0.0, 0.0], [0.04, 0.0]],
            anns_field=self.vector_field,
            search_params={"metric_type": "L2"},
            limit=6,
            output_fields=["id"],
            function_chains=chain,
        )

        expected_ids = [4, 1, 2, 3, 5, 6]
        assert [[hit["id"] for hit in hits] for hits in res] == [expected_ids, expected_ids]
        for hits in res:
            for hit in hits:
                assert self._hit_field(hit, "metadata") is None
                assert self._hit_field(hit, "profile") is None
                assert self._hit_field(hit, "base_score") is None
                assert self._hit_field(hit, 'metadata["rank"]') is None
                assert self._hit_field(hit, '$meta["profile"]["bonus"]') is None

    @pytest.mark.tags(CaseLabel.L0)
    @pytest.mark.parametrize("rank_path", ['$meta["rank"]', '$meta["profile"]["rank"]'])
    def test_search_with_l2_function_chain_hidden_dynamic_input_with_dynamic_output(self, rank_path):
        """
        target: preserve hidden L2 dynamic inputs when another dynamic key is requested
        method: sort by rank or profile.rank while returning only the unrelated title key
        expected: dynamic output projection preserves ordering without leaking hidden inputs
        requires: common.requery.searchPolicy=OutputVector (default) for the non-requery regression
        """
        client = self._client()
        collection_name = cf.gen_unique_str(prefix)
        schema = self.create_schema(client, auto_id=False, enable_dynamic_field=True)[0]
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field(self.vector_field, DataType.FLOAT_VECTOR, dim=self.dim)
        rows = [
            {
                "id": entity_id,
                self.vector_field: [0.01 * (entity_id - 1), 0.0],
                "title": f"article-{entity_id}",
                "rank": rank,
                "profile": {"rank": rank},
            }
            for entity_id, rank in [(1, 10), (2, 30), (3, 20)]
        ]
        self._create_collection_with_schema_and_rows(client, collection_name, schema, rows)
        chain = FunctionChain(FunctionChainStage.L2_RERANK, name="l2_hidden_dynamic_rank").sort(
            rank_path, desc=True, tie_break_col="$id"
        )
        self._set_input_data_types(chain, 0, DataType.INT64, DataType.NONE)

        # With only id requested, no dynamic-key pruning is applied. Requesting title
        # must preserve the hidden rank input as well as the same final ordering.
        for output_fields in [["id"], ["title"]]:
            res, _ = self.search(
                client,
                collection_name,
                data=[[0.0, 0.0], [0.02, 0.0]],
                anns_field=self.vector_field,
                search_params={"metric_type": "L2"},
                limit=3,
                output_fields=output_fields,
                function_chains=chain,
            )

            # Both ANN orders and the all-null $id tie-break order differ from this.
            assert [[hit["id"] for hit in hits] for hits in res] == [[2, 3, 1], [2, 3, 1]], output_fields
            for hits in res:
                for hit in hits:
                    if "title" in output_fields:
                        assert self._hit_field(hit, "title") == f"article-{hit['id']}"
                    else:
                        assert self._hit_field(hit, "title") is None
                    for hidden_field in ["rank", "profile", "$meta", rank_path]:
                        assert hidden_field not in hit
                        assert hidden_field not in hit.get("entity", {})

    @pytest.mark.tags(CaseLabel.L0)
    def test_search_with_l2_function_chain_multiple_paths_from_same_json_root(self):
        """
        target: test L2 projects multiple typed paths from one JSON root
        method: replace $score with metadata["rank"] plus metadata["ratio"]
        expected: INT64 and DOUBLE paths from the same root both affect result order
        """
        client = self._client()
        collection_name = self._create_l2_json_dynamic_collection(client)
        chain = (
            FunctionChain(FunctionChainStage.L2_RERANK, name="l2_same_json_root")
            .map(
                "$score",
                fn.num_combine(
                    col('metadata["rank"]'),
                    col('metadata["ratio"]'),
                    mode="sum",
                ),
            )
            .sort("$score", desc=True, tie_break_col="$id")
        )
        self._set_input_data_types(chain, 0, DataType.INT64, DataType.DOUBLE)

        res, _ = self.search(
            client,
            collection_name,
            data=[[0.0, 0.0]],
            anns_field=self.vector_field,
            search_params={"metric_type": "L2"},
            limit=6,
            output_fields=["id"],
            function_chains=chain,
        )

        assert [hit["id"] for hit in res[0]] == [4, 1, 2, 3, 5, 6]

    @pytest.mark.tags(CaseLabel.L0)
    def test_search_with_l2_function_chain_repeated_json_path_occurrences(self):
        """
        target: test repeated JSON path occurrences keep one type hint per occurrence
        method: add metadata["rank"] to itself and rerank by the result
        expected: the duplicated path is planned once but both expression arguments are evaluated
        """
        client = self._client()
        collection_name = self._create_l2_json_dynamic_collection(client)
        chain = (
            FunctionChain(FunctionChainStage.L2_RERANK, name="l2_repeated_json_path")
            .map(
                "$score",
                fn.num_combine(
                    col('metadata["rank"]'),
                    col('metadata["rank"]'),
                    mode="sum",
                ),
            )
            .sort("$score", desc=True, tie_break_col="$id")
        )
        self._set_input_data_types(chain, 0, DataType.INT64, DataType.INT64)

        res, _ = self.search(
            client,
            collection_name,
            data=[[0.0, 0.0]],
            anns_field=self.vector_field,
            search_params={"metric_type": "L2"},
            limit=6,
            output_fields=["id"],
            function_chains=chain,
        )

        assert [hit["id"] for hit in res[0]] == [1, 3, 4, 5, 2, 6]

    @pytest.mark.tags(CaseLabel.L0)
    def test_search_with_l2_function_chain_json_sort_limit_preserves_root_output(self):
        """
        target: test JSON path columns survive multiple operators without replacing root output
        method: sort by metadata["rank"], limit to three rows, and request complete metadata
        expected: ordering and limit apply while the original JSON objects are returned unchanged
        """
        client = self._client()
        collection_name = self._create_l2_json_dynamic_collection(client)
        chain = (
            FunctionChain(FunctionChainStage.L2_RERANK, name="l2_json_sort_limit")
            .sort('metadata["rank"]', desc=True, tie_break_col="$id")
            .limit(3)
        )
        self._set_input_data_types(chain, 0, DataType.INT64, DataType.NONE)

        res, _ = self.search(
            client,
            collection_name,
            data=[[0.0, 0.0]],
            anns_field=self.vector_field,
            search_params={"metric_type": "L2"},
            limit=6,
            output_fields=["metadata"],
            function_chains=chain,
        )

        assert [hit["id"] for hit in res[0]] == [1, 3, 4]
        expected_ranks = {1: 100, 3: 60, 4: 50}
        for hit in res[0]:
            metadata = self._hit_field(hit, "metadata")
            assert metadata["rank"] == expected_ranks[hit["id"]]
            assert self._hit_field(hit, 'metadata["rank"]') is None

    @pytest.mark.tags(CaseLabel.L0)
    def test_search_with_l2_function_chain_invalid_json_traversal_becomes_null(self):
        """
        target: test invalid JSON traversal is represented as a typed null
        method: traverse below a scalar intermediate value
        expected: every projected value is null and $id determines the final order
        """
        client = self._client()
        collection_name = self._create_l2_json_dynamic_collection(client)
        chain = FunctionChain(
            FunctionChainStage.L2_RERANK,
            name="l2_wrong_json_intermediate",
        ).sort(
            'metadata["rank"][0]',
            desc=True,
            tie_break_col="$id",
        )
        self._set_input_data_types(chain, 0, DataType.INT64, DataType.NONE)

        res, _ = self.search(
            client,
            collection_name,
            data=[[0.0, 0.0]],
            anns_field=self.vector_field,
            search_params={"metric_type": "L2"},
            limit=6,
            output_fields=["id"],
            function_chains=chain,
        )

        assert [hit["id"] for hit in res[0]] == [1, 2, 3, 4, 5, 6]

    @pytest.mark.tags(CaseLabel.L0)
    @pytest.mark.parametrize(
        "path",
        [
            'metadata["items"][0]["name"]',
            '$meta["profile"]["label"]',
        ],
    )
    def test_search_with_l2_function_chain_nested_varchar_path(self, path):
        """
        target: test L2 traverses nested JSON array and $meta object paths as VARCHAR
        method: sort by a nested path from a JSON field or the dynamic field
        expected: string values sort ascending and a missing or out-of-range path sorts last
        """
        client = self._client()
        collection_name = self._create_l2_json_dynamic_collection(client)
        chain = FunctionChain(
            FunctionChainStage.L2_RERANK,
            name="l2_nested_json_array",
        ).sort(
            path,
            desc=False,
            tie_break_col="$id",
        )
        self._set_input_data_types(chain, 0, DataType.VARCHAR, DataType.NONE)

        res, _ = self.search(
            client,
            collection_name,
            data=[[0.0, 0.0]],
            anns_field=self.vector_field,
            search_params={"metric_type": "L2"},
            limit=6,
            output_fields=["id"],
            function_chains=chain,
        )

        assert [hit["id"] for hit in res[0]] == [2, 4, 3, 1, 5, 6]

    @pytest.mark.tags(CaseLabel.L0)
    @pytest.mark.parametrize(
        "path",
        [
            'metadata["mixed"]',
            '$meta["profile"]["mixed"]',
        ],
    )
    def test_search_with_l2_function_chain_json_type_mismatch_becomes_null(self, path):
        """
        target: test L2 JSON and $meta projections do not cast mismatched values
        method: read a mixed-type path as INT64 and sort descending
        expected: string, JSON null, and missing values become null and sort after valid integers
        """
        client = self._client()
        collection_name = self._create_l2_json_dynamic_collection(client)
        chain = FunctionChain(
            FunctionChainStage.L2_RERANK,
            name="l2_json_mismatch",
        ).sort(
            path,
            desc=True,
            tie_break_col="$id",
        )
        self._set_input_data_types(chain, 0, DataType.INT64, DataType.NONE)

        res, _ = self.search(
            client,
            collection_name,
            data=[[0.0, 0.0]],
            anns_field=self.vector_field,
            search_params={"metric_type": "L2"},
            limit=6,
            output_fields=["id"],
            function_chains=chain,
        )

        assert [hit["id"] for hit in res[0]] == [4, 1, 2, 3, 5, 6]

    @pytest.mark.tags(CaseLabel.L0)
    @pytest.mark.parametrize(
        "path,data_type",
        [
            ('metadata["not_exists"]', DataType.INT64),
            ('$meta["not_exists"]', DataType.DOUBLE),
            ('metadata["enabled"]', DataType.INT64),
            ('$meta["profile"]["group"]', DataType.INT64),
        ],
    )
    def test_search_with_l2_function_chain_all_null_path_preserves_declared_type(
        self,
        path,
        data_type,
    ):
        """
        target: test L2 materializes all-null JSON and $meta paths with their declared types
        method: sort by a path that is missing everywhere or whose values all mismatch the declared type
        expected: all values are null and result order is determined by the $id tie-break column
        """
        client = self._client()
        collection_name = self._create_l2_json_dynamic_collection(client)
        chain = FunctionChain(
            FunctionChainStage.L2_RERANK,
            name="l2_all_missing_json_path",
        ).sort(
            path,
            desc=True,
            tie_break_col="$id",
        )
        self._set_input_data_types(chain, 0, data_type, DataType.NONE)

        res, _ = self.search(
            client,
            collection_name,
            data=[[0.0, 0.0]],
            anns_field=self.vector_field,
            search_params={"metric_type": "L2"},
            limit=6,
            output_fields=["id"],
            function_chains=chain,
        )

        assert [hit["id"] for hit in res[0]] == [1, 2, 3, 4, 5, 6]

    @pytest.mark.tags(CaseLabel.L0)
    def test_search_with_l2_function_chain_json_path_zero_hit(self):
        """
        target: test L2 JSON path columns are materialized for an empty candidate set
        method: execute a JSON-path FunctionChain with a filter that matches no rows
        expected: search returns an empty result without failing path-column construction
        """
        client = self._client()
        collection_name = self._create_l2_json_dynamic_collection(client)
        chain = FunctionChain(
            FunctionChainStage.L2_RERANK,
            name="l2_json_zero_hit",
        ).sort(
            'metadata["rank"]',
            desc=True,
            tie_break_col="$id",
        )
        self._set_input_data_types(chain, 0, DataType.INT64, DataType.NONE)

        res, _ = self.search(
            client,
            collection_name,
            data=[[0.0, 0.0]],
            anns_field=self.vector_field,
            search_params={"metric_type": "L2"},
            limit=5,
            filter="id < 0",
            function_chains=chain,
        )

        assert len(res) == 1
        assert len(res[0]) == 0

    @pytest.mark.tags(CaseLabel.L0)
    @pytest.mark.parametrize(
        "group_by_field,sort_path,data_type,expected_ids",
        [
            (
                'metadata["group"]',
                'metadata["rank"]',
                DataType.INT64,
                [1, 3, 5],
            ),
            (
                '$meta["profile"]["group"]',
                '$meta["profile"]["bonus"]',
                DataType.DOUBLE,
                [3, 5, 1],
            ),
        ],
    )
    def test_search_with_l2_function_chain_json_and_dynamic_group_by_same_root(
        self,
        group_by_field,
        sort_path,
        data_type,
        expected_ids,
    ):
        """
        target: test JSON and $meta group-by values survive L2 projection from the same root
        method: group and sort by two paths that share one physical JSON root field
        expected: one result per group is returned in descending declared-path order
        """
        client = self._client()
        collection_name = self._create_l2_json_dynamic_collection(client)
        chain = FunctionChain(
            FunctionChainStage.L2_RERANK,
            name="l2_json_group_by_same_root",
        ).sort(
            sort_path,
            desc=True,
            tie_break_col="$id",
        )
        self._set_input_data_types(chain, 0, data_type, DataType.NONE)

        res, _ = self.search(
            client,
            collection_name,
            data=[[0.0, 0.0]],
            anns_field=self.vector_field,
            search_params={"metric_type": "L2"},
            limit=3,
            group_by_field=group_by_field,
            group_size=1,
            output_fields=["id"],
            function_chains=chain,
        )

        assert [hit["id"] for hit in res[0]] == expected_ids

    @pytest.mark.tags(CaseLabel.L0)
    @pytest.mark.parametrize(
        "path,input_data_types,err_msg",
        [
            (
                'metadata["rank"]',
                None,
                "JSON path input requires an explicit data_type",
            ),
            (
                'metadata["rank"]',
                [DataType.FLOAT, DataType.NONE],
                "unsupported JSON path data type hint Float",
            ),
            (
                'metadata["rank"]',
                [DataType.JSON, DataType.NONE],
                "unsupported JSON path data type hint JSON",
            ),
            (
                'metadata["rank"]',
                [DataType.INT64],
                "$input_data_types count 1 does not match input count 2",
            ),
            (
                'metadata["rank"]',
                [],
                "$input_data_types count 0 does not match input count 2",
            ),
            (
                'metadata["rank"]',
                [DataType.INT64, DataType.NONE, DataType.NONE],
                "$input_data_types count 3 does not match input count 2",
            ),
            (
                'metadata["rank"]',
                int(DataType.INT64),
                "$input_data_types must be an array",
            ),
            (
                'metadata["rank"]',
                ["INT64", int(DataType.NONE)],
                "$input_data_types[0] must be an int64 data type enum value",
            ),
            (
                'metadata["rank"]',
                [9999, int(DataType.NONE)],
                "$input_data_types[0] has unknown data type value 9999",
            ),
            (
                "metadata",
                [DataType.INT64, DataType.NONE],
                "complete JSON root input is not supported",
            ),
            (
                "$meta",
                [DataType.INT64, DataType.NONE],
                "complete JSON root input is not supported",
            ),
            (
                'profile["bonus"]',
                [DataType.DOUBLE, DataType.NONE],
                "must use explicit $meta[...] syntax",
            ),
            (
                'id["nested"]',
                [DataType.INT64, DataType.NONE],
                "data type not supported accessed with []",
            ),
            (
                'metadata["items"][-1]["name"]',
                [DataType.VARCHAR, DataType.NONE],
                "cannot parse identifier",
            ),
            (
                "$score",
                [DataType.FLOAT, DataType.NONE],
                "system input does not accept data type hint Float",
            ),
        ],
    )
    def test_search_rejects_l2_json_path_invalid_input_type_contract(
        self,
        path,
        input_data_types,
        err_msg,
    ):
        """
        target: test L2 validates the low-level JSON path input type contract
        method: send malformed hints, unsupported inputs, complete roots, or bare dynamic paths
        expected: each invalid request is rejected before FunctionChain execution
        """
        client = self._client()
        collection_name = self._create_l2_json_dynamic_collection(client)
        chain = FunctionChain(
            FunctionChainStage.L2_RERANK,
            name="invalid_l2_json_input",
        ).sort(
            path,
            desc=True,
            tie_break_col="$id",
        )
        if input_data_types is not None:
            chain.ops[0].params["$input_data_types"] = input_data_types

        self._assert_search_error(client, collection_name, chain, err_msg)

    @pytest.mark.tags(CaseLabel.L0)
    def test_search_rejects_l2_meta_path_when_dynamic_field_is_disabled(self):
        """
        target: test explicit $meta paths require Dynamic Field to be enabled
        method: use a $meta path with a collection that has no dynamic field
        expected: schema-aware input planning rejects the identifier
        """
        client = self._client()
        collection_name = self._create_function_chain_collection(client)
        chain = FunctionChain(
            FunctionChainStage.L2_RERANK,
            name="l2_dynamic_disabled",
        ).sort(
            '$meta["profile"]["bonus"]',
            desc=True,
            tie_break_col="$id",
        )
        self._set_input_data_types(chain, 0, DataType.DOUBLE, DataType.NONE)

        self._assert_search_error(
            client,
            collection_name,
            chain,
            "cannot parse identifier",
        )

    @pytest.mark.tags(CaseLabel.L0)
    @pytest.mark.parametrize(
        "output",
        [
            'metadata["rerank_score"]',
            '$meta["rerank_score"]',
        ],
    )
    def test_search_rejects_l2_json_path_output(self, output):
        """
        target: test L2 does not allow operators to write JSON or $meta path columns
        method: map a numeric expression to a nested JSON output
        expected: the request is rejected during schema-aware output validation
        """
        client = self._client()
        collection_name = self._create_l2_json_dynamic_collection(client)
        chain = FunctionChain(FunctionChainStage.L2_RERANK, name="invalid_json_output").map(
            output,
            fn.num_combine(col("$score"), col("$score"), mode="sum"),
        )
        self._set_input_data_types(chain, 0, DataType.NONE, DataType.NONE)

        self._assert_search_error(
            client,
            collection_name,
            chain,
            "JSON root or path cannot be used as a function chain output",
        )

    @pytest.mark.tags(CaseLabel.L0)
    def test_search_rejects_l2_json_path_conflicting_input_types(self):
        """
        target: test one logical JSON path cannot have conflicting declared types
        method: consume metadata["rank"] as INT64 and DOUBLE in two map operations
        expected: input planning rejects the conflicting hints
        """
        client = self._client()
        collection_name = self._create_l2_json_dynamic_collection(client)
        chain = (
            FunctionChain(FunctionChainStage.L2_RERANK, name="conflicting_json_types")
            .map(
                "temporary_score",
                fn.num_combine(col('metadata["rank"]'), col("$score"), mode="sum"),
            )
            .map(
                "$score",
                fn.num_combine(
                    col('metadata["rank"]'),
                    col("temporary_score"),
                    mode="sum",
                ),
            )
        )
        self._set_input_data_types(chain, 0, DataType.INT64, DataType.NONE)
        self._set_input_data_types(chain, 1, DataType.DOUBLE, DataType.NONE)

        self._assert_search_error(
            client,
            collection_name,
            chain,
            "conflicting data type hints Int64 and Double for the same JSON path",
        )

    @pytest.mark.tags(CaseLabel.L0)
    def test_search_with_l2_function_chain_sdk_reranks_by_scalar_field(self):
        """
        target: test pymilvus FunctionChain SDK with L2 rerank
        method: map $score = num_combine($score, ts), then sort by $score desc
        expected: search succeeds and result order follows rewritten score
        """
        client = self._client()
        collection_name = self._create_function_chain_collection(client)

        res, _ = self.search(
            client,
            collection_name,
            data=[[0.0, 0.0]],
            anns_field=self.vector_field,
            search_params={"metric_type": "L2"},
            limit=3,
            output_fields=[self.scalar_field],
            function_chains=self._score_plus_ts_chain(FunctionChainStage.L2_RERANK),
        )

        assert [hit["id"] for hit in res[0]] == [3, 2, 1]
        assert [self._hit_field(hit, self.scalar_field) for hit in res[0]] == [30, 20, 10]

    @pytest.mark.tags(CaseLabel.L0)
    def test_search_with_l2_function_chain_sdk_uses_hidden_input_field(self):
        """
        target: test L2 FunctionChain SDK can use fields that are not returned
        method: rerank by ts while only requesting primary key output
        expected: search succeeds, result order follows ts, and ts is not returned
        """
        client = self._client()
        collection_name = self._create_function_chain_collection(client)

        res, _ = self.search(
            client,
            collection_name,
            data=[[0.0, 0.0]],
            anns_field=self.vector_field,
            search_params={"metric_type": "L2"},
            limit=3,
            output_fields=["id"],
            function_chains=self._score_plus_ts_chain(FunctionChainStage.L2_RERANK),
        )

        assert [hit["id"] for hit in res[0]] == [3, 2, 1]
        assert all(self._hit_field(hit, self.scalar_field) is None for hit in res[0])

    @pytest.mark.tags(CaseLabel.L0)
    def test_search_with_l2_function_chain_sdk_temp_column_not_returned(self):
        """
        target: test L2 FunctionChain SDK can use ordinary temporary columns
        method: write tmp_score, write it back to $score, then sort by $score desc
        expected: search succeeds, rerank order is correct, and tmp_score is not returned
        """
        client = self._client()
        collection_name = self._create_function_chain_collection(client)
        chain = (
            FunctionChain(FunctionChainStage.L2_RERANK, name="l2_temp_score")
            .map("tmp_score", fn.num_combine(col("$score"), col(self.scalar_field), mode="sum"))
            .map("$score", fn.num_combine(col("tmp_score"), col("$score"), mode="sum"))
            .sort(col("$score"), desc=True, tie_break_col=col("$id"))
        )

        res, _ = self.search(
            client,
            collection_name,
            data=[[0.0, 0.0]],
            anns_field=self.vector_field,
            search_params={"metric_type": "L2"},
            limit=3,
            output_fields=[self.scalar_field],
            function_chains=chain,
        )

        assert [hit["id"] for hit in res[0]] == [3, 2, 1]
        assert [self._hit_field(hit, self.scalar_field) for hit in res[0]] == [30, 20, 10]
        assert all(self._hit_field(hit, "tmp_score") is None for hit in res[0])

    @pytest.mark.tags(CaseLabel.L0)
    def test_search_with_l2_function_chain_sdk_limit_op(self):
        """
        target: test L2 FunctionChain SDK supports limit operator
        method: request limit=3 and apply function chain limit op with limit=2
        expected: search succeeds and returns only function-chain-limited results
        """
        client = self._client()
        collection_name = self._create_function_chain_collection(client)
        chain = FunctionChain(FunctionChainStage.L2_RERANK, name="l2_limit").limit(2)

        res, _ = self.search(
            client,
            collection_name,
            data=[[0.0, 0.0]],
            anns_field=self.vector_field,
            search_params={"metric_type": "L2"},
            limit=3,
            function_chains=chain,
        )

        assert len(res[0]) == 2

    @pytest.mark.tags(CaseLabel.L0)
    def test_search_rejects_l2_function_chain_write_readonly_system_column(self):
        """
        target: test L2 FunctionChain SDK rejects writes to read-only system columns
        method: write map output to $id
        expected: request fails because only $score is writable in L2 rerank chains
        """
        client = self._client()
        collection_name = self._create_function_chain_collection(client)
        chain = FunctionChain(FunctionChainStage.L2_RERANK, name="bad_l2_write_id").map(
            "$id",
            fn.num_combine(col("$score"), col(self.scalar_field), mode="sum"),
        )

        self._assert_search_error(client, collection_name, chain, 'system output "$id" is not writable')

    @pytest.mark.tags(CaseLabel.L0)
    def test_search_rejects_l2_function_chain_reserved_temp_output(self):
        """
        target: test L2 FunctionChain SDK rejects user temporary columns in system namespace
        method: write a map output named $tmp_score
        expected: request fails because $ prefix is reserved for system columns
        """
        client = self._client()
        collection_name = self._create_function_chain_collection(client)
        chain = FunctionChain(FunctionChainStage.L2_RERANK, name="bad_l2_reserved_temp_output").map(
            "$tmp_score",
            fn.num_combine(col("$score"), col(self.scalar_field), mode="sum"),
        )

        self._assert_search_error(client, collection_name, chain, 'system output "$tmp_score" is not writable')

    @pytest.mark.tags(CaseLabel.L0)
    def test_search_rejects_l2_function_chain_read_internal_system_input(self):
        """
        target: test L2 FunctionChain SDK rejects internal system input columns
        method: read $seg_offset from a map expression
        expected: request fails because L2 only exposes selected system inputs
        """
        client = self._client()
        collection_name = self._create_function_chain_collection(client)
        chain = FunctionChain(FunctionChainStage.L2_RERANK, name="bad_l2_seg_offset_input").map(
            "$score",
            fn.num_combine(col("$seg_offset"), col("$score"), mode="sum"),
        )

        self._assert_search_error(
            client,
            collection_name,
            chain,
            'unsupported function chain system input "$seg_offset"',
        )

    @pytest.mark.tags(CaseLabel.L0)
    def test_search_rejects_l2_function_chain_read_unknown_system_input(self):
        """
        target: test L2 FunctionChain SDK rejects unknown system input columns
        method: read $tmp_score from a map expression before it is produced
        expected: request fails because users cannot invent new $-prefixed system columns
        """
        client = self._client()
        collection_name = self._create_function_chain_collection(client)
        chain = FunctionChain(FunctionChainStage.L2_RERANK, name="bad_l2_unknown_system_input").map(
            "$score",
            fn.num_combine(col("$tmp_score"), col("$score"), mode="sum"),
        )

        self._assert_search_error(
            client,
            collection_name,
            chain,
            'unsupported function chain system input "$tmp_score"',
        )

    @pytest.mark.tags(CaseLabel.L0)
    def test_search_rejects_l2_function_chain_with_function_score(self):
        """
        target: test search rejects ambiguous L2 rerank APIs
        method: send boost FunctionScore and L2 function chain together
        expected: request fails because function chains and ranker are mutually exclusive
        """
        client = self._client()
        collection_name = self._create_function_chain_collection(client)
        function = Function(
            name="boost_ts",
            function_type=FunctionType.RERANK,
            input_field_names=[],
            output_field_names=[],
            params={"reranker": "boost", "weight": "1.5"},
        )
        function_score = FunctionScore(functions=[function])

        self._assert_search_error(
            client,
            collection_name,
            self._score_plus_ts_chain(FunctionChainStage.L2_RERANK),
            "function_chains and ranker cannot be used together",
            ranker=function_score,
        )

    @pytest.mark.tags(CaseLabel.L0)
    def test_search_rejects_l2_function_chain_with_order_by(self):
        """
        target: test search rejects order_by with L2 function rerank
        method: send order_by_fields and L2 function chain together
        expected: request fails because they define conflicting sort criteria
        """
        client = self._client()
        collection_name = self._create_function_chain_collection(client)

        self._assert_search_error(
            client,
            collection_name,
            self._score_plus_ts_chain(FunctionChainStage.L2_RERANK),
            "order_by and function rerank cannot be used together",
            order_by_fields=[{"field": self.scalar_field, "order": "asc"}],
        )
