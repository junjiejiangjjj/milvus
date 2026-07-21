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

#include <Python.h>

#include <atomic>
#include <barrier>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "pb/cgo_msg.pb.h"
#include "pyudf/pyudf_c.h"
#include "pyudf/pyudf_runtime.h"

namespace {

enum class WrapperMode {
    kWorking,
    kImportFailure,
    kMissingApiVersion,
    kIncompatibleApiVersion,
    kNonCallableLoader,
    kNonCallableCloser,
};

enum class LoadBehavior {
    kWorking,
    kThrows,
    kInvalidSequence,
};

enum class CloseBehavior {
    kWorking,
    kThrows,
};

WrapperMode wrapper_mode = WrapperMode::kWorking;
LoadBehavior load_behavior = LoadBehavior::kWorking;
CloseBehavior close_behavior = CloseBehavior::kWorking;
std::atomic<int> wrapper_initializations = 0;
std::atomic<int> load_calls = 0;
std::atomic<int> close_calls = 0;
std::atomic<int> temporary_file_sequence = 0;

bool
Fail(std::string_view message) {
    std::cerr << "FAILED: " << message << std::endl;
    return false;
}

#define CHECK(expression)                                    \
    do {                                                     \
        if (!(expression)) {                                 \
            return Fail("line " + std::to_string(__LINE__) + \
                        ": " #expression);                   \
        }                                                    \
    } while (false)

void
FreeStatus(CStatus* status) {
    if (status->error_code != 0 && status->error_msg != nullptr) {
        free(const_cast<char*>(status->error_msg));
    }
}

std::string
StatusMessage(const CStatus& status) {
    return status.error_msg == nullptr ? "" : status.error_msg;
}

bool
ExpectSuccess(CStatus status) {
    const auto message = StatusMessage(status);
    const bool succeeded = status.error_code == 0;
    FreeStatus(&status);
    if (!succeeded) {
        return Fail("expected success, got: " + message);
    }
    return true;
}

bool
ExpectFailure(CStatus status, std::string_view expected_message) {
    const auto message = StatusMessage(status);
    const bool failed = status.error_code != 0;
    FreeStatus(&status);
    if (!failed) {
        return Fail("expected failure");
    }
    if (message.find(expected_message) == std::string::npos) {
        return Fail("failure did not contain '" +
                    std::string(expected_message) + "': " + message);
    }
    return true;
}

PyObject*
LoadInstances(PyObject*, PyObject* arguments) {
    ++load_calls;
    if (load_behavior == LoadBehavior::kThrows) {
        PyErr_SetString(PyExc_ValueError, "injected factory/load failure");
        return nullptr;
    }
    if (load_behavior == LoadBehavior::kInvalidSequence) {
        return PyLong_FromLong(1);
    }

    PyObject* resource_name = nullptr;
    PyObject* local_path = nullptr;
    PyObject* stage = nullptr;
    PyObject* instance_count = nullptr;
    PyObject* resource_identity = nullptr;
    if (!PyArg_UnpackTuple(arguments,
                           "load_instances",
                           5,
                           5,
                           &resource_name,
                           &local_path,
                           &stage,
                           &instance_count,
                           &resource_identity)) {
        return nullptr;
    }
    static_cast<void>(resource_name);
    static_cast<void>(local_path);
    static_cast<void>(stage);
    static_cast<void>(resource_identity);
    const auto count = PyLong_AsLong(instance_count);
    if (count == -1 && PyErr_Occurred()) {
        return nullptr;
    }

    auto* instances = PyTuple_New(count);
    if (instances == nullptr) {
        return nullptr;
    }
    for (long index = 0; index < count; ++index) {
        auto* instance = PyLong_FromLong(index);
        if (instance == nullptr) {
            Py_DECREF(instances);
            return nullptr;
        }
        PyTuple_SET_ITEM(instances, index, instance);
    }
    return instances;
}

PyObject*
CloseInstances(PyObject*, PyObject* arguments) {
    PyObject* instances = nullptr;
    if (!PyArg_UnpackTuple(arguments, "close_instances", 1, 1, &instances)) {
        return nullptr;
    }
    ++close_calls;
    if (close_behavior == CloseBehavior::kThrows) {
        PyErr_SetString(PyExc_RuntimeError, "injected close failure");
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyMethodDef wrapper_methods[] = {
    {"load_instances", LoadInstances, METH_VARARGS, nullptr},
    {"close_instances", CloseInstances, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef wrapper_module = {
    PyModuleDef_HEAD_INIT,
    "milvus_pyudf_runtime",
    nullptr,
    -1,
    wrapper_methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

bool
SetModuleAttribute(PyObject* module, const char* name, PyObject* value) {
    if (value == nullptr) {
        return false;
    }
    const auto result = PyObject_SetAttrString(module, name, value);
    Py_DECREF(value);
    return result == 0;
}

PyMODINIT_FUNC
PyInit_milvus_pyudf_runtime(void) {
    ++wrapper_initializations;
    if (wrapper_mode == WrapperMode::kImportFailure) {
        PyErr_SetString(PyExc_ImportError,
                        "injected trusted wrapper import failure");
        return nullptr;
    }

    auto* module = PyModule_Create(&wrapper_module);
    if (module == nullptr) {
        return nullptr;
    }

    if (wrapper_mode != WrapperMode::kMissingApiVersion) {
        const auto version =
            wrapper_mode == WrapperMode::kIncompatibleApiVersion ? 2 : 1;
        if (PyModule_AddIntConstant(module, "RUNTIME_API_VERSION", version) !=
            0) {
            Py_DECREF(module);
            return nullptr;
        }
    }
    if (wrapper_mode == WrapperMode::kNonCallableLoader &&
        !SetModuleAttribute(module, "load_instances", PyLong_FromLong(1))) {
        Py_DECREF(module);
        return nullptr;
    }
    if (wrapper_mode == WrapperMode::kNonCallableCloser &&
        !SetModuleAttribute(module, "close_instances", PyLong_FromLong(1))) {
        Py_DECREF(module);
        return nullptr;
    }
    return module;
}

bool
RegisterWrapper(WrapperMode mode) {
    wrapper_mode = mode;
    if (PyImport_AppendInittab("milvus_pyudf_runtime",
                               &PyInit_milvus_pyudf_runtime) != 0) {
        return Fail("could not register built-in trusted wrapper");
    }
    return true;
}

void
ResetBehaviors() {
    load_behavior = LoadBehavior::kWorking;
    close_behavior = CloseBehavior::kWorking;
    wrapper_initializations = 0;
    load_calls = 0;
    close_calls = 0;
}

class TemporaryWheel {
 public:
    TemporaryWheel() {
        const auto file_id = temporary_file_sequence.fetch_add(1);
        path_ =
            std::filesystem::temp_directory_path() /
            ("milvus_pyudf_runtime_test_" +
             std::to_string(
                 std::chrono::steady_clock::now().time_since_epoch().count()) +
             "_" + std::to_string(file_id) + ".whl");
        std::ofstream wheel(path_, std::ios::binary);
        wheel << "test wheel placeholder";
        if (!wheel.good()) {
            throw std::runtime_error("could not create temporary wheel");
        }
    }

    ~TemporaryWheel() {
        std::error_code error;
        std::filesystem::remove(path_, error);
    }

    std::string
    path() const {
        return path_.string();
    }

 private:
    std::filesystem::path path_;
};

std::string
SerializeRequest(const std::string& local_path,
                 const std::string& resource_name = "rank_udf",
                 const std::string& resource_path = "/remote/rank_udf.whl",
                 const std::string& stage = "L2_rerank",
                 int32_t instance_count = 1) {
    milvus::proto::cgo::PyUDFLoadRequest request;
    request.set_resource_name(resource_name);
    request.set_resource_id(7);
    request.set_resource_path(resource_path);
    request.set_local_path(local_path);
    request.set_stage(stage);
    request.set_instance_count(instance_count);

    std::string serialized;
    if (!request.SerializeToString(&serialized)) {
        throw std::runtime_error("could not serialize PyUDF load request");
    }
    return serialized;
}

bool
InitializeWorkingRuntime() {
    ResetBehaviors();
    CHECK(RegisterWrapper(WrapperMode::kWorking));
    return ExpectSuccess(InitializePyUDFRuntime());
}

bool
TestLoadBeforeInitialization() {
    CPyUDFResource resource = reinterpret_cast<CPyUDFResource>(0x1);
    CHECK(ExpectFailure(LoadPyUDFResource(nullptr, 0, &resource),
                        "runtime has not been initialized"));
    CHECK(resource == nullptr);
    return true;
}

bool
TestPreinitializedCPythonIsRejected() {
    Py_Initialize();
    CHECK(Py_IsInitialized());
    CHECK(ExpectFailure(
        InitializePyUDFRuntime(),
        "CPython was initialized before the isolated PyUDF runtime"));
    CHECK(ExpectFailure(
        InitializePyUDFRuntime(),
        "CPython was initialized before the isolated PyUDF runtime"));
    return true;
}

bool
TestConcurrentIdempotentInitialization() {
    ResetBehaviors();
    CHECK(RegisterWrapper(WrapperMode::kWorking));

    constexpr size_t kThreadCount = 8;
    std::barrier start(kThreadCount + 1);
    std::vector<CStatus> statuses(kThreadCount);
    std::vector<std::thread> threads;
    threads.reserve(kThreadCount);
    for (size_t index = 0; index < kThreadCount; ++index) {
        threads.emplace_back([&start, &statuses, index]() {
            start.arrive_and_wait();
            statuses[index] = InitializePyUDFRuntime();
        });
    }
    start.arrive_and_wait();
    for (auto& thread : threads) {
        thread.join();
    }

    bool succeeded = true;
    for (auto& status : statuses) {
        succeeded = ExpectSuccess(status) && succeeded;
    }
    CHECK(succeeded);
    CHECK(wrapper_initializations == 1);
    CHECK(ExpectSuccess(InitializePyUDFRuntime()));
    CHECK(ExpectSuccess(InitializePyUDFRuntime()));
    CHECK(wrapper_initializations == 1);
    return true;
}

bool
TestWrapperImportFailure() {
    ResetBehaviors();
    CHECK(RegisterWrapper(WrapperMode::kImportFailure));
    CHECK(ExpectFailure(InitializePyUDFRuntime(),
                        "cannot import trusted runtime wrapper"));
    CHECK(ExpectFailure(InitializePyUDFRuntime(),
                        "injected trusted wrapper import failure"));
    return true;
}

bool
TestWrapperMissingApiVersion() {
    ResetBehaviors();
    CHECK(RegisterWrapper(WrapperMode::kMissingApiVersion));
    return ExpectFailure(InitializePyUDFRuntime(),
                         "trusted runtime wrapper has no API version");
}

bool
TestWrapperIncompatibleApiVersion() {
    ResetBehaviors();
    CHECK(RegisterWrapper(WrapperMode::kIncompatibleApiVersion));
    return ExpectFailure(
        InitializePyUDFRuntime(),
        "trusted runtime wrapper has incompatible API version");
}

bool
TestWrapperNonCallableLoader() {
    ResetBehaviors();
    CHECK(RegisterWrapper(WrapperMode::kNonCallableLoader));
    return ExpectFailure(
        InitializePyUDFRuntime(),
        "trusted runtime wrapper has no callable load_instances");
}

bool
TestWrapperNonCallableCloser() {
    ResetBehaviors();
    CHECK(RegisterWrapper(WrapperMode::kNonCallableCloser));
    return ExpectFailure(
        InitializePyUDFRuntime(),
        "trusted runtime wrapper has no callable close_instances");
}

bool
TestRequestValidation() {
    CHECK(InitializeWorkingRuntime());
    TemporaryWheel wheel;

    const auto expect_invalid = [](const uint8_t* request,
                                   uint64_t request_size,
                                   std::string_view message) {
        CPyUDFResource resource = reinterpret_cast<CPyUDFResource>(0x1);
        const bool failed = ExpectFailure(
            LoadPyUDFResource(request, request_size, &resource), message);
        return failed && resource == nullptr;
    };

    CHECK(expect_invalid(nullptr, 0, "serialized load request is empty"));
    const uint8_t malformed[] = {0xff};
    CHECK(expect_invalid(
        malformed, sizeof(malformed), "serialized load request is malformed"));
    const uint8_t one_byte[] = {0};
    CHECK(
        expect_invalid(one_byte,
                       static_cast<uint64_t>(INT_MAX) + 1,
                       "serialized load request exceeds protobuf parse limit"));
    CHECK(expect_invalid(nullptr, 1, "serialized load request pointer is nil"));
    const auto no_fields = std::string("\x10\x00", 2);
    CHECK(expect_invalid(reinterpret_cast<const uint8_t*>(no_fields.data()),
                         no_fields.size(),
                         "serialized load request has no protocol fields"));

    const auto blank_name = SerializeRequest(wheel.path(), " ");
    CHECK(expect_invalid(reinterpret_cast<const uint8_t*>(blank_name.data()),
                         blank_name.size(),
                         "blank or invalid UTF-8 protocol fields"));
    const auto blank_resource_path =
        SerializeRequest(wheel.path(), "rank_udf", "\t");
    CHECK(expect_invalid(
        reinterpret_cast<const uint8_t*>(blank_resource_path.data()),
        blank_resource_path.size(),
        "blank or invalid UTF-8 protocol fields"));
    const auto blank_local_path =
        SerializeRequest(" ", "rank_udf", "/remote/rank_udf.whl");
    CHECK(expect_invalid(
        reinterpret_cast<const uint8_t*>(blank_local_path.data()),
        blank_local_path.size(),
        "blank or invalid UTF-8 protocol fields"));
    const auto blank_stage = SerializeRequest(
        wheel.path(), "rank_udf", "/remote/rank_udf.whl", "\n");
    CHECK(expect_invalid(reinterpret_cast<const uint8_t*>(blank_stage.data()),
                         blank_stage.size(),
                         "blank or invalid UTF-8 protocol fields"));
    const auto no_instances = SerializeRequest(
        wheel.path(), "rank_udf", "/remote/rank_udf.whl", "L2_rerank", 0);
    CHECK(expect_invalid(reinterpret_cast<const uint8_t*>(no_instances.data()),
                         no_instances.size(),
                         "instance_count must be positive"));
    const auto negative_instances = SerializeRequest(
        wheel.path(), "rank_udf", "/remote/rank_udf.whl", "L2_rerank", -1);
    CHECK(expect_invalid(
        reinterpret_cast<const uint8_t*>(negative_instances.data()),
        negative_instances.size(),
        "instance_count must be positive"));
    const auto wrong_extension = SerializeRequest(wheel.path() + ".zip");
    CHECK(
        expect_invalid(reinterpret_cast<const uint8_t*>(wrong_extension.data()),
                       wrong_extension.size(),
                       "local wheel is not a readable regular .whl file"));
    const auto missing_wheel = SerializeRequest(wheel.path() + ".missing.whl");
    CHECK(expect_invalid(reinterpret_cast<const uint8_t*>(missing_wheel.data()),
                         missing_wheel.size(),
                         "local wheel is not a readable regular .whl file"));

    auto invalid_utf8 = SerializeRequest(wheel.path());
    const auto resource_name_position = invalid_utf8.find("rank_udf");
    CHECK(resource_name_position != std::string::npos);
    invalid_utf8[resource_name_position] = static_cast<char>(0xff);
    CHECK(expect_invalid(reinterpret_cast<const uint8_t*>(invalid_utf8.data()),
                         invalid_utf8.size(),
                         "serialized load request is malformed"));
    return true;
}

bool
TestPythonLoadExceptionPropagates() {
    CHECK(InitializeWorkingRuntime());
    TemporaryWheel wheel;
    const auto request = SerializeRequest(wheel.path());
    load_behavior = LoadBehavior::kThrows;

    CPyUDFResource resource = reinterpret_cast<CPyUDFResource>(0x1);
    CHECK(ExpectFailure(
        LoadPyUDFResource(reinterpret_cast<const uint8_t*>(request.data()),
                          request.size(),
                          &resource),
        "Python UDF load failed"));
    CHECK(resource == nullptr);
    CHECK(load_calls == 1);
    return true;
}

bool
TestInvalidPythonLoadResultIsRejected() {
    CHECK(InitializeWorkingRuntime());
    TemporaryWheel wheel;
    const auto request = SerializeRequest(wheel.path());
    load_behavior = LoadBehavior::kInvalidSequence;

    CPyUDFResource resource = reinterpret_cast<CPyUDFResource>(0x1);
    CHECK(ExpectFailure(
        LoadPyUDFResource(reinterpret_cast<const uint8_t*>(request.data()),
                          request.size(),
                          &resource),
        "trusted wrapper returned an invalid instance sequence"));
    CHECK(resource == nullptr);
    return true;
}

bool
TestValidLoadAndClose() {
    CHECK(InitializeWorkingRuntime());
    TemporaryWheel wheel;
    const auto request = SerializeRequest(
        wheel.path(), "rank_udf", "/remote/rank_udf.whl", "L2_rerank", 2);

    CPyUDFResource resource = nullptr;
    CHECK(ExpectSuccess(
        LoadPyUDFResource(reinterpret_cast<const uint8_t*>(request.data()),
                          request.size(),
                          &resource)));
    CHECK(resource != nullptr);
    CHECK(load_calls == 1);
    CHECK(ExpectSuccess(DeletePyUDFResource(resource)));
    CHECK(close_calls == 1);
    return true;
}

std::unique_ptr<milvus::pyudf::PyUDFResource>
LoadCppResource(const std::string& serialized_request) {
    return milvus::pyudf::LoadResource(
        reinterpret_cast<const uint8_t*>(serialized_request.data()),
        serialized_request.size());
}

bool
TestCloseExceptionPropagatesAndReleases() {
    CHECK(InitializeWorkingRuntime());
    TemporaryWheel wheel;
    const auto request = SerializeRequest(wheel.path());
    close_behavior = CloseBehavior::kThrows;

    CPyUDFResource resource = nullptr;
    CHECK(ExpectSuccess(
        LoadPyUDFResource(reinterpret_cast<const uint8_t*>(request.data()),
                          request.size(),
                          &resource)));
    CHECK(resource != nullptr);
    CHECK(ExpectFailure(DeletePyUDFResource(resource),
                        "Python resource close failed"));
    CHECK(close_calls == 1);
    return true;
}

bool
TestConcurrentIdempotentClose() {
    CHECK(InitializeWorkingRuntime());
    TemporaryWheel wheel;
    auto resource = LoadCppResource(SerializeRequest(wheel.path()));
    CHECK(resource != nullptr);

    constexpr size_t kThreadCount = 8;
    std::barrier start(kThreadCount + 1);
    std::atomic<bool> all_closed = true;
    std::vector<std::thread> threads;
    threads.reserve(kThreadCount);
    for (size_t index = 0; index < kThreadCount; ++index) {
        threads.emplace_back(
            [&start, &all_closed, resource = resource.get()]() {
                start.arrive_and_wait();
                try {
                    resource->Close();
                } catch (...) {
                    all_closed = false;
                }
            });
    }
    start.arrive_and_wait();
    for (auto& thread : threads) {
        thread.join();
    }

    CHECK(all_closed);
    CHECK(close_calls == 1);
    resource->Close();
    CHECK(close_calls == 1);
    return true;
}

struct TestCase {
    const char* name;
    bool (*run)();
};

constexpr TestCase test_cases[] = {
    {"load-before-initialization", TestLoadBeforeInitialization},
    {"preinitialized-cpython-rejected", TestPreinitializedCPythonIsRejected},
    {"concurrent-idempotent-initialize",
     TestConcurrentIdempotentInitialization},
    {"trusted-wrapper-import-failure", TestWrapperImportFailure},
    {"trusted-wrapper-missing-api-version", TestWrapperMissingApiVersion},
    {"trusted-wrapper-incompatible-api-version",
     TestWrapperIncompatibleApiVersion},
    {"trusted-wrapper-non-callable-loader", TestWrapperNonCallableLoader},
    {"trusted-wrapper-non-callable-closer", TestWrapperNonCallableCloser},
    {"request-validation", TestRequestValidation},
    {"python-load-exception", TestPythonLoadExceptionPropagates},
    {"invalid-python-load-result", TestInvalidPythonLoadResultIsRejected},
    {"valid-load-and-close", TestValidLoadAndClose},
    {"close-exception", TestCloseExceptionPropagatesAndReleases},
    {"concurrent-idempotent-close", TestConcurrentIdempotentClose},
};

}  // namespace

int
main(int argc, char** argv) {
    if (argc != 3 || std::string_view(argv[1]) != "--case") {
        std::cerr << "usage: " << argv[0] << " --case <test-case>" << std::endl;
        return EXIT_FAILURE;
    }

    for (const auto& test_case : test_cases) {
        if (argv[2] == std::string_view(test_case.name)) {
            try {
                if (!test_case.run()) {
                    return EXIT_FAILURE;
                }
            } catch (const std::exception& exception) {
                std::cerr << "FAILED: unexpected exception: "
                          << exception.what() << std::endl;
                return EXIT_FAILURE;
            } catch (...) {
                std::cerr << "FAILED: unexpected non-standard exception"
                          << std::endl;
                return EXIT_FAILURE;
            }
            return EXIT_SUCCESS;
        }
    }

    std::cerr << "unknown test case: " << argv[2] << std::endl;
    return EXIT_FAILURE;
}
