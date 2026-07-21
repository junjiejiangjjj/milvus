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

#include "pyudf/pyudf_runtime.h"

#include <Python.h>

#include <google/protobuf/message_lite.h>

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/EasyAssert.h"
#include "pb/cgo_msg.pb.h"

namespace milvus::pyudf {
namespace {

class GilGuard {
 public:
    GilGuard() : state_(PyGILState_Ensure()) {
    }

    ~GilGuard() {
        PyGILState_Release(state_);
    }

    GilGuard(const GilGuard&) = delete;
    GilGuard&
    operator=(const GilGuard&) = delete;

 private:
    PyGILState_STATE state_;
};

std::string
PythonExceptionString() {
    if (!PyErr_Occurred()) {
        return "Python operation failed without an exception";
    }

    PyObject* exception_type = nullptr;
    PyObject* exception_value = nullptr;
    PyObject* traceback = nullptr;
    PyErr_Fetch(&exception_type, &exception_value, &traceback);
    PyErr_NormalizeException(&exception_type, &exception_value, &traceback);

    std::string message;
    PyObject* traceback_module = PyImport_ImportModule("traceback");
    if (traceback_module != nullptr) {
        PyObject* formatted = PyObject_CallMethod(
            traceback_module,
            "format_exception",
            "OOO",
            exception_type == nullptr ? Py_None : exception_type,
            exception_value == nullptr ? Py_None : exception_value,
            traceback == nullptr ? Py_None : traceback);
        if (formatted != nullptr) {
            PyObject* separator = PyUnicode_FromString("");
            if (separator != nullptr) {
                PyObject* joined = PyUnicode_Join(separator, formatted);
                if (joined != nullptr) {
                    const char* text = PyUnicode_AsUTF8(joined);
                    if (text != nullptr) {
                        message = text;
                    }
                    Py_DECREF(joined);
                }
                Py_DECREF(separator);
            }
            Py_DECREF(formatted);
        }
        Py_DECREF(traceback_module);
    }

    if (message.empty() && exception_value != nullptr) {
        PyObject* rendered = PyObject_Str(exception_value);
        if (rendered != nullptr) {
            const char* text = PyUnicode_AsUTF8(rendered);
            if (text != nullptr) {
                message = text;
            }
            Py_DECREF(rendered);
        }
    }
    if (message.empty()) {
        message = "Python operation failed";
    }

    Py_XDECREF(traceback);
    Py_XDECREF(exception_value);
    Py_XDECREF(exception_type);
    PyErr_Clear();
    return message;
}

[[noreturn]] void
ThrowPythonFailure(const char* operation) {
    auto python_error = PythonExceptionString();
    ThrowInfo(ErrorCode::UnexpectedError,
              "py_udf: {}: {}",
              operation,
              python_error);
}

bool
IsBlank(const std::string& value) {
    return value.empty() ||
           std::all_of(value.begin(), value.end(), [](unsigned char c) {
               return std::isspace(c) != 0;
           });
}

bool
IsValidUTF8(const std::string& value) {
    const auto* text = reinterpret_cast<const unsigned char*>(value.data());
    size_t position = 0;
    while (position < value.size()) {
        const unsigned char first = text[position++];
        if (first <= 0x7f) {
            continue;
        }

        size_t continuation_count = 0;
        uint32_t codepoint = 0;
        if ((first & 0xe0) == 0xc0) {
            continuation_count = 1;
            codepoint = first & 0x1f;
        } else if ((first & 0xf0) == 0xe0) {
            continuation_count = 2;
            codepoint = first & 0x0f;
        } else if ((first & 0xf8) == 0xf0) {
            continuation_count = 3;
            codepoint = first & 0x07;
        } else {
            return false;
        }
        if (position + continuation_count > value.size()) {
            return false;
        }
        for (size_t index = 0; index < continuation_count; ++index) {
            const unsigned char next = text[position++];
            if ((next & 0xc0) != 0x80) {
                return false;
            }
            codepoint = (codepoint << 6) | (next & 0x3f);
        }
        if ((continuation_count == 1 && codepoint < 0x80) ||
            (continuation_count == 2 && codepoint < 0x800) ||
            (continuation_count == 3 && codepoint < 0x10000) ||
            codepoint > 0x10ffff ||
            (codepoint >= 0xd800 && codepoint <= 0xdfff)) {
            return false;
        }
    }
    return true;
}

class OwnedPyObject {
 public:
    OwnedPyObject() noexcept = default;

    static OwnedPyObject
    Adopt(PyObject* new_reference) noexcept {
        return OwnedPyObject(new_reference);
    }

    static OwnedPyObject
    FromBorrowed(PyObject* borrowed_reference) noexcept {
        return OwnedPyObject(Py_XNewRef(borrowed_reference));
    }

    ~OwnedPyObject() {
        Py_XDECREF(object_);
    }

    OwnedPyObject(const OwnedPyObject&) = delete;
    OwnedPyObject&
    operator=(const OwnedPyObject&) = delete;

    OwnedPyObject(OwnedPyObject&& other) noexcept
        : object_(other.release()) {
    }

    OwnedPyObject&
    operator=(OwnedPyObject&& other) noexcept {
        if (this != &other) {
            reset(other.release());
        }
        return *this;
    }

    [[nodiscard]] PyObject*
    get() const noexcept {
        return object_;
    }

    explicit operator bool() const noexcept {
        return object_ != nullptr;
    }

    [[nodiscard]] PyObject*
    release() noexcept {
        return std::exchange(object_, nullptr);
    }

    void
    reset(PyObject* new_reference = nullptr) noexcept {
        auto* old_reference = std::exchange(object_, new_reference);
        Py_XDECREF(old_reference);
    }

 private:
    explicit OwnedPyObject(PyObject* object) noexcept : object_(object) {
    }

    PyObject* object_ = nullptr;
};

class RuntimeState {
 public:
    void
    Initialize() {
        std::call_once(initialization_once_, [this]() {
            try {
                InitializeOnce();
                std::lock_guard lock(mutex_);
                initialized_ = true;
            } catch (...) {
                std::lock_guard lock(mutex_);
                initialization_error_ = std::current_exception();
            }
        });

        std::exception_ptr initialization_error;
        {
            std::lock_guard lock(mutex_);
            if (initialized_) {
                return;
            }
            initialization_error = initialization_error_;
        }
        if (initialization_error != nullptr) {
            std::rethrow_exception(initialization_error);
        }
        ThrowInfo(ErrorCode::UnexpectedError,
                  "py_udf: CPython initialization did not run");
    }

    bool
    initialized() const {
        std::lock_guard lock(mutex_);
        return initialized_;
    }

    PyObject*
    load_instances() const {
        return load_instances_;
    }

    PyObject*
    close_instances() const {
        return close_instances_;
    }

 private:
    void
    InitializeOnce() {
        if (Py_IsInitialized()) {
            ThrowInfo(
                ErrorCode::UnexpectedError,
                "py_udf: CPython was initialized before the isolated PyUDF "
                "runtime");
        }
        PyConfig config;
        PyConfig_InitIsolatedConfig(&config);
        config.use_environment = 0;
        config.user_site_directory = 0;
        config.site_import = 1;
        config.parse_argv = 0;

        const auto initialize_status = Py_InitializeFromConfig(&config);
        PyConfig_Clear(&config);
        if (PyStatus_Exception(initialize_status)) {
            ThrowInfo(ErrorCode::UnexpectedError,
                      "py_udf: cannot initialize isolated CPython: {}",
                      initialize_status.err_msg == nullptr
                          ? "unknown error"
                          : initialize_status.err_msg);
        }

        // Py_InitializeFromConfig returns with the initial GIL held. CPython
        // cannot be safely reinitialized after wrapper import begins, so an
        // import/contract failure is permanent for this process.
        try {
            ImportTrustedWrapper();
        } catch (...) {
            PyEval_SaveThread();
            throw;
        }
        PyEval_SaveThread();
    }

    void
    ImportTrustedWrapper() {
        // The selected Python environment must already contain the trusted
        // milvus_pyudf_runtime package in its system site-packages. Isolated
        // mode ignores PYTHONPATH and user site while site initialization makes
        // that preinstalled package available through the normal import path.

        // CPython owns its process lifetime after a successful initialization.
        // There is intentionally no Py_FinalizeEx path in this runtime.
        auto module = OwnedPyObject::Adopt(
            PyImport_ImportModule("milvus_pyudf_runtime"));
        if (module.get() == nullptr) {
            ThrowPythonFailure("cannot import trusted runtime wrapper");
        }
        auto api_version = OwnedPyObject::Adopt(
            PyObject_GetAttrString(module.get(), "RUNTIME_API_VERSION"));
        if (api_version.get() == nullptr) {
            ThrowPythonFailure("trusted runtime wrapper has no API version");
        }
        if (!PyLong_Check(api_version.get()) ||
            PyLong_AsLong(api_version.get()) != 1) {
            ThrowInfo(
                ErrorCode::UnexpectedError,
                "py_udf: trusted runtime wrapper has incompatible API version");
        }
        auto loader = OwnedPyObject::Adopt(
            PyObject_GetAttrString(module.get(), "load_instances"));
        if (loader.get() == nullptr || !PyCallable_Check(loader.get())) {
            ThrowPythonFailure(
                "trusted runtime wrapper has no callable load_instances");
        }
        auto closer = OwnedPyObject::Adopt(
            PyObject_GetAttrString(module.get(), "close_instances"));
        if (closer.get() == nullptr || !PyCallable_Check(closer.get())) {
            ThrowPythonFailure(
                "trusted runtime wrapper has no callable close_instances");
        }

        module_ = module.release();
        load_instances_ = loader.release();
        close_instances_ = closer.release();
    }

    mutable std::mutex mutex_;
    std::once_flag initialization_once_;
    bool initialized_ = false;
    std::exception_ptr initialization_error_;
    PyObject* module_ = nullptr;
    PyObject* load_instances_ = nullptr;
    PyObject* close_instances_ = nullptr;
};

RuntimeState&
Runtime() {
    // Intentionally leaked: Python objects must not be decref'ed during static
    // destruction after the interpreter's process-lifetime shutdown sequence.
    static auto* state = new RuntimeState();
    return *state;
}

class CPythonPyUDFResource final : public PyUDFResource {
 public:
    explicit CPythonPyUDFResource(PyObject* instances) : instances_(instances) {
    }

    void
    Close() override {
        std::lock_guard lock(mutex_);
        if (closed_) {
            return;
        }
        closed_ = true;

        GilGuard gil;
        std::exception_ptr first_failure;
        if (instances_ != nullptr) {
            auto result = OwnedPyObject::Adopt(PyObject_CallFunctionObjArgs(
                Runtime().close_instances(), instances_, nullptr));
            if (result.get() == nullptr) {
                try {
                    ThrowPythonFailure("Python resource close failed");
                } catch (...) {
                    first_failure = std::current_exception();
                }
            }
            Py_DECREF(instances_);
            instances_ = nullptr;
        }
        if (first_failure != nullptr) {
            std::rethrow_exception(first_failure);
        }
    }

    ~CPythonPyUDFResource() override {
        // DeletePyUDFResource calls Close. This fallback protects C++ ownership
        // paths and cannot report close failures from a destructor.
        try {
            Close();
        } catch (...) {
        }
    }

 private:
    std::mutex mutex_;
    bool closed_ = false;
    PyObject* instances_ = nullptr;
};

void
ValidateRequest(const uint8_t* serialized_request,
                uint64_t serialized_request_len,
                milvus::proto::cgo::PyUDFLoadRequest* request) {
    if (serialized_request_len == 0) {
        ThrowInfo(ErrorCode::UnexpectedError,
                  "py_udf: serialized load request is empty");
    }
    if (serialized_request == nullptr) {
        ThrowInfo(ErrorCode::UnexpectedError,
                  "py_udf: serialized load request pointer is nil");
    }
    if (serialized_request_len > static_cast<uint64_t>(INT_MAX)) {
        ThrowInfo(ErrorCode::UnexpectedError,
                  "py_udf: serialized load request exceeds protobuf parse "
                  "limit");
    }
    if (!request->ParseFromArray(serialized_request,
                                 static_cast<int>(serialized_request_len))) {
        ThrowInfo(ErrorCode::UnexpectedError,
                  "py_udf: serialized load request is malformed");
    }
    if (request->ByteSizeLong() == 0) {
        ThrowInfo(ErrorCode::UnexpectedError,
                  "py_udf: serialized load request has no protocol fields");
    }

    const auto valid_text = [](const std::string& value) {
        return !IsBlank(value) && IsValidUTF8(value);
    };
    if (!valid_text(request->resource_name()) ||
        !valid_text(request->resource_path()) ||
        !valid_text(request->local_path()) || !valid_text(request->stage())) {
        ThrowInfo(
            ErrorCode::UnexpectedError,
            "py_udf: serialized load request has blank or invalid UTF-8 "
            "protocol fields");
    }
    if (request->instance_count() <= 0) {
        ThrowInfo(
            ErrorCode::UnexpectedError,
            "py_udf: serialized load request instance_count must be positive");
    }

    const std::filesystem::path local_path(request->local_path());
    std::error_code error;
    if (local_path.extension() != ".whl" ||
        !std::filesystem::is_regular_file(local_path, error) || error) {
        ThrowInfo(
            ErrorCode::FileOpenFailed,
            "py_udf: local wheel is not a readable regular .whl file");
    }
    // Opening the exact path distinguishes an inaccessible file from a merely
    // stat-able path. The wrapper independently validates zip integrity.
    std::ifstream wheel(local_path, std::ios::binary);
    if (!wheel.good()) {
        ThrowInfo(ErrorCode::FileOpenFailed,
                  "py_udf: local wheel cannot be opened for reading");
    }
}

}  // namespace

bool
RuntimeBuildEnabled() noexcept {
    return true;
}

void
InitializeRuntime() {
    Runtime().Initialize();
}

std::unique_ptr<PyUDFResource>
LoadResource(const uint8_t* serialized_request,
             uint64_t serialized_request_len) {
    if (!Runtime().initialized()) {
        ThrowInfo(ErrorCode::UnexpectedError,
                  "py_udf: runtime has not been initialized");
    }

    milvus::proto::cgo::PyUDFLoadRequest request;
    ValidateRequest(serialized_request, serialized_request_len, &request);

    GilGuard gil;
    auto resource_identity = OwnedPyObject::Adopt(
        PyLong_FromLongLong(request.resource_id()));
    if (resource_identity.get() == nullptr) {
        ThrowPythonFailure("cannot build resource identity");
    }
    auto resource_name = OwnedPyObject::Adopt(
        PyUnicode_FromString(request.resource_name().c_str()));
    auto local_path = OwnedPyObject::Adopt(
        PyUnicode_FromString(request.local_path().c_str()));
    auto stage = OwnedPyObject::Adopt(
        PyUnicode_FromString(request.stage().c_str()));
    auto instance_count = OwnedPyObject::Adopt(
        PyLong_FromLong(request.instance_count()));
    if (resource_name.get() == nullptr || local_path.get() == nullptr ||
        stage.get() == nullptr || instance_count.get() == nullptr) {
        ThrowPythonFailure("cannot build Python load arguments");
    }
    auto instances = OwnedPyObject::Adopt(
        PyObject_CallFunctionObjArgs(Runtime().load_instances(),
                                     resource_name.get(),
                                     local_path.get(),
                                     stage.get(),
                                     instance_count.get(),
                                     resource_identity.get(),
                                     nullptr));
    if (instances.get() == nullptr) {
        ThrowPythonFailure("Python UDF load failed");
    }
    if (!PySequence_Check(instances.get()) ||
        PySequence_Size(instances.get()) != request.instance_count()) {
        ThrowInfo(
            ErrorCode::UnexpectedError,
            "py_udf: trusted wrapper returned an invalid instance sequence");
    }

    return std::make_unique<CPythonPyUDFResource>(instances.release());
}

}  // namespace milvus::pyudf
