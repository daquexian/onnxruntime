add_library(onnxruntime_mlas STATIC
    ${ONNXRUNTIME_ROOT}/core/mlas/lib/naive/mlas.cc)
target_include_directories(onnxruntime_mlas PRIVATE ${ONNXRUNTIME_ROOT}/core/mlas/inc ${ONNXRUNTIME_ROOT}/core/mlas/lib ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64)
set_target_properties(onnxruntime_mlas PROPERTIES FOLDER "ONNXRuntime")
