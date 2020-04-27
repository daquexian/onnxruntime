#include <onnx/onnx_pb.h>

#include <map>
#include <string>

using MyTensorShape = std::vector<int64_t>;

ONNX_NAMESPACE::ModelProto Simplify(ONNX_NAMESPACE::ModelProto model);
bool Check(const ONNX_NAMESPACE::ModelProto& model1, const ONNX_NAMESPACE::ModelProto& model2, std::map<std::string, MyTensorShape> input_shapes = std::map<std::string, MyTensorShape>{}, const int n = 1);
