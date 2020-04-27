#include <onnx/onnx_pb.h>

#include <map>
#include <string>

using MyTensorShape = std::vector<int64_t>;
using MyTensorShapeMap = std::map<std::string, MyTensorShape>;

ONNX_NAMESPACE::ModelProto Simplify(ONNX_NAMESPACE::ModelProto model, bool optimize=true, MyTensorShapeMap input_shapes=MyTensorShapeMap{});
bool Check(const ONNX_NAMESPACE::ModelProto& model1, const ONNX_NAMESPACE::ModelProto& model2, std::map<std::string, MyTensorShape> input_shapes = std::map<std::string, MyTensorShape>{}, const int n = 1);
bool CheckStaticInputShape(const ONNX_NAMESPACE::ModelProto &m, const std::string &name) {
