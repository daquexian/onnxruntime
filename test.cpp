// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <onnx/defs/attr_proto_util.h>
#include <onnx/shape_inference/implementation.h>
#include "core/graph/model.h"
#include "core/optimizer/constant_folding.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/optimizer_execution_frame.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensorprotoutils.h"
#include "onnx/defs/tensor_proto_util.h"
#include <onnx/optimizer/optimize.h>

#include <cmath>
#include <fstream>
#include <cstdlib>

using namespace onnxruntime;
using namespace onnxruntime::common;

using ONNX_NAMESPACE::ModelProto;
using ONNX_NAMESPACE::NodeProto;
using ONNX_NAMESPACE::ValueInfoProto;
using std::string;
using std::vector;

std::vector<NodeProto> GetConstNode(const ONNX_NAMESPACE::ModelProto& m) {
  /*
    const_nodes = []
    const_tensors = [x.name for x in m.graph.initializer]
    const_tensors.extend([node.output[0]
                          for node in m.graph.node if node.op_type == 'Constant'])
    # If one of the input of a node is produced (directly or indirectly) by nms,
    # we consider the output of this node doesn't have constant shape,
    # so we do not simplify a such node even if the node is Shape op
    tensors_nms = []
    for node in m.graph.node:
        if any(x in tensors_nms for x in node.input):
            tensors_nms.extend(node.output)
        elif node.op_type == 'Shape':
            const_nodes.append(node)
            const_tensors.extend(node.output)
        elif node.op_type == 'NonMaxSuppression':
            tensors_nms.extend(node.output)
        elif all([x in const_tensors for x in node.input]):
            const_nodes.append(node)
            const_tensors.extend(node.output)
    return copy.deepcopy(const_nodes)
    */
  std::vector<ONNX_NAMESPACE::NodeProto> const_nodes;
  std::vector<std::string> const_tensors;
  for (const auto& x : m.graph().initializer()) {
    const_tensors.push_back(x.name());
  }
  for (const auto& x : m.graph().node()) {
    if (x.has_op_type() && x.op_type() == "Constant") {
      const_tensors.push_back(x.output(0));
    }
  }
  std::vector<std::string> tensors_nms;
  for (const auto& node : m.graph().node()) {
    bool has_input_from_nms = false;
    for (const auto& x : node.input()) {
      if (std::find(tensors_nms.begin(), tensors_nms.end(), x) != tensors_nms.end()) {
        has_input_from_nms = true;
        break;
      }
    }
    if (has_input_from_nms) {
      for (const auto& x : node.output()) {
        tensors_nms.push_back(x);
      }
    } else if (node.has_op_type() && node.op_type() == "Shape") {
      const_nodes.push_back(node);
      for (const auto& x : node.output()) {
        const_tensors.push_back(x);
      }
    } else if (node.has_op_type() && node.op_type() == "NonMaxSuppression") {
      for (const auto& x : node.output()) {
        tensors_nms.push_back(x);
      }
    } else {
      bool all_const_input = true;
      for (const auto& x : node.input()) {
        if (std::find(const_tensors.begin(), const_tensors.end(), x) == const_tensors.end()) {
          all_const_input = false;
        }
      }
      if (all_const_input) {
        const_nodes.push_back(node);
        for (const auto& x : node.output()) {
          const_tensors.push_back(x);
        }
      }
    }
  }

  return const_nodes;
}

void AddFeatureToOutput(ModelProto& m, const std::vector<NodeProto>& nodes) {
  for (const auto& node : nodes) {
    for (const auto& output : node.output()) {
      auto* value_info = m.mutable_graph()->add_output();
      value_info->set_name(output);
    }
  }
}

using Result = std::map<std::string, Ort::Value>;

std::vector<std::string> GetInputNames(const ModelProto& model) {
  std::vector<std::string> inputs;
  for (const auto& x : model.graph().input()) {
    bool in_initer = false;
    for (const auto& initer : model.graph().initializer()) {
      if (x.name() == initer.name()) {
        in_initer = true;
        break;
      }
    }
    if (!in_initer) {
      inputs.push_back(x.name());
    }
  }
  return inputs;
}

ValueInfoProto GetValueInfoAll(const ModelProto& m, const std::string& name) {
  for (const auto& v : m.graph().value_info()) {
    if (v.has_name() && v.name() == name) {
      return v;
    }
  }
  for (const auto& v : m.graph().input()) {
    if (v.has_name() && v.name() == name) {
      return v;
    }
  }
  for (const auto& v : m.graph().output()) {
    if (v.has_name() && v.name() == name) {
      return v;
    }
  }
  throw std::runtime_error("Cannot find value info of " + name);
}

using MyTensorShape = std::vector<int64_t>;

MyTensorShape GetShapeFromValueInfoProto(const ValueInfoProto& v) {
  MyTensorShape shape;
  for (const auto& dim : v.type().tensor_type().shape().dim()) {
    shape.push_back(dim.dim_value());
  }
  return shape;
}

MyTensorShape GetShape(const ModelProto& m, const std::string& name) {
  const auto v = GetValueInfoAll(m, name);
  return GetShapeFromValueInfoProto(v);
}

int GetElemType(const ModelProto& m, const string& name) {
  const auto v = GetValueInfoAll(m, name);
  return v.type().tensor_type().elem_type();
}

std::vector<void*> ps;

Result GenerateRandInputs(const ModelProto& model, std::map<string, MyTensorShape> input_shapes = std::map<string, MyTensorShape>{}) {
  const auto input_names = GetInputNames(model);
  std::map<string, MyTensorShape> full_input_shapes;
  for (const auto& ipt : input_names) {
    if (input_shapes.find(ipt) != input_shapes.end()) {
      full_input_shapes[ipt] = input_shapes[ipt];
    } else {
      full_input_shapes[ipt] = GetShape(model, ipt);
    }
  }
  Result inputs;
  for (const auto& p : full_input_shapes) {
    size_t size = 1;
    for (const auto& dim : p.second) {
      if (dim <= 0) {
        throw std::runtime_error("Input " + p.first + " has dynamic size");
      }
      size *= dim;
    }

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const auto elem_type = GetElemType(model, p.first);
    switch (elem_type) {
#define ADD_INPUT_WITH_DTYPE(dtype_num, dtype)                                                                          \
  case dtype_num: {                                                                                                     \
    dtype* buf = static_cast<dtype*>(malloc(size * sizeof(dtype)));                                                     \
    ps.push_back(buf);                                                                                                  \
    srand(time(NULL));                                                                                                  \
    for (int i = 0; i < size; i++) {                                                                                    \
      buf[i] = rand() % 2000 - 1000;                                                                                    \
    }                                                                                                                   \
    inputs.emplace(p.first, Ort::Value::CreateTensor<dtype>(memory_info, buf, size, p.second.data(), p.second.size())); \
    break;                                                                                                              \
  }

      ADD_INPUT_WITH_DTYPE(1, float)
      ADD_INPUT_WITH_DTYPE(2, uint8_t)
      ADD_INPUT_WITH_DTYPE(3, int8_t)
      ADD_INPUT_WITH_DTYPE(4, uint16_t)
      ADD_INPUT_WITH_DTYPE(5, int16_t)
      ADD_INPUT_WITH_DTYPE(6, int32_t)
      ADD_INPUT_WITH_DTYPE(7, int64_t)
      ADD_INPUT_WITH_DTYPE(9, bool)
      ADD_INPUT_WITH_DTYPE(11, double)
      ADD_INPUT_WITH_DTYPE(12, uint32_t)
      ADD_INPUT_WITH_DTYPE(13, uint64_t)

#undef ADD_INPUT_WITH_DTYPE

      default: {
        throw std::runtime_error("Type " + std::to_string(elem_type) + " has not been supported");
      }
    }
  }
  return inputs;
}

Result Clone(Result& r) {
  Result res;

  for (auto& p : r) {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto& value = p.second;
    if (!value.IsTensor()) {
      throw std::runtime_error("Clone only supports tensor");
    }
    int elem_size;
    auto elem_type = value.GetTensorTypeAndShapeInfo().GetElementType();
    switch (elem_type) {
#define SET_ELEM_SIZE(onnx_dtype, size) \
  case onnx_dtype: {                    \
    elem_size = size;                   \
    break;                              \
  }

      SET_ELEM_SIZE(1, 4)
      SET_ELEM_SIZE(2, 1)
      SET_ELEM_SIZE(3, 1)
      SET_ELEM_SIZE(4, 2)
      SET_ELEM_SIZE(5, 2)
      SET_ELEM_SIZE(6, 4)
      SET_ELEM_SIZE(7, 8)
      SET_ELEM_SIZE(11, 8)
      SET_ELEM_SIZE(12, 4)
      SET_ELEM_SIZE(13, 8)
#undef SET_ELEM_SIZE

      default: {
        throw std::runtime_error("Type " + std::to_string(elem_type) + " in Clone has not been supported");
      }
    }
    auto byte_size = value.GetTensorTypeAndShapeInfo().GetElementCount() * elem_size;
    void* buf = malloc(byte_size);
    memcpy(buf, value.GetTensorMutableData<void>(), byte_size);
    ps.push_back(buf);
    auto new_value = Ort::Value::CreateTensor(memory_info, buf, byte_size, value.GetTensorTypeAndShapeInfo().GetShape().data(), value.GetTensorTypeAndShapeInfo().GetDimensionsCount(), value.GetTensorTypeAndShapeInfo().GetElementType());
    res.emplace(p.first, std::move(new_value));
  }
  return res;
}

Result Forward(Ort::Env& env, const ModelProto& model, Result&& known_inputs = Result{}, std::map<string, MyTensorShape> input_shapes = std::map<string, MyTensorShape>{}) {
  Ort::SessionOptions session_options;
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
  const auto model_str = model.SerializeAsString();
  Ort::Session sess(env, model_str.c_str(), model_str.size(), session_options);

  auto rand_inputs = GenerateRandInputs(model, input_shapes);
  vector<const char*> input_names;
  vector<Ort::Value> input_values;
  for (auto& x : rand_inputs) {
    input_names.push_back(x.first.c_str());
    if (known_inputs.find(x.first) != known_inputs.end()) {
      input_values.push_back(std::move(known_inputs.at(x.first)));
      // input_values.push_back(std::move(known_inputs[x.first]));
    } else {
      input_values.push_back(std::move(x.second));
    }
  }
  vector<const char*> output_names;
  for (const auto& x : model.graph().output()) {
    output_names.push_back(x.name().c_str());
  }

  auto outputs = sess.Run(Ort::RunOptions{nullptr}, input_names.data(), input_values.data(), input_names.size(), output_names.data(), output_names.size());
  Result res;
  for (size_t i = 0; i < output_names.size(); i++) {
    res.emplace(output_names[i], std::move(outputs[i]));
  }
  return res;
}

// modelproto passed by value
Result ForwardForNodeOutputs(Ort::Env& env, ModelProto model, const std::vector<NodeProto>& nodes) {  // add input_shapes
  AddFeatureToOutput(model, nodes);
  return Forward(env, model);
}

void InsertElem(google::protobuf::RepeatedPtrField<NodeProto>* repeated, int index, const NodeProto& elem) {
  repeated->Add()->CopyFrom(elem);
  for (int i = repeated->size() - 1; i > index; i--) {
    repeated->SwapElements(i, i - 1);
  }
}

void EliminateConstNodes(ModelProto& model, const vector<NodeProto>& const_nodes,
                         Result& res) {
  for (int i = 0; i < model.graph().node_size(); i++) {
    const auto& node = model.graph().node(i);

    bool is_constant = false;
    for (const auto& x : const_nodes) {
      if (x.SerializeAsString() == node.SerializeAsString()) {
        is_constant = true;
        break;
      }
    }

    if (is_constant) {
      for (const auto& output : node.output()) {
        NodeProto new_node = node;
        new_node.set_name("node_" + output);
        new_node.set_op_type("Constant");
        new_node.clear_input();
        new_node.clear_attribute();
        new_node.clear_output();
        new_node.add_output(output);
        auto& v = res.at(output);
#define ADD_ATTR_WITH_DTYPE(onnx_type, dtype)                                      \
  case onnx_type: {                                                                \
    std::vector<dtype> tmp;                                                        \
    for (size_t i = 0; i < v.GetTensorTypeAndShapeInfo().GetElementCount(); i++) { \
      tmp.push_back(v.GetTensorMutableData<dtype>()[i]);                           \
    }                                                                              \
    auto tp = ONNX_NAMESPACE::ToTensor(tmp);                                       \
    for (const auto& dim : v.GetTensorTypeAndShapeInfo().GetShape()) {             \
      tp.add_dims(dim);                                                            \
    }                                                                              \
    auto new_attr = ONNX_NAMESPACE::MakeAttribute("value", tp);                    \
    *new_node.add_attribute() = new_attr;                                          \
    break;                                                                         \
  }

        const auto dtype = v.GetTensorTypeAndShapeInfo().GetElementType();
        switch (dtype) {
          ADD_ATTR_WITH_DTYPE(1, float)
          // ADD_ATTR_WITH_DTYPE(2, uint8_t)
          // ADD_ATTR_WITH_DTYPE(3, int8_t)
          // ADD_ATTR_WITH_DTYPE(4, uint16_t)
          // ADD_ATTR_WITH_DTYPE(5, int16_t)
          ADD_ATTR_WITH_DTYPE(6, int32_t)
          ADD_ATTR_WITH_DTYPE(7, int64_t)
          ADD_ATTR_WITH_DTYPE(9, bool)
          ADD_ATTR_WITH_DTYPE(11, double)
          // ADD_ATTR_WITH_DTYPE(12, uint32_t)
          ADD_ATTR_WITH_DTYPE(13, uint64_t)
          default: {
            throw std::runtime_error("constant type " + std::to_string(dtype) + " is not supported");
          }
        }
#undef ADD_ATTR_WITH_DTYPE
        InsertElem(model.mutable_graph()->mutable_node(), i + 1, new_node);
      }
      model.mutable_graph()->mutable_node()->DeleteSubrange(i, 1);
    }
  }
}

ModelProto Simplify(ModelProto model) {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "dqxdqx");
  model = ONNX_NAMESPACE::optimization::OptimizeFixed(
      model,
      {"eliminate_deadend", "eliminate_identity", "eliminate_nop_dropout",
       "eliminate_nop_monotone_argmax", "eliminate_nop_pad",
       "extract_constant_to_initializer", "eliminate_unused_initializer",
       "eliminate_nop_transpose", "fuse_add_bias_into_conv",
       "fuse_consecutive_concats",
       "fuse_bn_into_conv",
       "fuse_consecutive_log_softmax",
       "fuse_consecutive_reduce_unsqueeze", "fuse_consecutive_squeezes",
       "fuse_consecutive_transposes", "fuse_matmul_add_bias_into_gemm",
       "fuse_pad_into_conv", "fuse_transpose_into_gemm"});
  ONNX_NAMESPACE::shape_inference::InferShapes(model);
  const auto const_nodes = GetConstNode(model);
  auto res = ForwardForNodeOutputs(env, model, const_nodes);
  EliminateConstNodes(model, const_nodes, res);

  model = ONNX_NAMESPACE::optimization::OptimizeFixed(
      model,
      {"eliminate_deadend", "eliminate_identity", "eliminate_nop_dropout",
       "eliminate_nop_monotone_argmax", "eliminate_nop_pad",
       "extract_constant_to_initializer", "eliminate_unused_initializer",
       "eliminate_nop_transpose", "fuse_add_bias_into_conv",
       "fuse_consecutive_concats",
       "fuse_bn_into_conv",
       "fuse_consecutive_log_softmax",
       "fuse_consecutive_reduce_unsqueeze", "fuse_consecutive_squeezes",
       "fuse_consecutive_transposes", "fuse_matmul_add_bias_into_gemm",
       "fuse_pad_into_conv", "fuse_transpose_into_gemm"});

  return model;
}

bool isEqual(Ort::Value& value1, Ort::Value& value2) {
  const auto elem_type = value1.GetTensorTypeAndShapeInfo().GetElementType();
  const auto elem_type2 = value2.GetTensorTypeAndShapeInfo().GetElementType();
  if (elem_type != elem_type2) {
    return false;
  }
  const auto cnt = value1.GetTensorTypeAndShapeInfo().GetElementCount();
  const auto cnt2 = value2.GetTensorTypeAndShapeInfo().GetElementCount();
  if (cnt != cnt2) {
    return false;
  }
  float atol = 1e-5;
  float rtol = 1e-3;
  switch (elem_type) {
#define COMPARE_WITH_DTYPE(onnx_dtype, dtype)                       \
  case onnx_dtype: {                                                \
    const auto* p1 = value1.GetTensorMutableData<dtype>();          \
    const auto* p2 = value2.GetTensorMutableData<dtype>();          \
    for (size_t i = 0; i < cnt; i++) {                              \
      if (std::abs(*p1 - *p2) > (atol + std::abs(rtol * (*p1)))) {  \
        std::cout << "p1: " << *p1 << ", p2: " << *p2 << std::endl; \
        return false;                                               \
      }                                                             \
      p1++;                                                         \
      p2++;                                                         \
    }                                                               \
    break;                                                          \
  }
    COMPARE_WITH_DTYPE(1, float)
    COMPARE_WITH_DTYPE(2, uint8_t)
    COMPARE_WITH_DTYPE(3, int8_t)
    COMPARE_WITH_DTYPE(4, uint16_t)
    COMPARE_WITH_DTYPE(5, int16_t)
    COMPARE_WITH_DTYPE(6, int32_t)
    COMPARE_WITH_DTYPE(7, int64_t)
    COMPARE_WITH_DTYPE(9, bool)
    COMPARE_WITH_DTYPE(11, double)
    COMPARE_WITH_DTYPE(12, uint32_t)
    COMPARE_WITH_DTYPE(13, uint64_t)
#undef COMPARE_WITH_DTYPE
    default: {
      throw std::runtime_error("type " + std::to_string(elem_type) + " is not supported in is_equal()");
    }
  }
  return true;
}

bool Check(const ModelProto& model1, const ModelProto& model2, std::map<string, MyTensorShape> input_shapes = std::map<string, MyTensorShape>{}, const int n = 3) {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "dqxdqx");
  // TODO: checker
  for (int i = 0; i < n; i++) {
    auto rand_inputs = GenerateRandInputs(model1, input_shapes);
    auto rand_inputs_copy = Clone(rand_inputs);
    auto res1 = Forward(env, model1, std::move(rand_inputs));
    auto res2 = Forward(env, model2, std::move(rand_inputs_copy));
    for (auto& x : res1) {
      if (res2.find(x.first) == res2.end()) {
        return false;
      }
      if (!isEqual(res1.at(x.first), res2.at(x.first))) {
        return false;
      }
    }
  }
  return true;
}

int main(int argc, char** argv) {
  ONNX_NAMESPACE::ModelProto model_proto;
  std::ifstream ifs(argv[1]);
  model_proto.ParseFromIstream(&ifs);
  auto new_model = Simplify(model_proto);
  bool check = Check(new_model, model_proto);
  if (check) {
    std::cout << "check ok" << std::endl;
  } else {
    std::cout << "check failed" << std::endl;
  }
  std::ofstream ofs(argv[2]);
  new_model.SerializeToOstream(&ofs);
  for (auto* p : ps) {
    free(p);
  }

  return 0;
}

// extern "C" {
//
// #ifdef __EMSCRIPTEN__
// bool onnxruntime_export(void* buffer1, const size_t bufferlen1) {
// #else
// int main(int argc, char** argv) {
// #endif
//   Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "dqxdqx");
//   std::shared_ptr<Model> model;
// #ifdef __EMSCRIPTEN__
//   ONNX_NAMESPACE::ModelProto model_proto;
//   model_proto.ParseFromString(std::string(static_cast<char*>(buffer1), bufferlen1));
//   Model::Load(model_proto, model, nullptr, logging::Logger());
// #else
//   std::string model_uri("/home/dev/files/squeezenet1.1.onnx");
//   Model::Load(model_uri, model, nullptr, logging::Logger());
// #endif
//
//   Graph& graph = model->MainGraph();
//   auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
//   size_t input_tensor_size = 224 * 224 * 3;  // simplify ... using known dim values to calculate size
//   std::vector<float> input_tensor_values(input_tensor_size);
//   for (unsigned int i = 0; i < input_tensor_size; i++) {
//     // input_tensor_values[i] = (float)i / (input_tensor_size + 1) * 10000 + 1000;
//     input_tensor_values[i] = 100;
//   }
//   std::vector<int64_t> input_node_dims{1, 3, 224, 224};  // simplify... this model has only 1 input node {1, 3, 224, 224}.
//   Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
//   const Tensor& tensor = reinterpret_cast<OrtValue**>(&input_tensor)[0]->Get<Tensor>();
//   std::cout << "data: " << tensor.Data<float>()[0] << ", " << tensor.Data<float>()[1] << std::endl;
//   ONNX_NAMESPACE::TypeProto type_proto;
//   type_proto.mutable_tensor_type()->set_elem_type(1);
//   type_proto.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
//   type_proto.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
//   type_proto.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(224);
//   type_proto.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(224);
//   ONNX_NAMESPACE::TensorProto in_tensorproto =
//       utils::TensorToTensorProto(tensor, "data", type_proto);
//
//   graph.AddInitializedTensor(in_tensorproto);
//
//   bool tmp;
//   test(graph, tmp, 0, logging::Logger());
//   std::cout << tmp << std::endl;
// #ifndef __EMSCRIPTEN__
//   auto new_proto = model->ToProto();
//   std::ofstream ofs(argv[1]);
//   new_proto.SerializeToOstream(&ofs);
//   ONNX_NAMESPACE::ModelProto opt_model = ONNX_NAMESPACE::optimization::OptimizeFixed(
//       new_proto,
//       {"eliminate_unused_initializer"});
//   opt_model.SerializeToOstream(&ofs);
// #endif
//   return 0;
// }
// }
