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
#include <utility>

using namespace onnxruntime;
using namespace onnxruntime::common;

using ONNX_NAMESPACE::ModelProto;
using ONNX_NAMESPACE::NodeProto;
using ONNX_NAMESPACE::ValueInfoProto;
using std::string;
using std::vector;

#define FOR(i, range) for (auto i = decltype(range)(0); i < range; i++)

namespace dqx {
struct Tensor {
  using dim_t = int64_t;
  using shape_t = std::vector<dim_t>;

  void* buf = nullptr;
  shape_t shape;
  int data_type;

  Tensor() = default;

  Tensor(const void* buf, shape_t shape, int data_type) : shape(std::move(shape)), data_type(data_type) {
    this->buf = malloc(byte_size());
    memcpy(this->buf, buf, byte_size());
  }

  dim_t elem_count() const {
    dim_t count = 1;
    for (const auto& x : shape) {
      count *= x;
    }
    return count;
  }

  dim_t byte_size() const {
    int elem_size;

    switch (data_type) {
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
        throw std::runtime_error("Type " + std::to_string(data_type) + " in Clone has not been supported");
      }
    }
    return elem_count() * elem_size;
  }

  template <typename T>
  T* mutable_data() {
    return static_cast<T*>(buf);
  }

  template <typename T>
  T* data() const {
    return static_cast<T*>(buf);
  }

  Tensor(const Tensor& t) {
    shape = t.shape;
    data_type = t.data_type;
    buf = malloc(byte_size());
    memcpy(buf, t.buf, byte_size());
  }

  Tensor& operator=(Tensor t) {
    using std::swap;
    swap(shape, t.shape);
    swap(data_type, t.data_type);
    swap(buf, t.buf);
    return *this;
  }

  Tensor(Tensor&& t) {
    shape = std::move(t.shape);
    data_type = t.data_type;
    buf = t.buf;
    t.buf = nullptr;
  }

  Tensor& operator=(Tensor&& t) {
    using std::swap;
    swap(shape, t.shape);
    swap(data_type, t.data_type);
    swap(buf, t.buf);
    return *this;
  }

  ~Tensor() {
    if (buf != nullptr) {
      free(buf);
    }
  }
};
}  // namespace dqx

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

using Result = std::map<std::string, dqx::Tensor>;

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
#define ADD_INPUT_WITH_DTYPE(dtype_num, dtype)                      \
  case dtype_num: {                                                 \
    dtype* buf = static_cast<dtype*>(malloc(size * sizeof(dtype))); \
    srand(time(NULL));                                              \
    FOR(i, size) {                                                  \
      buf[i] = rand() % 2000 - 1000;                                \
    }                                                               \
    inputs.emplace(p.first, dqx::Tensor(buf, p.second, elem_type)); \
    free(buf);                                                      \
    break;                                                          \
  }

      // inputs.emplace(p.first, Ort::Value::CreateTensor<dtype>(memory_info, buf, size, p.second.data(), p.second.size()));
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

Result ForwardWithInput(const ModelProto& model_proto, Result inputs) {
  Result outputs;
  std::shared_ptr<Model> model;
  Model::Load(model_proto, model, nullptr, logging::Logger());
  Graph& graph = model->MainGraph();

  auto GetInputArg = [&graph](const std::string& name) {
    for (const auto& x : graph.GetInputs()) {
      if (x->Name() == name) {
        return x;
      }
    }
  };

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  for (auto& input : inputs) {
    auto input_arg = GetInputArg(input.first);
    Ort::Value input_tensor = Ort::Value::CreateTensor(memory_info, input.second.buf, input.second.byte_size(), input.second.shape.data(), input.second.shape.size(), ONNXTensorElementDataType(input.second.data_type));
    ONNX_NAMESPACE::TensorProto input_tensorproto =
        utils::TensorToTensorProto(input_tensor.release()->Get<Tensor>(), input_arg->Name(), *input_arg->TypeAsProto());

    graph.AddInitializedTensor(input_tensorproto);
  }

  std::vector<std::string> output_names;
  for (const auto& x : graph.GetOutputs()) {
    output_names.push_back(x->Name());
  }
  auto ret = graph.Resolve();
  if (!ret.IsOK()) {
    throw std::runtime_error(ret.ErrorMessage() + " " + ret.ToString());
  }
  GraphViewer graph_viewer(graph);
  auto& order = graph_viewer.GetNodesInTopologicalOrder();

  for (NodeIndex i : order) {
    auto* node = graph.GetNode(i);
    if (!node) {
      continue;
    }

    // TODO:
    // ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    InitializedTensorSet constant_inputs;

    // we currently constant fold using the CPU EP only.
    // if the node is assigned to a different EP we can run it if it's an ONNX op as we have CPU based implementations
    // for all ONNX ops. if it's from a different domain we can't.
    // NOTE: This is in addition to the IsSupportedProvider check below which will optionally do further filtering
    // on the EPs we constant fold for.
    auto ep_type = node->GetExecutionProviderType();
    bool cpu_ep = ep_type == kCpuExecutionProvider;
    if (!cpu_ep && node->Domain() != kOnnxDomain) {
      continue;
    }

    // Check if constant folding can be applied on this node.
    // if (!graph_utils::IsSupportedProvider(*node, GetCompatibleExecutionProviders()) ||
    //     excluded_op_types_.find(node->OpType()) != excluded_op_types_.end() ||
    if (  // constant folding does not support executing a node that includes subgraphs (control flow operators,
        // such as If/Loop/Scan, fall into this category). individual nodes in the subgraph will be processed
        // by the Recurse call above
        node->ContainsSubgraph() ||
        !graph_utils::AllNodeInputsAreConstant(graph, *node, constant_inputs)) {
      continue;
    }

    // override the EP while setting up OptimizerExecutionFrame::Info so that it will use the CPU kernel for Compute.
    if (!cpu_ep) {
      node->SetExecutionProviderType(kCpuExecutionProvider);
    }

    // Create execution frame for executing constant nodes.
    OptimizerExecutionFrame::Info info({node}, constant_inputs);

    // undo the EP change in case something fails prior to node removal
    if (!cpu_ep) {
      node->SetExecutionProviderType(ep_type);
    }

    std::vector<int> fetch_mlvalue_idxs;
    for (const auto* node_out : node->OutputDefs()) {
      fetch_mlvalue_idxs.push_back(info.GetMLValueIndex(node_out->Name()));
    }

    OptimizerExecutionFrame frame(info, fetch_mlvalue_idxs);

    auto* kernel = info.GetKernel(node->Index());
    OpKernelContext op_kernel_context(&frame, kernel, nullptr, onnxruntime::logging::LoggingManager::DefaultLogger());

    auto s = kernel->Compute(&op_kernel_context);
    if (s != Status::OK()) {
      throw std::runtime_error(s.ErrorMessage());
    }

    std::vector<OrtValue> fetches;
    s = frame.GetOutputs(fetches);
    if (s != Status::OK()) {
      throw std::runtime_error(s.ErrorMessage());
    }

    // Go over all output node args and substitute them with the newly computed tensors, which will be
    // added to the graph as initializers.
    ORT_ENFORCE(fetches.size() == node->OutputDefs().size());
    for (size_t fetch_idx = 0; fetch_idx < fetches.size(); ++fetch_idx) {
      OrtValue& ort_value = fetches[fetch_idx];

      if (!ort_value.IsTensor()) {
        std::stringstream ss;
        ss << "Unsupported output type of " << ort_value.Type()
           << ". Can't constant fold " << node->OpType() << " node '" << node->Name() << "'";
        throw std::runtime_error(ss.str());
      }

      // Build the TensorProto that corresponds to the computed OrtValue and add it as initializer to the graph.
      const auto* constant_arg_out = node->OutputDefs()[fetch_idx];
      ORT_ENFORCE(ort_value.IsTensor());
      const Tensor& out_tensor = ort_value.Get<Tensor>();
      // std::cout << "name: " << constant_arg_out->Name();
      //       for (int i = 0; i < 20; i++) {
      //           std::cout << ", value " << i << ": " << out_tensor.Data<float>()[i];
      //       }
      // std::cout << std::endl;
      ONNX_NAMESPACE::TensorProto out_tensorproto =
          utils::TensorToTensorProto(out_tensor, constant_arg_out->Name(), *constant_arg_out->TypeAsProto());

      graph.AddInitializedTensor(out_tensorproto);

      if (std::find(output_names.begin(), output_names.end(), constant_arg_out->Name()) != output_names.end()) {
        const auto elem_type = constant_arg_out->TypeAsProto()->tensor_type().elem_type();

        auto tensor = dqx::Tensor(ort_value.Get<Tensor>().DataRaw(), out_tensor.Shape().GetDims(), elem_type);

        outputs.emplace(constant_arg_out->Name(), tensor);
      }
    }

    // Remove the output edges of the constant node and then remove the node itself.
    graph_utils::RemoveNodeOutputEdges(graph, *node);
    graph.RemoveNode(node->Index());
  }
  return outputs;
}

Result Forward(Ort::Env& env, const ModelProto& model, const Result& known_inputs = Result{}, std::map<string, MyTensorShape> input_shapes = std::map<string, MyTensorShape>{}) {
  auto rand_inputs = GenerateRandInputs(model, input_shapes);
  for (auto& x : known_inputs) {
    std::cout << rand_inputs[x.first].data<float>()[0] << std::endl;
    rand_inputs[x.first] = x.second;
    std::cout << rand_inputs[x.first].data<float>()[0] << std::endl;
  }
  Result res = ForwardWithInput(model, rand_inputs);
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
#define ADD_ATTR_WITH_DTYPE(onnx_type, dtype)                   \
  case onnx_type: {                                             \
    std::vector<dtype> tmp;                                     \
    FOR(i, v.elem_count()) {                                    \
      tmp.push_back(v.mutable_data<dtype>()[i]);                \
    }                                                           \
    auto tp = ONNX_NAMESPACE::ToTensor(tmp);                    \
    for (const auto& dim : v.shape) {                           \
      tp.add_dims(dim);                                         \
    }                                                           \
    auto new_attr = ONNX_NAMESPACE::MakeAttribute("value", tp); \
    *new_node.add_attribute() = new_attr;                       \
    break;                                                      \
  }

        const auto dtype = v.data_type;
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

bool isEqual(const dqx::Tensor& value1, const dqx::Tensor& value2) {
  const auto elem_type = value1.data_type;
  const auto elem_type2 = value2.data_type;
  if (elem_type != elem_type2) {
    return false;
  }
  const auto shape = value1.shape;
  const auto shape2 = value2.shape;
  if (shape != shape2) {
    return false;
  }
  const auto cnt = value1.elem_count();
  float atol = 1e-5;
  float rtol = 1e-3;
  switch (elem_type) {
#define COMPARE_WITH_SIGNED_DTYPE(onnx_dtype, dtype)                \
  case onnx_dtype: {                                                \
    const auto* p1 = value1.data<dtype>();                          \
    const auto* p2 = value2.data<dtype>();                          \
    FOR(i, cnt) {                                                   \
      if (std::abs(*p1 - *p2) > (atol + std::abs(rtol * (*p1)))) {  \
        std::cout << "p1: " << *p1 << ", p2: " << *p2 << std::endl; \
        return false;                                               \
      }                                                             \
      p1++;                                                         \
      p2++;                                                         \
    }                                                               \
    break;                                                          \
  }
#define COMPARE_WITH_UNSIGNED_DTYPE(onnx_dtype, dtype)                                       \
  case onnx_dtype: {                                                                         \
    const auto* p1 = value1.data<dtype>();                                                   \
    const auto* p2 = value2.data<dtype>();                                                   \
    FOR(i, cnt) {                                                                            \
      if ((std::max<dtype>(*p1, *p2) - std::min<dtype>(*p1, *p2)) > (atol + rtol * (*p1))) { \
        std::cout << "p1: " << *p1 << ", p2: " << *p2 << std::endl;                          \
        return false;                                                                        \
      }                                                                                      \
      p1++;                                                                                  \
      p2++;                                                                                  \
    }                                                                                        \
    break;                                                                                   \
  }

    COMPARE_WITH_SIGNED_DTYPE(1, float)
    COMPARE_WITH_UNSIGNED_DTYPE(2, uint8_t)
    COMPARE_WITH_SIGNED_DTYPE(3, int8_t)
    COMPARE_WITH_UNSIGNED_DTYPE(4, uint16_t)
    COMPARE_WITH_SIGNED_DTYPE(5, int16_t)
    COMPARE_WITH_SIGNED_DTYPE(6, int32_t)
    COMPARE_WITH_SIGNED_DTYPE(7, int64_t)
    COMPARE_WITH_SIGNED_DTYPE(9, bool)
    COMPARE_WITH_SIGNED_DTYPE(11, double)
    COMPARE_WITH_UNSIGNED_DTYPE(12, uint32_t)
    COMPARE_WITH_UNSIGNED_DTYPE(13, uint64_t)

#undef COMPARE_WITH_SIGNED_DTYPE
#undef COMPARE_WITH_UNSIGNED_DTYPE
    default: {
      throw std::runtime_error("type " + std::to_string(elem_type) + " is not supported in is_equal()");
    }
  }
  return true;
}

bool Check(const ModelProto& model1, const ModelProto& model2, std::map<string, MyTensorShape> input_shapes = std::map<string, MyTensorShape>{}, const int n = 1) {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "dqxdqx");
  // TODO: checker
  for (int i = 0; i < n; i++) {
    auto rand_inputs = GenerateRandInputs(model1, input_shapes);
    auto res1 = Forward(env, model1, rand_inputs);
    // auto res2 = Forward(env, model2, rand_inputs);
    auto res2 = Forward(env, model2, rand_inputs);
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

#ifndef __EMSCRIPTEN__
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

  return 0;
}
#endif

#ifdef __EMSCRIPTEN__
#ifdef ONNXSIM_STANDALONE

extern "C" {

struct WasmBuffer {
  unsigned char* output_buffer1 = nullptr;
  unsigned char* output_buffer2 = nullptr;
  unsigned char* output_buffer3 = nullptr;
  size_t output_buffer_size1 = 0;
  size_t output_buffer_size2 = 0;
  size_t output_buffer_size3 = 0;

  void freeBuffers() {
    freeBuffer1();
    freeBuffer2();
    freeBuffer3();
  }
  void freeBuffer1() {
    if (output_buffer1 != nullptr) {
      free(output_buffer1);
      output_buffer1 = nullptr;
      output_buffer_size1 = 0;
    }
  }
  void freeBuffer2() {
    if (output_buffer2 != nullptr) {
      free(output_buffer2);
      output_buffer2 = nullptr;
      output_buffer_size2 = 0;
    }
  }
  void freeBuffer3() {
    if (output_buffer3 != nullptr) {
      free(output_buffer3);
      output_buffer3 = nullptr;
      output_buffer_size3 = 0;
    }
  }
  void setBuffer1(void* buf, const size_t buflen) {
    // we own the buf
    output_buffer1 = static_cast<unsigned char*>(buf);
    output_buffer_size1 = buflen;
  }
  void setBuffer1(const std::string& str) {
    output_buffer1 = static_cast<unsigned char*>(malloc(str.size()));
    memcpy(output_buffer1, str.c_str(), str.size());
    output_buffer_size1 = str.size();
  }
  void setBuffer2(Buffer buf) { setBuffer2(buf.first, buf.second); }
  void setBuffer2(void* buf, const size_t buflen) {
    // we own the buf
    output_buffer2 = static_cast<unsigned char*>(buf);
    output_buffer_size2 = buflen;
  }
  void setBuffer2(const std::string& str) {
    output_buffer2 = static_cast<unsigned char*>(malloc(str.size()));
    memcpy(output_buffer2, str.c_str(), str.size());
    output_buffer_size2 = str.size();
  }
  void setBuffer3(const std::string& str) {
    output_buffer3 = static_cast<unsigned char*>(malloc(str.size()));
    memcpy(output_buffer3, str.c_str(), str.size());
    output_buffer_size3 = str.size();
  }
};

WasmBuffer* create_exporter() {
  WasmBuffer* ctx;

  ctx = static_cast<WasmBuffer*>(malloc(sizeof(WasmBuffer)));
  ctx->output_buffer_size1 = 0;
  ctx->output_buffer_size2 = 0;
  ctx->output_buffer_size3 = 0;

  return ctx;
}

void free_exporter(WasmBuffer* ctx) {
  if (ctx != NULL) {
    ctx->freeBuffers();
    free(ctx);
    ctx = NULL;
  }
}

unsigned char* get_buffer1(WasmBuffer* ctx) { return ctx->output_buffer1; }

size_t get_buffer_size1(WasmBuffer* ctx) { return ctx->output_buffer_size1; }

unsigned char* get_buffer2(WasmBuffer* ctx) { return ctx->output_buffer2; }

size_t get_buffer_size2(WasmBuffer* ctx) { return ctx->output_buffer_size2; }

unsigned char* get_buffer3(WasmBuffer* ctx) { return ctx->output_buffer3; }

size_t get_buffer_size3(WasmBuffer* ctx) { return ctx->output_buffer_size3; }

bool onnxsimplify_export(WasmBuffer* ctx, void* buf, const size_t len) {
  try {
    onnx::ModelProto opt_model;
    {
      onnx::ModelProto model;
      bool s1 = model.ParseFromArray(buf, len);
      free(buf);
      if (!s1) {
        ctx->setBuffer3("parsing ONNX model fails");
        return false;
      }
      opt_model = Simplify(model);
      bool check = Check(opt_model, model);
      if (check) {
        std::cout << "check ok" << std::endl;
      } else {
        std::cout << "check failed" << std::endl;
      }
    }
    auto byte_size = opt_model.ByteSizeLong();
    void *buf = malloc(byte_size);
    bool s2 = opt_model.SerializeToArray(buf, byte_size);
    if (!s2) {
      ctx->setBuffer3("serialing ONNX model fails");
      return false;
    }
    ctx->setBuffer1(buf, byte_size);
    return true;
  } catch (std::exception &e) {
    ctx->setBuffer3(e.what());
    return false;
  }

}
}

#endif
#endif

