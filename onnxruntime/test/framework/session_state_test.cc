// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>

#include "core/framework/execution_providers.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/graph/graph.h"
#include "core/graph/model.h"
#include "core/graph/op.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/graph/function_container.h"
#include "gtest/gtest.h"

using namespace ONNX_NAMESPACE;
using namespace std;

namespace onnxruntime {
namespace test {
class TestOpKernel : public OpKernel {
 public:
  TestOpKernel(const OpKernelInfo& p) : OpKernel(p) {}
  Status Compute(OpKernelContext* context) const {
    UNUSED_PARAMETER(context);
    return Status::OK();
  }
  Status ComputeAsync(OpKernelContext* context, DoneCallback done) const {
    UNUSED_PARAMETER(context);
    UNUSED_PARAMETER(done);
    return Status::OK();
  }
};

TEST(SessionStateTest, AddGetKernelTest) {
  ExecutionProviders execution_providers;
  SessionState s{execution_providers};

  onnxruntime::Model model("graph_1");
  auto& graph = model.MainGraph();
  std::vector<onnxruntime::NodeArg*> inputs;
  std::vector<onnxruntime::NodeArg*> outputs;
  TypeProto output_type;
  output_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
  output_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  onnxruntime::NodeArg output_arg("node_1_out_1", &output_type);
  outputs.push_back(&output_arg);
  onnxruntime::Node* p_node = graph.AddNode("node_1", "Variable", "node 1.", inputs, outputs);

  KernelDef kernel_def;
  CPUExecutionProvider execution_provider{CPUExecutionProviderInfo{"CPUExecutionProvider"}};

  OpKernelInfo p_info(*p_node, kernel_def, execution_provider, s);
  unique_ptr<TestOpKernel> p_kernel;
  p_kernel.reset(new TestOpKernel(p_info));
  size_t orig_num_outputs = p_kernel->Node().OutputDefs().size();
  std::cout << "node_idx: " << p_node->Index() << std::endl;

  s.SetGraph(graph);
  s.AddKernel(p_node->Index(), std::move(p_kernel));
  auto test_kernel = s.GetKernel(p_node->Index());
  std::cout << "orig: " << orig_num_outputs << " new: " << test_kernel->Node().OutputDefs().size() << std::endl;
  EXPECT_EQ(orig_num_outputs, test_kernel->Node().OutputDefs().size());
}
}  // namespace test
}  // namespace onnxruntime
