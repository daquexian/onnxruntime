#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "ml_common.h"

namespace Lotus {
namespace ML {
template <typename T>
class TreeEnsembleRegressor final : public OpKernel {
 public:
  explicit TreeEnsembleRegressor(const OpKernelInfo& info);
  Common::Status Compute(OpKernelContext* context) const override;

 private:
  Common::Status ProcessTreeNode(std::unordered_map<int64_t, float>& classes, int64_t treeindex, const T* Xdata, int64_t feature_base) const;

  std::vector<int64_t> nodes_treeids_;
  std::vector<int64_t> nodes_nodeids_;
  std::vector<int64_t> nodes_featureids_;
  std::vector<float> nodes_values_;
  std::vector<float> nodes_hitrates_;
  std::vector<int64_t> nodes_modes_;
  std::vector<int64_t> nodes_truenodeids_;
  std::vector<int64_t> nodes_falsenodeids_;
  std::vector<int64_t> missing_tracks_true_;

  std::vector<int64_t> target_nodeids_;
  std::vector<int64_t> target_treeids_;
  std::vector<int64_t> target_ids_;
  std::vector<float> target_weights_;

  std::vector<float> base_values_;
  int64_t n_targets_;
  Lotus::ML::POST_EVAL_TRANSFORM transform_;
  Lotus::ML::AGGREGATE_FUNCTION aggregate_function_;
  std::vector<std::tuple<int64_t, int64_t, int64_t, float>> leafnode_data_;
  std::unordered_map<int64_t, size_t> leafdata_map_;
  std::vector<int64_t> roots_;
  int64_t offset_;
  int64_t max_tree_depth_;
  const int64_t four_billion_ = 4000000000L;
};
}  // namespace ML
}  // namespace Lotus
