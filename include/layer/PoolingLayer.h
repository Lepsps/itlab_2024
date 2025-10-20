#ifndef POOLING_LAYER_H
#define POOLING_LAYER_H

#include <cstdint>
#include <string>

#include "layer/layer.h"
#include "tensor/tensor.h"

enum class PoolingType : std::uint8_t { kMax, kAvg, kL2 };

struct PoolingLayerInfo {
  PoolingType pool_type{PoolingType::kMax};
  int pool_size_x{2};
  int pool_size_y{2};
  int stride_x{1};
  int stride_y{1};
  int pad_x{0};
  int pad_y{0};
  bool exclude_padding{true};
};

class PoolingLayerMock : public Layer {
 private:
  PoolingLayerInfo pool_info_;
  Shape input_shape_;
  Shape output_shape_;
  size_t h_in_idx_ = 0;
  size_t w_in_idx_ = 0;
  bool configured_ = false;

 public:
  PoolingLayerMock(int id, const PoolingLayerInfo& info);

  void configure(const Shape& input_shape, Shape& output_shape_ref);

  void exec(const Tensor<double>& input, Tensor<double>& output) override;

  Shape get_output_shape() override;

  std::string get_type_name() const override;
};

#endif