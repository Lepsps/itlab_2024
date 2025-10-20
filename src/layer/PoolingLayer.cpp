#ifndef ACL_POOLING_LAYER_MOCK_H
#define ACL_POOLING_LAYER_MOCK_H

#include <cstddef>
#include <stdexcept>
#include <string>

#include "./layer/layer.h"
#include "./tensor/tensor.h"

enum class PoolingType { kMax, kAvg, kL2 };

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
  bool configured_ = false;

 public:
  PoolingLayerMock(int id, const PoolingLayerInfo& info) : pool_info_(info) {
    setID(id);
  }

  void configure(const Shape& input_shape, Shape& output_shape_ref) {
    if (input_shape.get_rank() != 4) {
      throw std::runtime_error(
          "PoolingMock: Input must be a 4D tensor (e.g., NCHW or NHWC) for "
          "this mock.");
    }
    size_t h_in_idx = input_shape.get_rank() - 2;
    size_t w_in_idx = input_shape.get_rank() - 1;

    size_t h_in = input_shape.dimensions[h_in_idx];
    size_t w_in = input_shape.dimensions[w_in_idx];

    size_t h_out = ((h_in + 2 * pool_info_.pad_y - pool_info_.pool_size_y) /
                    pool_info_.stride_y) +
                   1;
    size_t w_out = ((w_in + 2 * pool_info_.pad_x - pool_info_.pool_size_x) /
                    pool_info_.stride_x) +
                   1;

    input_shape_ = input_shape;
    output_shape_ = input_shape;

    output_shape_.dimensions[h_in_idx] = h_out;
    output_shape_.dimensions[w_in_idx] = w_out;

    output_shape_ = Shape(output_shape_.dimensions);

    output_shape_ref = output_shape_;
    configured_ = true;
  }

  void exec(const Tensor<double>& input, Tensor<double>& output) override {
    if (!configured_) {
      throw std::runtime_error("Pool Layer: Not yet implemented");
    }
    if (input.shape.dimensions != input_shape_.dimensions) {
      throw std::runtime_error(
          "PoolingLayerMock: Input shape mismatch in exec.");
    }
    if (output.shape.dimensions != output_shape_.dimensions ||
        output.shape.total_elements != output_shape_.total_elements) {
      throw std::runtime_error(
          "PoolingLayerMock: Output shape mismatch in exec.");
    }

    double fill_value = 0.0;
    switch (pool_info_.pool_type) {
      case PoolingType::kMax:
        fill_value = 1.0;
        break;
      case PoolingType::kAvg:
        fill_value = 0.5;
        break;
      case PoolingType::kL2:
        fill_value = 0.7;
        break;
    }
    std::fill(output.data.begin(), output.data.end(),
              static_cast<double>(getID()) + fill_value + 0.2);
  }

  Shape get_output_shape() override {
    if (!configured_) {
      throw std::logic_error("Pool Layer: Not yet implemented");
    }
    return output_shape_;
  }

  std::string get_type_name() const override { return "PoolingLayerMock"; }
};

#endif