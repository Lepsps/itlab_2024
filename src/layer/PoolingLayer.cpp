#include "./layer/PoolingLayer.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

PoolingLayerMock::PoolingLayerMock(int id, const PoolingLayerInfo& info)
    : pool_info_(info) {
  setID(id);
}

void PoolingLayerMock::configure(const Shape& input_shape,
                                 Shape& output_shape_ref) {
  if (input_shape.get_rank() != 4) {
    throw std::runtime_error(
        "PoolingMock: Input must be a 4D tensor (e.g., NCHW or NHWC) for this "
        "mock.");
  }
  h_in_idx_ = input_shape.get_rank() - 2;
  w_in_idx_ = input_shape.get_rank() - 1;

  size_t h_in = input_shape.dimensions[h_in_idx_];
  size_t w_in = input_shape.dimensions[w_in_idx_];

  size_t h_out = ((h_in + 2 * pool_info_.pad_y - pool_info_.pool_size_y) /
                  pool_info_.stride_y) +
                 1;
  size_t w_out = ((w_in + 2 * pool_info_.pad_x - pool_info_.pool_size_x) /
                  pool_info_.stride_x) +
                 1;

  input_shape_ = input_shape;

  std::vector<size_t> output_dims = input_shape.dimensions;
  output_dims[h_in_idx_] = h_out;
  output_dims[w_in_idx_] = w_out;
  output_shape_ = Shape(output_dims);

  output_shape_ref = output_shape_;
  configured_ = true;
}

void PoolingLayerMock::exec(const Tensor<double>& input,
                            Tensor<double>& output) {
  if (!configured_) {
    throw std::runtime_error("Pool Layer: Not yet implemented");
  }
  if (input.shape.dimensions != input_shape_.dimensions) {
    throw std::runtime_error("PoolingLayerMock: Input shape mismatch in exec.");
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

Shape PoolingLayerMock::get_output_shape() {
  if (!configured_) {
    throw std::logic_error("Pool Layer: Not yet implemented");
  }
  return output_shape_;
}

std::string PoolingLayerMock::get_type_name() const {
  return "PoolingLayerMock";
}