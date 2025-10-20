#include "./layer/ConvLayer.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

ConvolutionLayerMock::ConvolutionLayerMock(int id) { setID(id); }

void ConvolutionLayerMock::configure(const Shape& input_s,
                                     const Shape& weights_s,
                                     const Shape* biases_s, Shape& output_s_ref,
                                     const ConvPadStrideInfo& info) {
  input_shape_config_ = input_s;
  weights_shape_config_ = weights_s;
  conv_info_ = info;

  has_biases_ = static_cast<bool>(biases_s);

  if (input_s.get_rank() < 3) {
    throw std::runtime_error(
        "ConvMockSimp: Input rank must be at least 3 (W, H, C).");
  }
  if (weights_s.get_rank() != 4) {
    throw std::runtime_error(
        "ConvMockSimp: Weights rank must be 4 (KW, KH, IC, OC).");
  }

  size_t c_in, h_in, w_in;
  if (input_s.get_rank() == 4) {
    c_in = input_s.dimensions[1];
    h_in = input_s.dimensions[2];
    w_in = input_s.dimensions[3];
  } else {
    w_in = input_s.dimensions[0];
    h_in = input_s.dimensions[1];
    c_in = input_s.dimensions[2];
  }

  size_t kw = weights_s.dimensions[0];
  size_t kh = weights_s.dimensions[1];
  size_t ic_w = weights_s.dimensions[2];
  size_t oc_w = weights_s.dimensions[3];

  if (c_in != ic_w) {
    throw std::runtime_error(
        "ConvMockSimp: Input channels mismatch with weights input channels.");
  }

  if (has_biases_) {
    biases_shape_config_ = *biases_s;
    if (biases_shape_config_.get_rank() != 1 ||
        biases_shape_config_.dimensions[0] != oc_w) {
      throw std::runtime_error(
          "ConvMockSimp: Biases must be 1D and size must match output "
          "channels");
    }
  }

  size_t effective_kernel_w = kw;
  size_t effective_kernel_h = kh;

  if (h_in + 2 * conv_info_.pad_y < effective_kernel_h ||
      w_in + 2 * conv_info_.pad_x < effective_kernel_w) {
    throw std::runtime_error(
        "ConvMockSimp: Kernel size is larger than padded input dimensions.");
  }

  size_t w_out = ((w_in + 2 * conv_info_.pad_x - effective_kernel_w) /
                  conv_info_.stride_x) +
                 1;
  size_t h_out = ((h_in + 2 * conv_info_.pad_y - effective_kernel_h) /
                  conv_info_.stride_y) +
                 1;

  std::vector<size_t> output_dims = {w_out, h_out, oc_w};
  if (input_s.get_rank() > 3) {
    output_dims.push_back(input_s.dimensions[0]);
  }

  output_shape_computed_ = Shape(output_dims);
  output_s_ref = output_shape_computed_;
  configured_ = true;
}

void ConvolutionLayerMock::exec(const Tensor<double>& input,
                                Tensor<double>& output) {
  if (!configured_) {
    throw std::runtime_error("ConvolutionLayerMock: Not yet implemented.");
  }
  if (input.shape.dimensions != input_shape_config_.dimensions) {
    throw std::runtime_error(
        "ConvolutionLayerMock: Input shape mismatch with configured shape.");
  }
  if (output.shape.dimensions != output_shape_computed_.dimensions) {
    throw std::runtime_error(
        "ConvolutionLayerMock: Output shape mismatch with computed shape.");
  }

  double fill_value = static_cast<double>(getID()) + 0.5;
  if (has_biases_) {
    fill_value += 0.01;
  }
  std::fill(output.data.begin(), output.data.end(), fill_value);
}

Shape ConvolutionLayerMock::get_output_shape() {
  if (!configured_) {
    throw std::logic_error("ConvLayer: Not yet implemented");
  }
  return output_shape_computed_;
}

std::string ConvolutionLayerMock::get_type_name() const {
  return "ConvolutionLayerMock";
}