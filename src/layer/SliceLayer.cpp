#ifndef ACL_SLICE_LAYER_MOCK_H
#define ACL_SLICE_LAYER_MOCK_H

#include <cstddef>
#include <stdexcept>
#include <string>

#include "./layer/layer.h"
#include "./tensor/tensor.h"

class SliceLayerMock : public Layer {
 private:
  Shape input_shape_config_;
  Shape output_shape_computed_;
  std::vector<int> slice_starts_;
  std::vector<int> slice_sizes_;
  bool configured_ = false;

 public:
  SliceLayerMock(int id) { setID(id); }

  void configure(const Shape& input_shape, const std::vector<int>& starts,
                 const std::vector<int>& sizes, Shape& output_shape_ref) {
    size_t rank = input_shape.get_rank();
    if (starts.size() != rank || sizes.size() != rank) {
      throw std::runtime_error(
          "SliceMock: 'starts' and 'sizes' vectors must match input rank.");
    }

    std::vector<size_t> output_dims(rank);
    for (size_t i = 0; i < rank; ++i) {
      if (starts[i] < 0 ||
          static_cast<size_t>(starts[i]) >= input_shape.dimensions[i]) {
        throw std::runtime_error(
            "SliceMock: Start coordinate out of bounds for axis ");
      }

      size_t current_size;
      if (sizes[i] == -1) {
        current_size =
            input_shape.dimensions[i] - static_cast<size_t>(starts[i]);
      } else if (sizes[i] < 0) {
        throw std::runtime_error(
            "SliceMock: Size cannot be negative (unless -1 for 'to end') for "
            "axis ");
      } else {
        current_size = static_cast<size_t>(sizes[i]);
      }

      if (static_cast<size_t>(starts[i]) + current_size >
          input_shape.dimensions[i]) {
        throw std::runtime_error(
            "SliceMock: Slice (start + size) exceeds dimension for axis ");
      }
      if (current_size == 0) {
        throw std::runtime_error(
            "SliceMock: Slice size cannot be zero for axis ");
      }
      output_dims[i] = current_size;
    }
    input_shape_config_ = input_shape;
    slice_starts_ = starts;
    slice_sizes_ = sizes;
    output_shape_computed_ = Shape(output_dims);
    output_shape_ref = output_shape_computed_;

    configured_ = true;
  }

  void exec(const Tensor<double>& input, Tensor<double>& output) override {
    if (!configured_) {
      throw std::logic_error("Slice Layer: Not yet implemented");
    }
    if (input.shape.dimensions != input_shape_config_.dimensions) {
      throw std::runtime_error(
          "SliceLayerMock: Input shape mismatch with configured shape.");
    }
    if (output.shape.dimensions != output_shape_computed_.dimensions) {
      throw std::runtime_error(
          "SliceLayerMock: Output shape mismatch with computed shape.");
    }

    std::fill(output.data.begin(), output.data.end(),
              static_cast<double>(getID()) + 0.8);
  }

  Shape get_output_shape() override {
    if (!configured_) {
      throw std::logic_error("Slice Layer: Not yet implemented");
    }
    return output_shape_computed_;
  }

  std::string get_type_name() const override { return "SliceLayerMock"; }
};

#endif