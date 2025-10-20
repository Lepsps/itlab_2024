#ifndef ACL_RESHAPE_LAYER_MOCK_H
#define ACL_RESHAPE_LAYER_MOCK_H

#include <stdexcept>
#include <string>

#include "./layer/layer.h"
#include "./tensor/tensor.h"

class ReshapeLayerMock : public Layer {
 private:
  Shape input_shape_config_;
  Shape target_output_shape_config_;
  bool configured_ = false;

 public:
  ReshapeLayerMock(int id) { setID(id); }

  void configure(const Shape& input_shape, const Shape& target_output_shape,
                 Shape& output_shape_ref) {
    if (input_shape.total_elements != target_output_shape.total_elements) {
      throw std::runtime_error(
          "ReshapeMock: Total number of elements must remain the same for "
          "reshape.");
    }
    for (size_t dim_size : target_output_shape.dimensions) {
      if (dim_size == 0 && target_output_shape.total_elements != 0) {
        throw std::runtime_error(
            "ReshapeMock: Target output shape dimension cannot be zero if "
            "total elements is not zero.");
      }
    }

    input_shape_config_ = input_shape;
    target_output_shape_config_ = target_output_shape;
    output_shape_ref = target_output_shape_config_;

    configured_ = true;
  }

  void exec(const Tensor<double>& input, Tensor<double>& output) override {
    if (!configured_) {
      throw std::runtime_error("Reshape Layer: Not yet implemented.");
    }
    if (input.shape.dimensions != input_shape_config_.dimensions) {
      throw std::runtime_error(
          "ReshapeLayerMock: Input shape mismatch with configured shape.");
    }
    if (output.shape.dimensions != target_output_shape_config_.dimensions) {
      throw std::runtime_error(
          "ReshapeLayerMock: Output shape mismatch with target shape.");
    }
    if (input.data.size() != output.data.size()) {
      throw std::runtime_error(
          "ReshapeLayerMock: Input and output data buffer sizes mismatch.");
    }
    std::fill(output.data.begin(), output.data.end(),
              static_cast<double>(getID()) + 0.9);
  }

  Shape get_output_shape() override {
    if (!configured_) {
      throw std::logic_error("Reshape Layer: Not yet implemented");
    }
    return target_output_shape_config_;
  }

  std::string get_type_name() const override { return "ReshapeLayerMock"; }
};

#endif