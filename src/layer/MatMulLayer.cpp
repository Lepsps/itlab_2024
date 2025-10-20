#ifndef ACL_MATMUL_LAYER_MOCK_H
#define ACL_MATMUL_LAYER_MOCK_H

#include <cstddef>
#include <stdexcept>
#include <string>

#include "./layer/layer.h"
#include "./tensor/tensor.h"

struct MatMulInfo {
  bool transpose_x{false};
  bool transpose_y{false};
};

class MatMulLayerMock : public Layer {
 private:
  MatMulInfo matmul_info_;
  Shape input_x_shape_;
  Shape input_y_shape_;
  Shape output_shape_;
  bool configured_ = false;

 public:
  MatMulLayerMock(int id, const MatMulInfo& info) : matmul_info_(info) {
    setID(id);
  }

  void configure(const Shape& input_x_shape, const Shape& input_y_shape,
                 Shape& output_shape_ref) {
    size_t m;
    size_t k_x;
    size_t k_y;
    size_t n;

    if (input_x_shape.get_rank() != 2 || input_y_shape.get_rank() != 2) {
      throw std::runtime_error(
          "MatMulMock: Inputs must be 2D tensors for this mock.");
    }

    m = matmul_info_.transpose_x ? input_x_shape.dimensions[1]
                                 : input_x_shape.dimensions[0];
    k_x = matmul_info_.transpose_x ? input_x_shape.dimensions[0]
                                   : input_x_shape.dimensions[1];

    k_y = matmul_info_.transpose_y ? input_y_shape.dimensions[1]
                                   : input_y_shape.dimensions[0];
    n = matmul_info_.transpose_y ? input_y_shape.dimensions[0]
                                 : input_y_shape.dimensions[1];

    if (k_x != k_y) {
      throw std::runtime_error(
          "MatMulMock: Inner dimensions do not match for matrix "
          "multiplication ");
    }

    input_x_shape_ = input_x_shape;
    input_y_shape_ = input_y_shape;
    output_shape_ = Shape({m, n});
    output_shape_ref = output_shape_;

    configured_ = true;
  }

  void exec(const Tensor<double>& input_x, Tensor<double>& output) override {
    if (!configured_) {
      throw std::runtime_error("MatMulLayerMock: Not yet implemented");
    }
    if (input_x.shape.dimensions != input_x_shape_.dimensions) {
      throw std::runtime_error(
          "MatMulLayerMock: Input X shape mismatch in exec.");
    }
    if (output.shape.dimensions != output_shape_.dimensions ||
        output.shape.total_elements != output_shape_.total_elements) {
      throw std::runtime_error(
          "MatMulLayerMock: Output shape mismatch in exec.");
    }
    std::fill(output.data.begin(), output.data.end(),
              static_cast<double>(getID()) + 0.1);
  }

  Shape get_output_shape() override {
    if (!configured_) {
      throw std::logic_error("MatMul: Not yet implemented");
    }
    return output_shape_;
  }

  std::string get_type_name() const override { return "MatMulLayerMock"; }
};

#endif