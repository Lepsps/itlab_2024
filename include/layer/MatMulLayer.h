#ifndef MATMUL_LAYER_H
#define MATMUL_LAYER_H

#include <string>

#include "layer/layer.h"
#include "tensor/tensor.h"

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
  MatMulLayerMock(int id, const MatMulInfo& info);

  void configure(const Shape& input_x_shape, const Shape& input_y_shape,
                 Shape& output_shape_ref);

  void exec(const Tensor<double>& input_x, Tensor<double>& output) override;

  Shape get_output_shape() override;

  std::string get_type_name() const override;
};

#endif