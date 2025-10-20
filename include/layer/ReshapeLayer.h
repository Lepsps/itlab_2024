#ifndef RESHAPE_LAYER_H
#define RESHAPE_LAYER_H

#include <string>

#include "layer/layer.h"
#include "tensor/tensor.h"

class ReshapeLayerMock : public Layer {
 private:
  Shape input_shape_config_;
  Shape target_output_shape_config_;
  bool configured_ = false;

 public:
  explicit ReshapeLayerMock(int id);

  void configure(const Shape& input_shape, const Shape& target_output_shape,
                 Shape& output_shape_ref);

  void exec(const Tensor<double>& input, Tensor<double>& output) override;

  Shape get_output_shape() override;

  std::string get_type_name() const override;
};

#endif