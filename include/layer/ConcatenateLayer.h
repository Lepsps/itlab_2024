#ifndef CONCATENATE_LAYER_H
#define CONCATENATE_LAYER_H

#include <string>
#include <vector>

#include "layer/layer.h"
#include "tensor/tensor.h"

class ConcatenateLayerMock : public Layer {
 private:
  std::vector<Shape> input_shapes_config_;
  Shape output_shape_computed_;
  unsigned int concatenation_axis_;
  bool configured_ = false;

 public:
  explicit ConcatenateLayerMock(int id);

  void configure(const std::vector<Shape>& inputs_shapes, unsigned int axis,
                 Shape& output_shape_ref);

  void exec(const Tensor<double>& input, Tensor<double>& output) override;

  Shape get_output_shape() override;

  std::string get_type_name() const override;
};

#endif