#ifndef SPLIT_LAYER_H
#define SPLIT_LAYER_H

#include <string>
#include <vector>

#include "layer/layer.h"
#include "tensor/tensor.h"

class SplitLayerMock : public Layer {
 private:
  Shape input_shape_config_;
  std::vector<Shape> output_shapes_computed_;
  unsigned int split_axis_;
  unsigned int num_splits_;
  bool configured_ = false;

 public:
  explicit SplitLayerMock(int id);

  void configure(const Shape& input_shape, unsigned int axis,
                 unsigned int num_splits, Shape& first_output_shape_ref);

  void exec(const Tensor<double>& input, Tensor<double>& output) override;

  Shape get_output_shape() override;

  const std::vector<Shape>& get_all_split_output_shapes() const;

  std::string get_type_name() const override;
};

#endif