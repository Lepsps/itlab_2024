#ifndef SLICE_LAYER_H
#define SLICE_LAYER_H

#include <string>
#include <vector>

#include "layer/layer.h"
#include "tensor/tensor.h"

class SliceLayerMock : public Layer {
 private:
  Shape input_shape_config_;
  Shape output_shape_computed_;
  std::vector<int> slice_starts_;
  std::vector<int> slice_sizes_;
  bool configured_ = false;

 public:
  explicit SliceLayerMock(int id);

  void configure(const Shape& input_shape, const std::vector<int>& starts,
                 const std::vector<int>& sizes, Shape& output_shape_ref);

  void exec(const Tensor<double>& input, Tensor<double>& output) override;

  Shape get_output_shape() override;

  std::string get_type_name() const override;
};

#endif