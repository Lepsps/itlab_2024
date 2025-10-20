#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include <string>
#include <vector>

#include "layer/layer.h"
#include "tensor/tensor.h"

struct ConvPadStrideInfo {
  unsigned int stride_x{1};
  unsigned int stride_y{1};
  unsigned int pad_x{0};
  unsigned int pad_y{0};

  ConvPadStrideInfo(unsigned int sx = 1, unsigned int sy = 1,
                    unsigned int px = 0, unsigned int py = 0)
      : stride_x(sx), stride_y(sy), pad_x(px), pad_y(py) {}
};

class ConvolutionLayerMock : public Layer {
 private:
  ConvPadStrideInfo conv_info_;
  Shape input_shape_config_;
  Shape weights_shape_config_;
  Shape biases_shape_config_;
  Shape output_shape_computed_;
  bool has_biases_ = false;
  bool configured_ = false;

 public:
  explicit ConvolutionLayerMock(int id);

  void configure(const Shape& input_s, const Shape& weights_s,
                 const Shape* biases_s, Shape& output_s_ref,
                 const ConvPadStrideInfo& info);

  void exec(const Tensor<double>& input, Tensor<double>& output) override;

  Shape get_output_shape() override;

  std::string get_type_name() const override;
};

#endif