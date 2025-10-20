#ifndef ELEMENTWISE_LAYER_H
#define ELEMENTWISE_LAYER_H

#include <cstdint>
#include <string>

#include "layer/layer.h"
#include "tensor/tensor.h"

enum class ElementwiseOp : std::uint8_t {
  kAdd,
  kSub,
  kMul,
  kDiv,
  kMax,
  kMin,
  kSquaredDiff
};

class ElementwiseLayerMock : public Layer {
 private:
  ElementwiseOp op_type_;
  Shape common_shape_;
  bool configured_ = false;

 public:
  ElementwiseLayerMock(int id, ElementwiseOp op);

  void configure(const Shape& input1_shape, const Shape& input2_shape,
                 Shape& output_shape_ref);

  void exec(const Tensor<double>& input, Tensor<double>& output) override;

  Shape get_output_shape() override;

  std::string get_type_name() const override;
};

#endif