#include "./layer/ConcatenateLayer.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "tensor/tensor.h"

ConcatenateLayerMock::ConcatenateLayerMock(int id) { setID(id); }

void ConcatenateLayerMock::configure(const std::vector<Shape>& inputs_shapes,
                                     unsigned int axis,
                                     Shape& output_shape_ref) {
  if (inputs_shapes.empty()) {
    throw std::runtime_error("ConcatMock: Input shapes list cannot be empty.");
  }

  const Shape& first_shape = inputs_shapes[0];
  if (axis >= first_shape.get_rank()) {
    throw std::runtime_error(
        "ConcatMock: Concatenation axis is out of bounds.");
  }

  size_t rank = first_shape.get_rank();
  size_t concatenated_dim_size = 0;

  for (const auto& shape : inputs_shapes) {
    if (shape.get_rank() != rank) {
      throw std::runtime_error(
          "ConcatMock: All input tensors must have the same rank.");
    }
    for (unsigned int i = 0; i < rank; ++i) {
      if (i == axis) {
        concatenated_dim_size += shape.dimensions[i];
      } else {
        if (shape.dimensions[i] != first_shape.dimensions[i]) {
          throw std::runtime_error(
              "ConcatMock: Input tensor dimensions must match along "
              "non-concatenation axes.");
        }
      }
    }
  }

  std::vector<size_t> output_dims = first_shape.dimensions;
  output_dims[axis] = concatenated_dim_size;

  input_shapes_config_ = inputs_shapes;
  concatenation_axis_ = axis;
  output_shape_computed_ = Shape(output_dims);
  output_shape_ref = output_shape_computed_;

  configured_ = true;
}

void ConcatenateLayerMock::exec(const Tensor<double>&, Tensor<double>& output) {
  if (!configured_) {
    throw std::logic_error("Concatenate: Not yet implemented");
  }
  if (output.shape.dimensions != output_shape_computed_.dimensions) {
    throw std::runtime_error(
        "ACLConcatenateLayerMock: Output shape mismatch with computed shape.");
  }
  std::fill(output.data.begin(), output.data.end(),
            static_cast<double>(getID()) + 0.6);
}

Shape ConcatenateLayerMock::get_output_shape() {
  if (!configured_) {
    throw std::logic_error("Concatenate: Not yet implemented");
  }
  return output_shape_computed_;
}

std::string ConcatenateLayerMock::get_type_name() const {
  return "ConcatenateLayerMock";
}