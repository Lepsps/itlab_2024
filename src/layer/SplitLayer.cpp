#ifndef ACL_SPLIT_LAYER_MOCK_H
#define ACL_SPLIT_LAYER_MOCK_H

#include <cstddef>
#include <stdexcept>
#include <string>

#include "./layer/layer.h"
#include "./tensor/tensor.h"

class SplitLayerMock : public Layer {
 private:
  Shape input_shape_config_;
  std::vector<Shape> output_shapes_computed_;
  unsigned int split_axis_;
  unsigned int num_splits_;

  bool configured_ = false;

 public:
  SplitLayerMock(int id) { setID(id); }

  void configure(const Shape& input_shape, unsigned int axis,
                 unsigned int num_splits, Shape& first_output_shape_ref) {
    if (num_splits == 0) {
      throw std::runtime_error("SplitMock: Number of splits cannot be zero.");
    }
    if (axis >= input_shape.get_rank()) {
      throw std::runtime_error("SplitMock: Split axis is out of bounds.");
    }
    if (input_shape.dimensions[axis] % num_splits != 0) {
      throw std::runtime_error(
          "SplitMock: Dimension size along split axis must be divisible by "
          "num_splits.");
    }

    input_shape_config_ = input_shape;
    split_axis_ = axis;
    num_splits_ = num_splits;
    output_shapes_computed_.clear();

    size_t split_dim_size = input_shape.dimensions[axis] / num_splits;
    for (unsigned int i = 0; i < num_splits; ++i) {
      std::vector<size_t> part_dims = input_shape.dimensions;
      part_dims[axis] = split_dim_size;
      output_shapes_computed_.push_back(Shape(part_dims));
    }

    if (!output_shapes_computed_.empty()) {
      first_output_shape_ref = output_shapes_computed_[0];
    }

    configured_ = true;
  }

  void exec(const Tensor<double>& input, Tensor<double>& output) override {
    if (!configured_) {
      throw std::logic_error("Split Layer: Not yet implemented");
    }
    if (input.shape.dimensions != input_shape_config_.dimensions) {
      throw std::runtime_error(
          "SplitLayerMock: Input shape mismatch with configured shape.");
    }
    if (output_shapes_computed_.empty() ||
        output.shape.dimensions != output_shapes_computed_[0].dimensions) {
      throw std::runtime_error(
          "SplitLayerMock: Output shape must match the shape of the first "
          "split part.");
    }

    std::fill(output.data.begin(), output.data.end(),
              static_cast<double>(getID()) + 0.7);
  }

  Shape get_output_shape() override {
    if (!configured_ || output_shapes_computed_.empty()) {
      throw std::logic_error("SplitLayerMock:  Not yet implemented.");
    }
    return output_shapes_computed_[0];
  }

  const std::vector<Shape>& get_all_split_output_shapes() const {
    if (!configured_) {
      throw std::logic_error("Split Layer: Not yet implemented");
    }
    return output_shapes_computed_;
  }

  std::string get_type_name() const override { return "SplitLayerMock"; }
};

#endif