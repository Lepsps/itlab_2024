#include <stdexcept>

#include "./layer/SplitLayer.h"
#include "./tensor/tensor.h"
#include "gtest/gtest.h"

TEST(SplitLayerMockTest, configure_success_simple_split) {
  SplitLayerMock layer(1);
  Shape input_shape({10, 20, 30});
  unsigned int axis = 1;
  unsigned int num_splits = 2;
  Shape first_out_shape_ref;
  Shape expected_first_out_shape({10, 10, 30});

  ASSERT_NO_THROW(
      layer.configure(input_shape, axis, num_splits, first_out_shape_ref));
  EXPECT_EQ(first_out_shape_ref.dimensions,
            expected_first_out_shape.dimensions);
  EXPECT_EQ(layer.get_output_shape().dimensions,
            expected_first_out_shape.dimensions);
  const auto& all_shapes = layer.get_all_split_output_shapes();
  ASSERT_EQ(all_shapes.size(), num_splits);
  for (const auto& shape : all_shapes) {
    EXPECT_EQ(shape.dimensions, expected_first_out_shape.dimensions);
  }
}

TEST(SplitLayerMockTest, configure_fail_zero_splits) {
  SplitLayerMock layer(2);
  Shape input_shape({10, 20, 30});
  Shape first_out_shape_ref;

  EXPECT_THROW(layer.configure(input_shape, 1, 0, first_out_shape_ref),
               std::runtime_error);
}

TEST(SplitLayerMockTest, configure_fail_axis_out_of_bounds) {
  SplitLayerMock layer(3);
  Shape input_shape({10, 20, 30});
  Shape first_out_shape_ref;

  EXPECT_THROW(layer.configure(input_shape, 3, 2, first_out_shape_ref),
               std::runtime_error);
}

TEST(SplitLayerMockTest, configure_fail_not_divisible) {
  SplitLayerMock layer(4);
  Shape input_shape({10, 21, 30});
  Shape first_out_shape_ref;

  EXPECT_THROW(layer.configure(input_shape, 1, 2, first_out_shape_ref),
               std::runtime_error);
}

TEST(SplitLayerMockTest, exec_before_configure_fail) {
  SplitLayerMock layer(5);
  Tensor<double> input_tensor(Shape({10, 20, 30}));
  Tensor<double> output_tensor(Shape({10, 10, 30}));

  EXPECT_THROW(layer.exec(input_tensor, output_tensor), std::logic_error);
}

TEST(SplitLayerMockTest, get_output_shape_before_configure_fail) {
  SplitLayerMock layer(6);

  EXPECT_THROW(layer.get_output_shape(), std::logic_error);
}

TEST(SplitLayerMockTest, exec_success_after_configure) {
  SplitLayerMock layer(7);
  Shape input_shape({10, 20, 30});
  Shape first_out_shape_ref;

  layer.configure(input_shape, 1, 2, first_out_shape_ref);

  Tensor<double> input_tensor(input_shape);
  Tensor<double> output_tensor(first_out_shape_ref);

  ASSERT_NO_THROW(layer.exec(input_tensor, output_tensor));
}

TEST(SplitLayerMockTest, exec_fail_input_shape_mismatch) {
  SplitLayerMock layer(8);
  Shape config_input_shape({10, 20, 30});
  Shape first_out_shape_ref;

  layer.configure(config_input_shape, 1, 2, first_out_shape_ref);

  Tensor<double> wrong_input_tensor(Shape({5, 5, 5}));
  Tensor<double> output_tensor(first_out_shape_ref);
  EXPECT_THROW(layer.exec(wrong_input_tensor, output_tensor),
               std::runtime_error);
}

TEST(SplitLayerMockTest, exec_fail_output_shape_mismatch) {
  SplitLayerMock layer(9);
  Shape input_shape({10, 20, 30});
  Shape first_out_shape_ref;

  layer.configure(input_shape, 1, 2, first_out_shape_ref);

  Tensor<double> input_tensor(input_shape);
  Tensor<double> wrong_output_tensor(Shape({5, 5, 5}));
  EXPECT_THROW(layer.exec(input_tensor, wrong_output_tensor),
               std::runtime_error);
}

TEST(SplitLayerMockTest, get_type_name_returns_correct_name) {
  SplitLayerMock layer(10);

  EXPECT_EQ(layer.get_type_name(), "SplitLayerMock");
}