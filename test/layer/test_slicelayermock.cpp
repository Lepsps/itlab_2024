#include "./layer/SliceLayer.h"
#include "./tensor/tensor.h"
#include "gtest/gtest.h"

TEST(SliceLayerMockTest, configure_success_simple_slice) {
  SliceLayerMock layer(30);
  Shape input_shape({10, 20, 5});
  std::vector<int> starts = {1, 2, 0};
  std::vector<int> sizes = {5, 10, 3};
  Shape output_shape_ref;
  Shape expected_output_shape({5, 10, 3});

  ASSERT_NO_THROW(
      layer.configure(input_shape, starts, sizes, output_shape_ref));
  EXPECT_EQ(output_shape_ref.dimensions, expected_output_shape.dimensions);
  EXPECT_EQ(layer.get_output_shape().dimensions,
            expected_output_shape.dimensions);
}

TEST(SliceLayerMockTest, configure_success_slice_to_end) {
  SliceLayerMock layer(31);
  Shape input_shape({10, 20, 5});
  std::vector<int> starts = {1, 2, 0};
  std::vector<int> sizes = {5, -1, 3};
  Shape output_shape_ref;
  Shape expected_output_shape({5, 18, 3});

  ASSERT_NO_THROW(
      layer.configure(input_shape, starts, sizes, output_shape_ref));
  EXPECT_EQ(output_shape_ref.dimensions, expected_output_shape.dimensions);
}

TEST(SliceLayerMockTest, configure_fail_starts_sizes_rank_mismatch) {
  SliceLayerMock layer(32);
  Shape input_shape({10, 20, 5});
  std::vector<int> starts = {1, 2};
  std::vector<int> sizes = {5, 10, 3};
  Shape output_shape_ref;

  EXPECT_THROW(layer.configure(input_shape, starts, sizes, output_shape_ref),
               std::runtime_error);
}

TEST(SliceLayerMockTest, configure_fail_start_out_of_bounds) {
  SliceLayerMock layer(33);
  Shape input_shape({10, 20, 5});
  std::vector<int> starts = {1, 20, 0};
  std::vector<int> sizes = {5, 1, 3};
  Shape output_shape_ref;

  EXPECT_THROW(layer.configure(input_shape, starts, sizes, output_shape_ref),
               std::runtime_error);
}

TEST(SliceLayerMockTest, configure_fail_negative_start) {
  SliceLayerMock layer(34);
  Shape input_shape({10, 20, 5});
  std::vector<int> starts = {1, -1, 0};
  std::vector<int> sizes = {5, 1, 3};
  Shape output_shape_ref;

  EXPECT_THROW(layer.configure(input_shape, starts, sizes, output_shape_ref),
               std::runtime_error);
}

TEST(SliceLayerMockTest, configure_fail_negative_size_not_minus_one) {
  SliceLayerMock layer(35);
  Shape input_shape({10, 20, 5});
  std::vector<int> starts = {1, 2, 0};
  std::vector<int> sizes = {5, -2, 3};
  Shape output_shape_ref;

  EXPECT_THROW(layer.configure(input_shape, starts, sizes, output_shape_ref),
               std::runtime_error);
}

TEST(SliceLayerMockTest, configure_fail_slice_exceeds_dimension) {
  SliceLayerMock layer(36);
  Shape input_shape({10, 20, 5});
  std::vector<int> starts = {1, 2, 0};
  std::vector<int> sizes = {5, 19, 3};
  Shape output_shape_ref;

  EXPECT_THROW(layer.configure(input_shape, starts, sizes, output_shape_ref),
               std::runtime_error);
}

TEST(SliceLayerMockTest, configure_fail_zero_size) {
  SliceLayerMock layer(37);
  Shape input_shape({10, 20, 5});
  std::vector<int> starts = {1, 2, 0};
  std::vector<int> sizes = {5, 0, 3};
  Shape output_shape_ref;

  EXPECT_THROW(layer.configure(input_shape, starts, sizes, output_shape_ref),
               std::runtime_error);
}

TEST(SliceLayerMockTest, exec_before_configure_fail) {
  SliceLayerMock layer(38);
  Tensor<double> input(Shape({1, 1, 1}));
  Tensor<double> output(Shape({1, 1, 1}));

  EXPECT_THROW(layer.exec(input, output), std::logic_error);
}

TEST(SliceLayerMockTest, get_output_shape_before_configure_fail) {
  SliceLayerMock layer(39);

  EXPECT_THROW(layer.get_output_shape(), std::logic_error);
}

TEST(SliceLayerMockTest, exec_success_after_configure) {
  SliceLayerMock layer(40);
  Shape input_shape({10, 20, 5});
  std::vector<int> starts = {1, 2, 0};
  std::vector<int> sizes = {5, 10, 3};
  Shape output_shape_ref;

  layer.configure(input_shape, starts, sizes, output_shape_ref);

  Tensor<double> input_tensor(input_shape);
  Tensor<double> output_tensor(output_shape_ref);

  ASSERT_NO_THROW(layer.exec(input_tensor, output_tensor));
}

TEST(SliceLayerMockTest, get_type_name_returns_correct_name) {
  SliceLayerMock layer(41);

  EXPECT_EQ(layer.get_type_name(), "SliceLayerMock");
}