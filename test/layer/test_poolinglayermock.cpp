#include "./layer/PoolingLayer.h"
#include "./tensor/tensor.h"
#include "gtest/gtest.h"

TEST(PoolingLayerMockTest, configure_success_simple_pooling) {
  PoolingLayerInfo pool_info;
  PoolingLayerMock layer(90, pool_info);
  Shape input_shape({1, 3, 32, 32});
  Shape output_shape_ref;
  Shape expected_output_shape({1, 3, 31, 31});

  ASSERT_NO_THROW(layer.configure(input_shape, output_shape_ref));
  EXPECT_EQ(output_shape_ref.dimensions, expected_output_shape.dimensions);
  EXPECT_EQ(layer.get_output_shape().dimensions,
            expected_output_shape.dimensions);
}

TEST(PoolingLayerMockTest, configure_success_pooling_with_stride_and_pad) {
  PoolingLayerInfo pool_info;
  pool_info.pool_size_x = 3;
  pool_info.pool_size_y = 3;
  pool_info.stride_x = 2;
  pool_info.stride_y = 2;
  pool_info.pad_x = 1;
  pool_info.pad_y = 1;
  PoolingLayerMock layer(91, pool_info);
  Shape input_shape({1, 3, 32, 32});
  Shape output_shape_ref;
  Shape expected_output_shape({1, 3, 16, 16});

  ASSERT_NO_THROW(layer.configure(input_shape, output_shape_ref));
  EXPECT_EQ(output_shape_ref.dimensions, expected_output_shape.dimensions);
}

TEST(PoolingLayerMockTest, configure_fail_input_rank_not_4) {
  PoolingLayerInfo pool_info;
  PoolingLayerMock layer(92, pool_info);
  Shape input_shape({1, 32, 32});
  Shape output_shape_ref;

  EXPECT_THROW(layer.configure(input_shape, output_shape_ref),
               std::runtime_error);
}

TEST(PoolingLayerMockTest, exec_before_configure_fail) {
  PoolingLayerInfo pool_info;
  PoolingLayerMock layer(93, pool_info);
  Tensor<double> input(Shape({1, 1, 1, 1}));
  Tensor<double> output(Shape({1, 1, 1, 1}));

  EXPECT_THROW(layer.exec(input, output), std::runtime_error);
}

TEST(PoolingLayerMockTest, get_output_shape_before_configure_fail) {
  PoolingLayerInfo pool_info;
  PoolingLayerMock layer(94, pool_info);

  EXPECT_THROW(layer.get_output_shape(), std::logic_error);
}

TEST(PoolingLayerMockTest, exec_success_after_configure) {
  PoolingLayerInfo pool_info;
  PoolingLayerMock layer(95, pool_info);
  Shape input_s({1, 3, 4, 4}), out_ref;

  layer.configure(input_s, out_ref);
  Tensor<double> t_in(input_s);
  Tensor<double> t_out(out_ref);

  ASSERT_NO_THROW(layer.exec(t_in, t_out));
}

TEST(PoolingLayerMockTest, get_type_name_returns_correct_name) {
  PoolingLayerInfo pool_info;
  PoolingLayerMock layer(96, pool_info);

  EXPECT_EQ(layer.get_type_name(), "PoolingLayerMock");
}