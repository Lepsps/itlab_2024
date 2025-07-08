#include "./layer/MatMulLayer.cpp"
#include "./tensor/tensor.h"
#include "gtest/gtest.h"

TEST(MatMulLayerMockTest, configure_success_simple_mat_mul) {
  MatMulInfo mm_info;
  MatMulLayerMock layer(70, mm_info);
  Shape shape_x({2, 3});
  Shape shape_y({3, 4});
  Shape output_shape_ref;
  Shape expected_output_shape({2, 4});

  ASSERT_NO_THROW(layer.configure(shape_x, shape_y, output_shape_ref));
  EXPECT_EQ(output_shape_ref.dimensions, expected_output_shape.dimensions);
  EXPECT_EQ(layer.get_output_shape().dimensions,
            expected_output_shape.dimensions);
}

TEST(MatMulLayerMockTest, configure_success_transpose_x) {
  MatMulInfo mm_info;
  mm_info.transpose_x = true;
  MatMulLayerMock layer(71, mm_info);
  Shape shape_x({3, 2});
  Shape shape_y({3, 4});
  Shape output_shape_ref;
  Shape expected_output_shape({2, 4});

  ASSERT_NO_THROW(layer.configure(shape_x, shape_y, output_shape_ref));
  EXPECT_EQ(output_shape_ref.dimensions, expected_output_shape.dimensions);
}

TEST(MatMulLayerMockTest, configure_success_transpose_y) {
  MatMulInfo mm_info;
  mm_info.transpose_y = true;
  MatMulLayerMock layer(72, mm_info);
  Shape shape_x({2, 3});
  Shape shape_y({4, 3});
  Shape output_shape_ref;
  Shape expected_output_shape({2, 4});

  ASSERT_NO_THROW(layer.configure(shape_x, shape_y, output_shape_ref));
  EXPECT_EQ(output_shape_ref.dimensions, expected_output_shape.dimensions);
}

TEST(MatMulLayerMockTest, configure_success_transpose_both) {
  MatMulInfo mm_info;
  mm_info.transpose_x = true;
  mm_info.transpose_y = true;
  MatMulLayerMock layer(73, mm_info);
  Shape shape_x({3, 2});
  Shape shape_y({4, 3});
  Shape output_shape_ref;
  Shape expected_output_shape({2, 4});

  ASSERT_NO_THROW(layer.configure(shape_x, shape_y, output_shape_ref));
  EXPECT_EQ(output_shape_ref.dimensions, expected_output_shape.dimensions);
}

TEST(MatMulLayerMockTest, configure_fail_input_x_not_2d) {
  MatMulInfo mm_info;
  MatMulLayerMock layer(74, mm_info);
  Shape shape_x({2, 3, 1});
  Shape shape_y({3, 4});
  Shape output_shape_ref;

  EXPECT_THROW(layer.configure(shape_x, shape_y, output_shape_ref),
               std::runtime_error);
}

TEST(MatMulLayerMockTest, configure_fail_input_y_not_2d) {
  MatMulInfo mm_info;
  MatMulLayerMock layer(75, mm_info);
  Shape shape_x({2, 3});
  Shape shape_y({3, 4, 1});
  Shape output_shape_ref;

  EXPECT_THROW(layer.configure(shape_x, shape_y, output_shape_ref),
               std::runtime_error);
}

TEST(MatMulLayerMockTest, configure_fail_dimension_mismatch) {
  MatMulInfo mm_info;
  MatMulLayerMock layer(76, mm_info);
  Shape shape_x({2, 3});
  Shape shape_y({4, 4});
  Shape output_shape_ref;

  EXPECT_THROW(layer.configure(shape_x, shape_y, output_shape_ref),
               std::runtime_error);
}

TEST(MatMulLayerMockTest, exec_before_configure_fail) {
  MatMulInfo mm_info;
  MatMulLayerMock layer(77, mm_info);
  Tensor<double> input(Shape({1, 1}));
  Tensor<double> output(Shape({1, 1}));

  EXPECT_THROW(layer.exec(input, output), std::runtime_error);
}

TEST(MatMulLayerMockTest, get_output_shape_before_configure_fail) {
  MatMulInfo mm_info;
  MatMulLayerMock layer(78, mm_info);

  EXPECT_THROW(layer.get_output_shape(), std::logic_error);
}

TEST(MatMulLayerMockTest, exec_success_after_configure) {
  MatMulInfo mm_info;
  MatMulLayerMock layer(79, mm_info);
  Shape shape_x({2, 3}), shape_y({3, 4}), out_ref;

  layer.configure(shape_x, shape_y, out_ref);
  Tensor<double> t_in(shape_x);
  Tensor<double> t_out(out_ref);

  ASSERT_NO_THROW(layer.exec(t_in, t_out));
}

TEST(MatMulLayerMockTest, get_type_name_returns_correct_name) {
  MatMulInfo mm_info;
  MatMulLayerMock layer(80, mm_info);

  EXPECT_EQ(layer.get_type_name(), "MatMulLayerMock");
}