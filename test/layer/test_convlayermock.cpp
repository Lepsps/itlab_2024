#include <stdexcept>

#include "./layer/ConvLayer.h"
#include "./tensor/tensor.h"
#include "gtest/gtest.h"

TEST(ConvolutionLayerMockTest, configure_success_simple_conv_no_bias) {
  ConvolutionLayerMock layer(50);
  Shape input_s({32, 32, 3});
  Shape weights_s({3, 3, 3, 16});
  Shape output_s_ref;
  ConvPadStrideInfo conv_info(1, 1, 0, 0);
  Shape expected_output_shape({30, 30, 16});

  ASSERT_NO_THROW(
      layer.configure(input_s, weights_s, nullptr, output_s_ref, conv_info));
  EXPECT_EQ(output_s_ref.dimensions, expected_output_shape.dimensions);
  EXPECT_EQ(layer.get_output_shape().dimensions,
            expected_output_shape.dimensions);
}

TEST(ConvolutionLayerMockTest, configure_success_conv_with_bias_and_padding) {
  ConvolutionLayerMock layer(51);
  Shape input_s({32, 32, 3});
  Shape weights_s({3, 3, 3, 16});
  Shape biases_s({16});
  Shape output_s_ref;
  ConvPadStrideInfo conv_info(1, 1, 1, 1);
  Shape expected_output_shape({32, 32, 16});

  ASSERT_NO_THROW(
      layer.configure(input_s, weights_s, &biases_s, output_s_ref, conv_info));
  EXPECT_EQ(output_s_ref.dimensions, expected_output_shape.dimensions);
}

TEST(ConvolutionLayerMockTest, configure_success_conv_with_stride) {
  ConvolutionLayerMock layer(52);
  Shape input_s({32, 32, 3});
  Shape weights_s({3, 3, 3, 16});
  Shape output_s_ref;
  ConvPadStrideInfo conv_info(2, 2, 0, 0);
  Shape expected_output_shape({15, 15, 16});

  ASSERT_NO_THROW(
      layer.configure(input_s, weights_s, nullptr, output_s_ref, conv_info));
  EXPECT_EQ(output_s_ref.dimensions, expected_output_shape.dimensions);
}

TEST(ConvolutionLayerMockTest, configure_fail_input_rank_too_low) {
  ConvolutionLayerMock layer(53);
  Shape input_s({32, 32});
  Shape weights_s({3, 3, 3, 16});
  Shape output_s_ref;
  ConvPadStrideInfo conv_info;

  EXPECT_THROW(
      layer.configure(input_s, weights_s, nullptr, output_s_ref, conv_info),
      std::runtime_error);
}

TEST(ConvolutionLayerMockTest, configure_fail_weights_rank_not_4) {
  ConvolutionLayerMock layer(54);
  Shape input_s({32, 32, 3});
  Shape weights_s({3, 3, 3});
  Shape output_s_ref;
  ConvPadStrideInfo conv_info;

  EXPECT_THROW(
      layer.configure(input_s, weights_s, nullptr, output_s_ref, conv_info),
      std::runtime_error);
}

TEST(ConvolutionLayerMockTest, configure_fail_channel_mismatch) {
  ConvolutionLayerMock layer(55);
  Shape input_s({32, 32, 3});
  Shape weights_s({3, 3, 1, 16});
  Shape output_s_ref;
  ConvPadStrideInfo conv_info;

  EXPECT_THROW(
      layer.configure(input_s, weights_s, nullptr, output_s_ref, conv_info),
      std::runtime_error);
}

TEST(ConvolutionLayerMockTest, configure_fail_bias_rank_not_1) {
  ConvolutionLayerMock layer(56);
  Shape input_s({32, 32, 3});
  Shape weights_s({3, 3, 3, 16});
  Shape biases_s({16, 1});
  Shape output_s_ref;
  ConvPadStrideInfo conv_info;

  EXPECT_THROW(
      layer.configure(input_s, weights_s, &biases_s, output_s_ref, conv_info),
      std::runtime_error);
}

TEST(ConvolutionLayerMockTest, configure_fail_bias_size_mismatch) {
  ConvolutionLayerMock layer(57);
  Shape input_s({32, 32, 3});
  Shape weights_s({3, 3, 3, 16});
  Shape biases_s({15});
  Shape output_s_ref;
  ConvPadStrideInfo conv_info;

  EXPECT_THROW(
      layer.configure(input_s, weights_s, &biases_s, output_s_ref, conv_info),
      std::runtime_error);
}

TEST(ConvolutionLayerMockTest,
     configure_fail_kernel_too_large_for_padded_input) {
  ConvolutionLayerMock layer(58);
  Shape input_s({3, 3, 1});
  Shape weights_s({5, 5, 1, 1});
  Shape output_s_ref;
  ConvPadStrideInfo conv_info(1, 1, 0, 0);

  EXPECT_THROW(
      layer.configure(input_s, weights_s, nullptr, output_s_ref, conv_info),
      std::runtime_error);
}

TEST(ConvolutionLayerMockTest, exec_before_configure_fail) {
  ConvolutionLayerMock layer(59);
  Tensor<double> input(Shape({1, 1, 1}));
  Tensor<double> output(Shape({1, 1, 1}));

  EXPECT_THROW(layer.exec(input, output), std::runtime_error);
}

TEST(ConvolutionLayerMockTest, get_output_shape_before_configure_fail) {
  ConvolutionLayerMock layer(60);

  EXPECT_THROW(layer.get_output_shape(), std::logic_error);
}

TEST(ConvolutionLayerMockTest, exec_success_after_configure) {
  ConvolutionLayerMock layer(61);
  Shape input_s({3, 3, 1});
  Shape weights_s({3, 3, 1, 1});
  Shape out_ref;
  ConvPadStrideInfo info;

  layer.configure(input_s, weights_s, nullptr, out_ref, info);
  Tensor<double> t_in(input_s);
  Tensor<double> t_out(out_ref);

  ASSERT_NO_THROW(layer.exec(t_in, t_out));
}

TEST(ConvolutionLayerMockTest, get_type_name_returns_correct_name) {
  ConvolutionLayerMock layer(62);

  EXPECT_EQ(layer.get_type_name(), "ConvolutionLayerMock");
}