#include <stdexcept>
#include <string>

#include "./tensor/tensor.h"
#include "gtest/gtest.h"

TEST(ShapeTest, get_rank_and_elem_checks) {
  Shape s({2, 3, 4});

  ASSERT_EQ(s.get_rank(), 3);
  ASSERT_EQ(s.total_elements, 24);
}

TEST(TensorTestDouble, can_at_to_tensor) {
  Tensor<double> t({2, 3}, Layout::kNd);

  t.at({0, 0}) = 1.0;
  t.at({0, 1}) = 2.0;
  t.at({0, 2}) = 3.0;
  t.at({1, 0}) = 4.0;
  t.at({1, 1}) = 5.0;
  t.at({1, 2}) = 6.0;

  ASSERT_DOUBLE_EQ(t.at({0, 0}), 1.0);
  ASSERT_DOUBLE_EQ(t.at({0, 1}), 2.0);
  ASSERT_DOUBLE_EQ(t.at({0, 2}), 3.0);
  ASSERT_DOUBLE_EQ(t.at({1, 0}), 4.0);
  ASSERT_DOUBLE_EQ(t.at({1, 1}), 5.0);
  ASSERT_DOUBLE_EQ(t.at({1, 2}), 6.0);

  const Tensor<double> &ct = t;
  ASSERT_DOUBLE_EQ(ct.at({0, 1}), 2.0);
}

TEST(TensorTestDouble, can_get_linear_index2D_ND_RowMajor) {
  Tensor<double> t({2, 3}, Layout::kNd);

  ASSERT_EQ(t.get_linear_index({0, 0}), 0 * 3 + 0);
  ASSERT_EQ(t.get_linear_index({0, 1}), 0 * 3 + 1);
  ASSERT_EQ(t.get_linear_index({0, 2}), 0 * 3 + 2);
  ASSERT_EQ(t.get_linear_index({1, 0}), 1 * 3 + 0);
  ASSERT_EQ(t.get_linear_index({1, 1}), 1 * 3 + 1);
  ASSERT_EQ(t.get_linear_index({1, 2}), 1 * 3 + 2);
}

TEST(TensorTestDouble, can_get_linear_index4D_NCHW) {
  Tensor<double> t({2, 3, 4, 5}, Layout::kNchw);

  ASSERT_EQ(t.get_linear_index({0, 0, 0, 0}), 0);
  ASSERT_EQ(t.get_linear_index({1, 2, 3, 4}), 119);
}

TEST(TensorTestDouble, can_get_linear_index4D_NHWC) {
  Tensor<double> t({2, 3, 4, 5}, Layout::kNhwc);

  ASSERT_EQ(t.get_linear_index({0, 0, 0, 0}), 0);
  ASSERT_EQ(t.get_linear_index({1, 2, 3, 4}), 119);
  ASSERT_EQ(t.get_linear_index({0, 1, 2, 3}), 40);
}

TEST(TensorTestDouble, can_get_linear_index4D_ND_is_RowMajor) {
  Tensor<double> t4d_nd({2, 3, 4, 5}, Layout::kNd);

  ASSERT_EQ(t4d_nd.get_linear_index({0, 0, 0, 0}), 0);
  ASSERT_EQ(t4d_nd.get_linear_index({1, 2, 3, 4}), 119);
}

TEST(TensorTestDouble, cant_get_linear_index_out_of_bounds) {
  Tensor<double> t2d({2, 3}, Layout::kNd);
  Tensor<double> t4d_nchw({2, 3, 4, 5}, Layout::kNchw);
  Tensor<double> t4d_nhwc({2, 3, 4, 5}, Layout::kNhwc);
  Tensor<double> t4d_nd({2, 3, 4, 5}, Layout::kNd);

  EXPECT_THROW(t2d.get_linear_index({2, 0}), std::out_of_range);
  EXPECT_THROW(t2d.get_linear_index({0, 3}), std::out_of_range);
  EXPECT_THROW(t4d_nchw.get_linear_index({2, 0, 0, 0}), std::out_of_range);
  EXPECT_THROW(t4d_nhwc.get_linear_index({0, 3, 0, 0}), std::out_of_range);
  EXPECT_THROW(t4d_nd.get_linear_index({0, 0, 4, 0}), std::out_of_range);
}

TEST(TensorTestDouble, cant_get_linear_index_with_wrong_num_of_indicies) {
  Tensor<double> t2d({2, 3}, Layout::kNd);
  Tensor<double> t4d_nchw({2, 3, 4, 5}, Layout::kNchw);

  EXPECT_THROW(t2d.get_linear_index({0}), std::runtime_error);
  EXPECT_THROW(t4d_nchw.get_linear_index({0, 0, 0}), std::runtime_error);
}