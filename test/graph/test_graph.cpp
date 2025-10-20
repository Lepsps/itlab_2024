#include <unordered_set>

#include "./graph/graph.h"
#include "./layer/ConvLayer.cpp"
#include "./layer/PoolingLayer.cpp"
#include "./tensor/tensor.h"
#include "gtest/gtest.h"

TEST(NetworkTest, IsEmpty_InitiallyTrue) {
  Network network;

  EXPECT_TRUE(network.isEmpty());
  EXPECT_EQ(network.getLayers(), 0);
  EXPECT_EQ(network.getEdges(), 0);
}

TEST(NetworkTest, AddLayer_IncrementsLayerCount) {
  Network network;
  ConvolutionLayerMock conv1(1);

  network.addLayer(conv1);

  EXPECT_FALSE(network.isEmpty());
  EXPECT_EQ(network.getLayers(), 1);
}

TEST(NetworkTest, AddExistingLayer_DoesNotIncrementCount) {
  Network network;
  ConvolutionLayerMock conv1(1);

  network.addLayer(conv1);
  network.addLayer(conv1);

  EXPECT_EQ(network.getLayers(), 1);
}

TEST(NetworkTest, AddEdge_IncrementsEdgeCount) {
  Network network;
  ConvolutionLayerMock conv1(1);
  ConvolutionLayerMock conv2(2);

  network.addLayer(conv1);
  network.addLayer(conv2);
  network.addEdge(conv1, conv2);

  EXPECT_EQ(network.getEdges(), 1);
}

TEST(NetworkTest, AddEdge_LayersNotInGraph_AddsThem) {
  Network network;
  ConvolutionLayerMock conv1(1);
  ConvolutionLayerMock conv2(2);

  network.addEdge(conv1, conv2);

  EXPECT_EQ(network.getLayers(), 2);
  EXPECT_EQ(network.getEdges(), 1);
}

TEST(NetworkTest, RemoveLayer_DecrementsCountsAndRemovesEdges) {
  Network network;
  ConvolutionLayerMock conv1(1);
  ConvolutionLayerMock conv2(2);
  ConvolutionLayerMock conv3(3);

  network.addEdge(conv1, conv2);
  network.addEdge(conv2, conv3);
  network.removeLayer(conv2);

  EXPECT_EQ(network.getLayers(), 2);
  EXPECT_EQ(network.getEdges(), 0);
  EXPECT_FALSE(network.hasPath(conv1, conv3));
}

TEST(NetworkTest, HasPath_SimplePath_ReturnsTrue) {
  Network network;
  ConvolutionLayerMock conv1(1);
  ConvolutionLayerMock conv2(2);

  network.addEdge(conv1, conv2);

  EXPECT_TRUE(network.hasPath(conv1, conv2));
}

TEST(NetworkTest, HasPath_NoPath_ReturnsFalse) {
  Network network;
  ConvolutionLayerMock conv1(1);
  ConvolutionLayerMock conv2(2);
  ConvolutionLayerMock conv3(3);

  network.addEdge(conv1, conv2);

  EXPECT_FALSE(network.hasPath(conv1, conv3));
  EXPECT_FALSE(network.hasPath(conv3, conv1));
}

TEST(NetworkTest, HasPath_LayerNotInGraph_ReturnsFalse) {
  Network network;
  ConvolutionLayerMock conv1(1);
  ConvolutionLayerMock conv2(2);

  network.addLayer(conv1);

  EXPECT_FALSE(network.hasPath(conv1, conv2));
}

TEST(NetworkTest, Inference_ReturnsCorrectOrder) {
  Network network;
  ConvolutionLayerMock l1(1), l2(2), l3(3), l4(4);

  network.addEdge(l1, l2);
  network.addEdge(l1, l3);
  network.addEdge(l2, l4);
  std::vector<int> order = network.inference(l1.getID());

  ASSERT_EQ(order.size(), 4);
  EXPECT_EQ(order[0], 1);
  bool found2 = false, found3 = false;
  for (size_t i = 1; i < 3; ++i) {
    if (order[i] == 2) found2 = true;
    if (order[i] == 3) found3 = true;
  }
  EXPECT_TRUE(found2);
  EXPECT_TRUE(found3);
}

TEST(NetworkTest, Run_SimpleLinearNet_Success) {
  Network network;
  ConvPadStrideInfo conv_info_default;
  PoolingLayerInfo pool_info_default;
  pool_info_default.pool_size_x = 1;
  pool_info_default.pool_size_y = 1;
  pool_info_default.stride_x = 1;
  pool_info_default.stride_y = 1;

  ConvolutionLayerMock conv1(10);
  Shape input_s1({1, 1, 3, 3});
  Shape weights_s1({3, 3, 1, 1});
  Shape output_s1_ref;
  PoolingLayerMock pool2(20, pool_info_default);
  Shape output_s2_ref;
  Tensor<double> net_input(input_s1);
  Tensor<double> net_output(output_s2_ref);

  conv1.configure(input_s1, weights_s1, nullptr, output_s1_ref,
                  conv_info_default);
  pool2.configure(output_s1_ref, output_s2_ref);
  network.addLayer(conv1);
  network.addLayer(pool2);
  network.addEdge(conv1, pool2);
  std::fill(net_input.data.begin(), net_input.data.end(), 1.0);
  network.setInput(conv1, net_input);
  network.setOutput(pool2, net_output);

  ASSERT_NO_THROW(network.run());
}

TEST(NetworkTest, Run_Fail_NoInputSet) {
  Network network;
  ConvolutionLayerMock conv1(1);
  Tensor<double> dummy_out(Shape({1}));

  network.addLayer(conv1);
  network.setOutput(conv1, dummy_out);

  EXPECT_THROW(network.run(), std::runtime_error);
}

TEST(NetworkTest, Run_Fail_NoOutputSet) {
  Network network;
  ConvolutionLayerMock conv1(1);
  Tensor<double> dummy_in(Shape({1}));

  network.addLayer(conv1);
  network.setInput(conv1, dummy_in);

  EXPECT_THROW(network.run(), std::runtime_error);
}

TEST(NetworkTest, Run_Fail_NoPathFromStartToEnd) {
  Network network;
  ConvolutionLayerMock conv1(1), conv2(2), conv3(3);
  Tensor<double> dummy_in(Shape({1})), dummy_out(Shape({1}));

  network.addEdge(conv1, conv2);
  network.setInput(conv1, dummy_in);
  network.setOutput(conv3, dummy_out);

  EXPECT_THROW(network.run(), std::runtime_error);
}

TEST(NetworkTest, GetLayersTypeVector_SimpleNet) {
  Network network;
  ConvolutionLayerMock conv1(1);
  PoolingLayerInfo pool_info_default;
  PoolingLayerMock pool1(2, pool_info_default);
  Tensor<double> dummy_input(Shape({1}));

  network.addEdge(conv1, pool1);
  network.setInput(conv1, dummy_input);
  std::vector<std::string> types = network.getLayersTypeVector();

  ASSERT_EQ(types.size(), 2);
  EXPECT_EQ(types[0], conv1.get_type_name());
  EXPECT_EQ(types[1], pool1.get_type_name());
}

TEST(NetworkTest, GetLayersTypeVector_NoStartSet_ReturnsError) {
  Network network;
  ConvolutionLayerMock conv1(1);

  network.addLayer(conv1);
  std::vector<std::string> types = network.getLayersTypeVector();

  ASSERT_EQ(types.size(), 1);
  EXPECT_TRUE(
      types[0].find("Error: Input layer (start_ ID) has not been set") !=
      std::string::npos);
}

TEST(NetworkTest, Build_DiamondShapeGraph_StructureIsCorrect) {
  Network network;
  ConvolutionLayerMock start_node(1), left_node(2), right_node(3),
      merge_node(4);
  network.addEdge(start_node, left_node);
  network.addEdge(start_node, right_node);
  network.addEdge(left_node, merge_node);
  network.addEdge(right_node, merge_node);

  EXPECT_EQ(network.getLayers(), 4);
  EXPECT_EQ(network.getEdges(), 4);

  EXPECT_TRUE(network.hasPath(start_node, merge_node));
  EXPECT_TRUE(network.hasPath(start_node, left_node));
  EXPECT_TRUE(network.hasPath(start_node, right_node));

  EXPECT_FALSE(network.hasPath(left_node, right_node));
  EXPECT_FALSE(network.hasPath(right_node, left_node));
}

TEST(NetworkTest, Build_GraphWithSideInput_InferenceFindsAllNodes) {
  Network network;

  ConvolutionLayerMock input_node(1);
  ConvolutionLayerMock split_node(2);
  ConvolutionLayerMock left_branch(3);
  ConvolutionLayerMock right_branch(4);
  ConvolutionLayerMock merge_node(5);
  ConvolutionLayerMock side_input(6);

  network.addEdge(input_node, split_node);
  network.addEdge(split_node, left_branch);
  network.addEdge(split_node, right_branch);
  network.addEdge(left_branch, merge_node);
  network.addEdge(right_branch, merge_node);
  network.addEdge(side_input, merge_node);

  std::vector<int> order = network.inference(input_node.getID());

  EXPECT_EQ(order.size(), 5);

  std::unordered_set<int> visited_nodes(order.begin(), order.end());

  EXPECT_EQ(visited_nodes.count(1), 1);
  EXPECT_EQ(visited_nodes.count(2), 1);
  EXPECT_EQ(visited_nodes.count(3), 1);
  EXPECT_EQ(visited_nodes.count(4), 1);
  EXPECT_EQ(visited_nodes.count(5), 1);
  EXPECT_EQ(visited_nodes.count(6), 0);

  EXPECT_TRUE(network.hasPath(side_input, merge_node));
}