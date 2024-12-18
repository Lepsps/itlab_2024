#include "./graph/graph.h"
#include "./graph/graph.cpp"
#include <gtest.h>


TEST(Graph, can_create_graph)
{
	ASSERT_NO_THROW(Graph g);
}

TEST(Graph, can_add_vertex_to_graph)
{
	Graph g;

	ASSERT_NO_THROW(g.addVertex(1));
}

TEST(Graph, can_add_edge_to_graph)
{
	Graph g;

	ASSERT_NO_THROW(g.addEdge(1,0));
}

TEST(Graph, can_add_edge_with_same_id_to_graph)
{
	Graph g;

	ASSERT_NO_THROW(g.addEdge(1, 1));
}

TEST(Graph, can_add_vertex_and_edge_to_graph)
{
	Graph g;

	ASSERT_NO_THROW(g.addEdge(1, 0));
	ASSERT_NO_THROW(g.addVertex(1));
}

TEST(Graph, can_get_vertex)
{
	Graph g;
	g.addEdge(1, 0);

	ASSERT_NO_THROW(g.getVertex());
}

TEST(Graph, can_remove_vertex)
{
	Graph g;
	g.addEdge(1, 0);

	g.removeVertex(1);

	ASSERT_EQ(g.vertexCount(),1);
}

TEST(Graph, can_get_vertex_count)
{
	Graph g;

	g.addEdge(1, 0);
	g.addVertex(2);
	g.addVertex(3);
	g.removeVertex(1);
	g.addVertex(1);
	g.removeVertex(3);

	ASSERT_EQ(g.vertexCount(), 3);
}

TEST(Graph, can_get_edge_count)
{
	Graph g;

	g.addEdge(1, 0);
	g.addEdge(2, 0);
	g.addEdge(0, 2);
	g.addEdge(2, 2);

	ASSERT_EQ(g.edgeCount(), 3);
}

TEST(Graph, check_graph_is_empty)
{
	Graph g;

	ASSERT_TRUE(g.empty());
}

TEST(Graph, check_graph_is_not_empty)
{
	Graph g;

	g.addEdge(1, 0);

	ASSERT_FALSE(g.empty());
}

TEST(Graph, check_graph_no_path_between_vertexes)
{
	Graph g;

	g.addEdge(0,1);
	g.addEdge(0, 3);
	g.addEdge(1, 2);
	g.addEdge(3, 1);

	ASSERT_FALSE(g.hasPath(1,0));
}

TEST(Graph, check_graph_has_path_between_vertexes)
{
	Graph g;

	g.addEdge(0, 1);
	g.addEdge(0, 3);
	g.addEdge(1, 2);
	g.addEdge(3, 1);

	ASSERT_TRUE(g.hasPath(3,1));
}

TEST(Graph, check_graph_can_find_path_between_vertexes_after_delete)
{
	Graph g;

	g.addEdge(0, 1);
	g.addEdge(0, 3);
	g.addEdge(1, 2);
	g.addEdge(3, 1);
	g.removeVertex(3);

	ASSERT_FALSE(g.hasPath(3, 1));
	ASSERT_FALSE(g.hasPath(0, 3));
}

TEST(Graph, can_create_bfs_path)
{
	Graph g;

	g.addEdge(0, 1);
	g.addEdge(0, 3);
	g.addEdge(1, 2);

	ASSERT_NO_THROW(g.BFS(0));
}

TEST(Graph, can_create_bfs_path_in_empty_graph)
{
	Graph g;

	ASSERT_NO_THROW(g.BFS(0));
}

TEST(Graph, check_bfs_path)
{
	Graph g;
	std::vector<int> v1 = { 0,1,3,2,4,6,7,5,8,9 };

	g.addEdge(0, 1);
	g.addEdge(0, 3);
	g.addEdge(1, 2);
	g.addEdge(3, 4);
	g.addEdge(4, 5);
	g.addEdge(3,6);
	g.addEdge(3, 7);
	g.addEdge(7,8);
	g.addEdge(8, 9);
	std::vector<int> v = g.BFS(0);

	ASSERT_EQ(v, v1);
}