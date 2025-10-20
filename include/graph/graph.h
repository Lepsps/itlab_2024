#ifndef GRAPH_H
#define GRAPH_H

#include <string>
#include <unordered_map>
#include <vector>

#include "./layer/layer.h"
#include "./tensor/tensor.h"

class Network {
 private:
  std::unordered_map<int, Layer*> layers_;
  Tensor<double> inputTensor_;
  Tensor<double>* outputTensor_;
  int start_ = -1;
  int end_ = -1;
  bool bfs_helper(int start, int vert, bool flag,
                  std::vector<int>* v_ord) const;

 public:
  Network();

  bool addLayer(Layer& lay, const std::vector<int>& inputs = {},
                const std::vector<int>& outputs = {});
  void addEdge(Layer& layPrev, Layer& layNext);
  void removeEdge(Layer& layPrev, Layer& layNext);
  void removeLayer(Layer& lay);
  int getLayers() const;
  int getEdges() const;
  bool isEmpty() const;
  bool hasPath(Layer& layPrev, Layer& layNext) const;
  std::vector<int> inference(int start) const;
  void setInput(Layer& lay, Tensor<double>& vec);
  void setOutput(Layer& lay, Tensor<double>& vec);
  void run();
  std::vector<std::string> getLayersTypeVector() const;
  ~Network();
};

#endif
