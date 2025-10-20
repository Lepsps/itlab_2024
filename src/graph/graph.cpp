#include "./graph/graph.h"

#include <iostream>
#include <list>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "./layer/layer.h"
#include "./tensor/tensor.h"

Network::Network() : inputTensor_(), outputTensor_(nullptr) {}

bool Network::addLayer(Layer& lay, const std::vector<int>& inputs,
                       const std::vector<int>& outputs) {
  if (layers_.find(lay.getID()) == layers_.end()) {
    layers_[lay.getID()] = &lay;

    for (int input_layer_id : inputs) {
      auto it = layers_.find(input_layer_id);
      if (it != layers_.end()) {
        Layer* prev_layer = it->second;
        prev_layer->addNeighbor(&lay);
      }
    }

    for (int output_layer_id : outputs) {
      auto it = layers_.find(output_layer_id);
      if (it != layers_.end()) {
        Layer* next_layer = it->second;
        lay.addNeighbor(next_layer);
      }
    }
    return true;
  }
  return false;
}

void Network::addEdge(Layer& layPrev, Layer& layNext) {
  if (layPrev.getID() == layNext.getID()) {
    throw std::invalid_argument("Cannot add edge from a layer to itself.");
  }
  if (layers_.find(layPrev.getID()) == layers_.end()) {
    addLayer(layPrev, {}, {});
  }
  if (layers_.find(layNext.getID()) == layers_.end()) {
    addLayer(layNext, {}, {});
  }
  layPrev.addNeighbor(&layNext);
}

void Network::removeEdge(Layer& layPrev, Layer& layNext) {
  if (layers_.find(layPrev.getID()) != layers_.end()) {
    layPrev.removeNeighbor(&layNext);
  }
}

void Network::removeLayer(Layer& lay) {
  int layer_id = lay.getID();

  if (layers_.find(layer_id) == layers_.end()) {
    return;
  }

  for (auto& pair : layers_) {
    pair.second->removeNeighbor(&lay);
  }

  auto it = layers_.find(layer_id);
  if (it != layers_.end()) {
    layers_.erase(it);
  }

  if (start_ == layer_id) {
    start_ = -1;
  }
  if (end_ == layer_id) {
    end_ = -1;
  }
}

int Network::getLayers() const { return static_cast<int>(layers_.size()); }

int Network::getEdges() const {
  int count = 0;
  for (const auto& layer : layers_) {
    count += layer.second->neighbors_.size();
  }
  return count;
}

bool Network::isEmpty() const { return layers_.empty(); }

bool Network::bfs_helper(int start, int vert, bool flag,
                         std::vector<int>* v_ord) const {
  std::unordered_map<int, bool> visited;
  std::queue<int> queue;

  queue.push(start);
  visited[start] = true;

  while (!queue.empty()) {
    int current = queue.front();
    queue.pop();

    if (flag && current == vert) {
      return true;
    }

    if (v_ord != nullptr) {
      v_ord->push_back(current);
    }

    if (layers_.count(current) > 0) {
      Layer* current_layer = layers_.at(current);

      for (Layer* neighbor : current_layer->neighbors_) {
        if (visited.find(neighbor->getID()) == visited.end()) {
          visited[neighbor->getID()] = true;
          queue.push(neighbor->getID());
        }
      }
    }
  }

  return false;
}

bool Network::hasPath(Layer& layPrev, Layer& layNext) const {
  if (layers_.find(layPrev.getID()) == layers_.end() ||
      layers_.find(layNext.getID()) == layers_.end()) {
    return false;
  }
  return bfs_helper(layPrev.getID(), layNext.getID(), true, nullptr);
}

std::vector<int> Network::inference(int start) const {
  std::vector<int> v_ord;
  bfs_helper(start, -1, false, &v_ord);
  return v_ord;
}

void Network::setInput(Layer& lay, Tensor<double>& vec) {
  if (start_ != -1) {
    throw std::runtime_error("Input layer already set.");
  }
  if (!layers_.empty()) {
    addLayer(lay);
  }
  inputTensor_ = vec;
  start_ = lay.getID();
}

void Network::setOutput(Layer& lay, Tensor<double>& vec) {
  if (end_ != -1) {
    throw std::runtime_error("Output layer already set.");
  }

  if (layers_.find(lay.getID()) == layers_.end()) {
    addLayer(lay);
  }

  end_ = lay.getID();
  outputTensor_ = &vec;
}

void Network::run() {
  if (start_ == -1 || end_ == -1) {
    throw std::runtime_error("Input or output layer not set.");
  }

  std::vector<int> path = inference(start_);

  bool end_in_path = false;
  for (int layer_id : path) {
    if (layer_id == end_) {
      end_in_path = true;
      break;
    }
  }
  if (path.empty() || !end_in_path) {
    throw std::runtime_error(
        "No path from start to end layer found, or traversal is empty.");
  }

  Tensor<double> curr_tensor = inputTensor_;

  std::unordered_map<int, Tensor<double>> layer_outputs;
  layer_outputs[start_] = inputTensor_;

  bool on_path = false;

  for (int layer_id : path) {
    if (layers_.find(layer_id) == layers_.end()) {
      throw std::runtime_error(
          "Layer_id from BFS traversal not found in graph.");
    }
    Layer* curr_layer_ptr = layers_.at(layer_id);
    if (!curr_layer_ptr) {
      throw std::runtime_error("Layer with ID is null.");
    }

    Tensor<double> curr_input({0});

    if (layer_id == start_) {
      curr_input = inputTensor_;
      on_path = true;
    } else if (on_path) {
      curr_input = curr_tensor;
    } else {
      continue;
    }

    Tensor<double> temp_tensor(curr_layer_ptr->get_output_shape());
    curr_layer_ptr->exec(curr_input, temp_tensor);
    curr_tensor = temp_tensor;

    if (layer_id == end_) {
      if (outputTensor_ == nullptr) {
        throw std::runtime_error("Output tensor pointer is not set.");
      }
      *outputTensor_ = curr_tensor;
    }
  }
}

std::vector<std::string> Network::getLayersTypeVector() const {
  std::vector<std::string> layer_types_vector;

  if (start_ == -1) {
    layer_types_vector.emplace_back(
        "Error: Input layer (start_ ID) has not been set via setInput().");
    return layer_types_vector;
  }

  if (layers_.find(start_) == layers_.end()) {
    layer_types_vector.emplace_back(
        "Error: Start layer with ID not found in the graph's layers map.");
    return layer_types_vector;
  }

  std::vector<int> traversal_order = inference(start_);

  if (traversal_order.empty()) {
    if (layers_.count(start_)) {
      layer_types_vector.emplace_back(
          "Warning: BFS traversal from start layer ID yielded no layers (or "
          "only the start layer was expected).");
      layer_types_vector.emplace_back("Start layer type: " +
                                      layers_.at(start_)->get_type_name());
    } else {
      layer_types_vector.emplace_back(
          "Error: BFS traversal from start layer ID failed, and start layer "
          "itself is not in graph.");
    }
    return layer_types_vector;
  }

  for (int layer_id : traversal_order) {
    auto it = layers_.find(layer_id);
    if (it == layers_.end()) {
      layer_types_vector.emplace_back(
          "Error: Layer ID from BFS traversal not found in graph's layers map");
      continue;
    }

    Layer* current_layer = it->second;
    layer_types_vector.push_back(current_layer->get_type_name());
  }
  return layer_types_vector;
}

Network::~Network() = default;