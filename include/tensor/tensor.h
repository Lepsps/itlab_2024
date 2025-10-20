#ifndef TENSOR_H
#define TENSOR_H

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct Shape {
  std::vector<size_t> dimensions;
  size_t total_elements;

  Shape() : total_elements(0) {}
  Shape(std::vector<size_t> dims);
  size_t get_rank() const;
};

enum Layout : std::uint8_t { kNchw, kNhwc, kNd };

template <typename T>
class Tensor {
 public:
  Shape shape;
  Layout layout;
  std::vector<T> data;

  Tensor() : shape(), layout(Layout::kNd), data() {}
  Tensor(const Shape &sh, Layout l = Layout::kNd);
  Tensor(std::vector<size_t> dims, Layout l = Layout::kNd);
  size_t get_linear_index(const std::vector<size_t> &indices) const;
  T &at(const std::vector<size_t> &indices);
  const T &at(const std::vector<size_t> &indices) const;
};

template <typename T>
Tensor<T>::Tensor(const Shape &sh, Layout l)
    : shape(sh), layout(l), data(sh.total_elements) {}

template <typename T>
Tensor<T>::Tensor(std::vector<size_t> dims, Layout l)
    : Tensor(Shape(std::move(dims)), l) {}

template <typename T>
size_t Tensor<T>::get_linear_index(const std::vector<size_t> &indices) const {
  if (indices.size() != shape.get_rank()) {
    throw std::runtime_error("Incorrect number of indices provided.");
  }
  for (size_t i = 0; i < indices.size(); ++i) {
    if (indices[i] >= shape.dimensions[i]) {
      std::string error_msg = "Index out of range for dimension ";
      throw std::out_of_range(error_msg);
    }
  }

  size_t linear_index = 0;
  size_t n = shape.get_rank();

  if (n == 0) {
    if (shape.total_elements == 1 && indices.empty()) {
      return 0;
    }
    if (shape.total_elements == 0 && indices.empty()) {
      return 0;
    }
    throw std::logic_error("Invalid access to rank-0 tensor or empty tensor.");
  }

  if (n == 4 && layout == Layout::kNhwc) {
    if (shape.dimensions.size() != 4) {
      throw std::logic_error(
          "kNhwc layout is specified for a tensor not of rank 4.");
    }

    size_t c_dim = shape.dimensions[1];
    size_t h_dim = shape.dimensions[2];
    size_t w_dim = shape.dimensions[3];

    linear_index = indices[0] * (h_dim * w_dim * c_dim) +
                   indices[2] * (w_dim * c_dim) + indices[3] * (c_dim) +
                   indices[1];
  } else {
    for (size_t i = 0; i < n; ++i) {
      size_t term_stride = 1;
      for (size_t j = i + 1; j < n; ++j) {
        term_stride *= shape.dimensions[j];
      }
      linear_index += indices[i] * term_stride;
    }
  }
  return linear_index;
}

template <typename T>
T &Tensor<T>::at(const std::vector<size_t> &indices) {
  return data[get_linear_index(indices)];
}

template <typename T>
const T &Tensor<T>::at(const std::vector<size_t> &indices) const {
  return data[get_linear_index(indices)];
}

#endif