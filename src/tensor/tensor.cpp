#include "./tensor/tensor.h"

#include <cstddef>
#include <numeric>
#include <utility>
#include <vector>

Shape::Shape(std::vector<size_t> dims) : dimensions(std::move(dims)) {
  if (this->dimensions.empty()) {
    total_elements = 1;
  } else {
    total_elements = std::accumulate(dimensions.begin(), dimensions.end(),
                                     static_cast<size_t>(1),
                                     [](size_t a, size_t b) { return a * b; });
  }
}

size_t Shape::get_rank() const { return dimensions.size(); }