#pragma once

#include "json.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <dirent.h>
#include <algorithm>
#include <vector>

using json = nlohmann::json;

typedef uint16_t f16_t;
typedef uint8_t f8e5m2_t;

// TODO: Should this be narrowed down to what we actually support for model weight representation?
enum class DType {
  F32,
  F16,
  BF16,
  F8E5M2,
  F8E4M3,
  I32,
  I16,
  I8,
  U8,
};
std::string dtype_to_string(DType dtype);
size_t dtype_size(DType dtype);

struct Tensor {
  std::string name;
  DType dtype;
  std::array<int, 4> shape = {0, 0, 0, 0};
  void* data = nullptr;
  size_t size; // size in bytes (number of elements * element size)

  // Returns 0 if successful, other if failed
  int from_json(const std::string& name, const json& j, void* bytes_ptr, size_t bytes_size);
};

struct YALMData {
  json metadata;
  std::unordered_map<std::string, Tensor> tensors;

  // Update YALMData with tensors from a file
  // If read_metadata is true, also update metadata from this file
  // Returns 0 if successful, other if failed
  int update_from_file(const std::string& filename, bool read_metadata = false);

  // Initialize YALMData from all files in a directory
  // Metadata is read from the first file (in sorted order)
  // Returns 0 if successful, other if failed
  int from_directory(const std::string& dirname);
};