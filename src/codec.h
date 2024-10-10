#pragma once

#include "json.hpp"

#include <string>

using json = nlohmann::json;

// TODO: Should this be narrowed down to what we actually support for model weight representation?
enum class DType {
	dt_f32,
	dt_f16,
	dt_bf16,
	dt_f8e5m2,
	dt_f8e4m3,
	dt_i32,
	dt_i16,
	dt_i8,
	dt_u8,
};

constexpr size_t MAX_TENSORS = 1024;

struct Tensor {
  std::string name;
  DType dtype;
  int shape[8] = {}; // Initialize the shape array with zeros
  void* data = nullptr;
  size_t size;

  // Returns 0 if successful, other if failed
  int from_json(const std::string& name, const json& j, void* bytes_ptr);
};

struct YALMData {
  void* data = nullptr;
  size_t size;

  json metadata;

  // TODO: use a vector instead of this C-style array?
  Tensor tensors[MAX_TENSORS];
  int n_tensors;

  // Returns 0 if successful, other if failed
  int from_file(const char* filename);
};