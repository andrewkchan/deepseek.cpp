#pragma once

#include "json.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <dirent.h>
#include <algorithm>
#include <vector>
#include <optional>

#include "immintrin.h"
#include "f16cintrin.h"

using json = nlohmann::json;

typedef uint16_t f16_t;
typedef uint8_t f8e5m2_t;

#if defined(__AVX2__) && defined(__F16C__)
inline float half_to_float(f16_t x) {
  return _cvtsh_ss(x);
}
inline f16_t float_to_half(float x) {
  return _cvtss_sh(x, 0);
}
#else
inline float half_to_float(f16_t x) {
  assert(false && "float16 not supported on this platform");
  return 0.0f;
}
inline f16_t float_to_half(float x) {
  assert(false && "float16 not supported on this platform");
  return 0;
}
#endif

inline float float8e5m2_to_float(f8e5m2_t x) {
  f16_t val = 0;
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  memcpy(&val, &x, sizeof(f8e5m2_t));
#else
  memcpy((char*)&val + sizeof(f8e5m2_t), &x, sizeof(f8e5m2_t));
#endif
  return half_to_float(val);
}
[[maybe_unused]] inline f8e5m2_t float_to_float8e5m2(float x) {
  f16_t val = float_to_half(x);
  f8e5m2_t out;
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  memcpy(&out, (char*)&val, sizeof(f8e5m2_t)); // TODO: round instead of truncate?
#else
  memcpy(&out, (char*)&val + sizeof(f8e5m2_t), sizeof(f8e5m2_t)); // TODO: round instead of truncate?
#endif
  return out;
}

// Quant of tensors saved in the file.
// This corresponds to PyTorch tensor dtypes.
enum class CodecDType {
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
std::string codec_dtype_to_string(CodecDType dtype);
std::optional<CodecDType> string_to_codec_dtype(const std::string& dtype_str);
size_t codec_dtype_size(CodecDType dtype);

// Internal Quant.
// This corresponds to the in-memory representation of tensors in the model.
enum class Quant {
  F32,
  F16,
  F8E5M2,
  Q2_K, // 2-bit llama.cpp K-quants
};

std::string quant_to_string(Quant quant);
std::optional<Quant> string_to_quant(const std::string& quant_str);
double bits_per_weight(Quant quant, size_t blockwise_quant_size);
CodecDType quant_to_codec_dtype(Quant quant);

struct Tensor {
  std::string name;
  CodecDType dtype;
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