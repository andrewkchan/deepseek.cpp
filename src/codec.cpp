#include "codec.h"

#include "quant.h"

#include <fcntl.h>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

std::string quant_to_string(Quant quant) {
  switch (quant) {
    case Quant::F32: return "F32";
    case Quant::F16: return "F16";
    case Quant::F8E5M2: return "F8_E5M2";
    case Quant::Q2_K: return "Q2_K";
    case Quant::Q3_K: return "Q3_K";
  }
  __builtin_unreachable();
}

std::optional<Quant> string_to_quant(const std::string& quant_str) {
  if (quant_str == "F32") {
    return Quant::F32;
  } else if (quant_str == "F16") {
    return Quant::F16;
  } else if (quant_str == "F8_E5M2") {
    return Quant::F8E5M2;
  } else if (quant_str == "Q2_K") {
    return Quant::Q2_K;
  } else if (quant_str == "Q3_K") {
    return Quant::Q3_K;
  } else {
    return std::nullopt;
  }
}

double bits_per_weight(Quant quant, size_t blockwise_quant_size) {
  if (blockwise_quant_size > 0 && quant != Quant::F8E5M2) {
    std::cerr << "blockwise quantization should only be used with F8E5M2" << std::endl;
    assert(false);
  }
  switch (quant) {
    case Quant::F32: return 4;
    case Quant::F16: return 2;
    case Quant::F8E5M2: return (4 + blockwise_quant_size) / blockwise_quant_size;
    case Quant::Q2_K: return 2.5625;
    case Quant::Q3_K: return 3.4375;
  }
  __builtin_unreachable();
}

CodecDType quant_to_codec_dtype(Quant quant) {
  switch (quant) {
    case Quant::F32: return CodecDType::F32;
    case Quant::F16: return CodecDType::F16;
    case Quant::F8E5M2: return CodecDType::F8E5M2;
    case Quant::Q2_K: return CodecDType::U8;
    case Quant::Q3_K: return CodecDType::U8;
  }
  __builtin_unreachable();
}

bool is_k_quant(Quant quant) {
  return quant == Quant::Q2_K || quant == Quant::Q3_K;
}

std::string codec_dtype_to_string(CodecDType dtype) {
  switch (dtype) {
    case CodecDType::F32: return "F32";
    case CodecDType::F16: return "F16";
    case CodecDType::BF16: return "BF16";
    case CodecDType::F8E5M2: return "F8_E5M2";
    case CodecDType::F8E4M3: return "F8_E4M3";
    case CodecDType::I32: return "I32";
    case CodecDType::I16: return "I16";
    case CodecDType::I8: return "I8";
    case CodecDType::U8: return "U8";
  }
  return "UNKNOWN";
}

std::optional<CodecDType> string_to_codec_dtype(const std::string& dtype_str) {
  if (dtype_str == "F32") {
    return CodecDType::F32;
  } else if (dtype_str == "F16") {
    return CodecDType::F16;
  } else if (dtype_str == "BF16") {
    return CodecDType::BF16;
  } else if (dtype_str == "F8_E5M2") {
    return CodecDType::F8E5M2;
  } else if (dtype_str == "F8_E4M3") {
    return CodecDType::F8E4M3;
  } else if (dtype_str == "I32") {
    return CodecDType::I32;
  } else if (dtype_str == "I16") {
    return CodecDType::I16;
  } else if (dtype_str == "I8") {
    return CodecDType::I8;
  } else if (dtype_str == "U8") {
    return CodecDType::U8;
  } else {
    return std::nullopt;
  }
}

size_t codec_dtype_size(CodecDType dtype) {
  switch (dtype) {
    case CodecDType::F32: return 4;
    case CodecDType::F16: return 2;
    case CodecDType::BF16: return 2;
    case CodecDType::F8E5M2: return 1;
    case CodecDType::F8E4M3: return 1;
    case CodecDType::I32: return 4;
    case CodecDType::I16: return 2;
    case CodecDType::I8: return 1;
    case CodecDType::U8: return 1;
  }
  return 0;
}

int Tensor::from_json(const std::string& name, const json& val, void* bytes_ptr, size_t bytes_size) {
  this->name = name;
  std::string dtype_str = val.value("dtype", ""); 
  if (auto dtype = string_to_codec_dtype(dtype_str)) {
    this->dtype = *dtype;
  } else {
    std::cerr << "bad dtype" << std::endl;
    return -1;
  }
  size_t dsize = codec_dtype_size(this->dtype);

  size_t numel = 1;
  if (val.at("shape").size() > 4) {
    std::cerr << "shape exceeds 4 dimensions" << std::endl;
  }
  for (size_t i = 0; i < val.at("shape").size() && i < 4; i++) {
    if (val.at("shape")[i].get<int>() != val.at("shape")[i]) {
      std::cerr << "bad shape" << std::endl;
      return -1;
    }
    shape[i] = val.at("shape")[i].get<int>();
    numel *= shape[i];
  }
  if (val.at("data_offsets").size() != 2) {
    return -1;
  }
  size_t offset_start = static_cast<size_t>(val.at("data_offsets")[0]);
  size_t offset_end = static_cast<size_t>(val.at("data_offsets")[1]);
  if (offset_start < 0 || offset_end <= offset_start || offset_end > bytes_size) {
    std::cerr << "bad offsets" << std::endl;
    return -1;
  }
  this->data = (char*)bytes_ptr + offset_start;
  this->size = offset_end - offset_start;
  // validate the shape matches the size
  if (numel * dsize != this->size) {
    std::cerr << "bad size" << std::endl;
    return -1;
  }
  return 0;
}

int YALMData::update_from_file(const std::string& filename, bool read_metadata, bool lock_model_weights) {
  std::cout << "loading data from file: " << filename << std::endl;
  int fd = open(filename.c_str(), O_RDONLY);
  if (fd == -1) {
    return -1;
  }

  struct stat st;
  if (fstat(fd, &st) != 0) {
    close(fd);
    return -1;
  }
  
  size_t size = st.st_size;
  int mmap_flags = MAP_PRIVATE;
  if (lock_model_weights) {
    // Eagerly load memory-mapped file into memory.
    // This ensures the mlock call later is locking memory already in RAM.
    mmap_flags |= MAP_POPULATE;
  }
  void* data = mmap(NULL, size, PROT_READ | PROT_WRITE, mmap_flags, fd, 0);
  if (data == MAP_FAILED) {
    close(fd);
    return -1;
  }
  if (lock_model_weights && mlock(data, size) != 0) {
    std::cerr << "Warning: mlock failed for model data. Performance may be suboptimal. Are you running as sudo?" << std::endl;
  }

#ifdef __linux__
  // increases readahead buffer size, resulting in faster cold loads
  posix_fadvise(fd, 0, size, POSIX_FADV_SEQUENTIAL);
#endif

  close(fd);

  // Parse the metadata JSON and the tensors
  if (size < sizeof(uint64_t)) {
    munmap(data, size);
    return -1;
  }

  uint64_t json_size = *(uint64_t*)data;
  if (json_size == 0 || json_size > size - sizeof(uint64_t)) {
    munmap(data, size);
    return -1;
  }

  char* json_ptr = (char*)data + sizeof(uint64_t);
  void* bytes_ptr = (char*)data + sizeof(uint64_t) + json_size;
  size_t bytes_size = size - sizeof(uint64_t) - json_size;

  std::string json_str(json_ptr, json_size);
  json header = json::parse(json_str);

  for (auto& [key, val] : header.items()) {
    if (key == "__metadata__" && read_metadata) {
      metadata = val;
    } else if (key != "__metadata__") {
      Tensor& tensor = tensors[key];
      if (tensor.from_json(key, val, bytes_ptr, bytes_size) != 0) {
        std::cerr << "failed to parse tensor " << key << std::endl;
        munmap(data, size);
        return -1;
      }
    }
  }

  return 0;
}

int YALMData::from_directory(const std::string& dirname, bool lock_model_weights) {
  std::vector<std::string> files;
  DIR* dir = opendir(dirname.c_str());
  if (dir == nullptr) {
    std::cout << "failed to open directory" << std::endl;
    return -1;
  }

  // Collect all files
  struct dirent* entry;
  while ((entry = readdir(dir)) != nullptr) {
    std::string filename = entry->d_name;
    // Skip . and .. directory entries
    if (filename != "." && filename != "..") {
      files.push_back(dirname + "/" + filename);
    }
  }
  closedir(dir);

  if (files.empty()) {
    std::cout << "no files found" << std::endl;
    return -1;
  }

  // Sort files to ensure consistent ordering
  std::sort(files.begin(), files.end());

  // Read first file with metadata
  if (update_from_file(files[0], true, lock_model_weights) != 0) {
    std::cout << "failed to read metadata" << std::endl;
    return -1;
  }

  std::cout << "read metadata " << metadata << std::endl;

  // Read remaining files without metadata
  for (size_t i = 1; i < files.size(); i++) {
    if (update_from_file(files[i], false, lock_model_weights) != 0) {
      std::cout << "failed to read file " << files[i] << std::endl;
      return -1;
    }
  }

  return 0;
}