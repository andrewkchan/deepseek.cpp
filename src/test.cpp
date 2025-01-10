#include <iostream>
#include <memory>
#include <omp.h>
#include <random>
#include <thread>
#include <vector>

#include "immintrin.h"

#include "model.h"
#include "time.h"

bool floatEquals(float a, float b, float epsilon = 1e-5) {
  return std::abs(a - b) < epsilon;
}

bool arrayEquals(const std::vector<float>& a, const std::vector<float>& b, float epsilon = 1e-4) {
  if (a.size() != b.size()) {
    return false;
  }
  for (size_t i = 0; i < a.size(); i++) {
    if (!floatEquals(a[i], b[i], epsilon)) {
      return false;
    }
  }
  return true;
}

void assertArrayEquals(const std::vector<float>& actual, const std::vector<float>& expected, const std::string& message, float epsilon = 1e-4) {
  if (!arrayEquals(actual, expected, epsilon)) {
    std::cerr << "Assertion failed: " << message << std::endl;
    std::cerr << "actual: ";
    for (size_t i = 0; i < actual.size(); i++) {
      std::cerr << actual[i] << " ";
    }
    std::cerr << std::endl;
    std::cerr << "expected: ";
    for (size_t i = 0; i < expected.size(); i++) {
      std::cerr << expected[i] << " ";
    }
    std::cerr << std::endl;
    exit(1);
  }
}

void assertArrayEquals(float* actual, const std::vector<float>& expected, const std::string& message) {
  std::vector<float> actual_array;
  for (size_t i = 0; i < expected.size(); i++) {
    actual_array.push_back(actual[i]);
  }
  assertArrayEquals(actual_array, expected, message);
}

std::vector<f16_t> float_array_to_half(const std::vector<float>& data) {
#if defined(__AVX2__) && defined(__F16C__)
  std::vector<f16_t> half_data(data.size());
  for (size_t i = 0; i < data.size(); i++) {
    half_data[i] = _cvtss_sh(data[i], 0);
  }
  return half_data;
#else
  assert(false && "Cannot convert to half due to missing F16C extensions");
  return {};
#endif
}

void test_attn() {
  constexpr int TEST_SEQ_LEN = 4;
  constexpr int TEST_DIM = 6;
  constexpr int TEST_HEAD_DIM = 3;
  constexpr int TEST_N_HEADS = 2;
  constexpr int TEST_N_KV_HEADS = 1;
  std::shared_ptr<Config> config = std::make_shared<Config>();
  config->dim = TEST_DIM;
  config->hidden_dim = TEST_DIM;
  config->head_dim = TEST_HEAD_DIM;
  config->n_heads = TEST_N_HEADS;
  config->n_kv_heads = TEST_N_KV_HEADS;
  config->vocab_size = 1;
  config->max_seq_len = TEST_SEQ_LEN;
  InferenceState s(config);
  // (n_heads, head_dim) - query vectors
  std::vector<float> q{
    0., 1e4, 0., // h=0
    0., 0., 1e4 // h=1
  };
  for (size_t i = 0; i < q.size(); i++) {
    s.q()[i] = q[i];
  }
  std::vector<f16_t> kb = float_array_to_half({
    1., 0., 0., // t=0
    0., 1., 0., // t=1
    0., 0., 1., // t=2
    -1., 0., 0. // t=3
  }); // (kv_len, n_kv_heads, head_dim) - buffer containing key vectors of the sequence for all KV heads
  std::vector<f16_t> vb = float_array_to_half({
    1., 0., 0., // t=0
    0., 1., 0., // t=1
    0., 0., 1., // t=2
    -1., 0., 0. // t=3
  }); // (kv_len, n_kv_heads, head_dim) - buffer containing value vectors of the sequence for all KV heads

  // Multihead attention. Iterate over all heads.
  int q_per_kv_head = TEST_N_HEADS / TEST_N_KV_HEADS; // query heads per kv head (for MultiQueryAttention/GroupedQueryAttention)
  int h;
#pragma omp parallel for private(h)
  for (h = 0; h < TEST_N_HEADS; h++) {
    int kv_head_offset = (h / q_per_kv_head) * TEST_HEAD_DIM;
    f16_t* kh = kb.data() + kv_head_offset;
    f16_t* vh = vb.data() + kv_head_offset;
    attn(s.xb(h), s.att(h), s.q(h), kh, vh, TEST_HEAD_DIM, TEST_N_KV_HEADS, TEST_SEQ_LEN);
  }
  // attention scores
  // h=0
  assertArrayEquals(s.att(0), {
    0., 1., 0., 0.
  }, "att(h=0)");
  // h=1
  assertArrayEquals(s.att(1), {
    0., 0., 1., 0.
  }, "att(h=1)");
  assertArrayEquals(s.xb(), {
    0., 1., 0., // h=0
    0., 0., 1. // h=1
  }, "xout");
}

void fill_random(float* data, size_t N, unsigned long seed, float scale_factor = 1.0) {
  std::default_random_engine gen(seed);
  std::normal_distribution<float> dist(0.0, 1.0);
  for (size_t i = 0; i < N; i++) {
    data[i] = dist(gen) * scale_factor;
  }
}

void fill_random(f16_t* data, size_t N, unsigned long seed, float scale_factor = 1.0) {
#if defined(__AVX2__) && defined(__F16C__)
  std::default_random_engine gen(seed);
  std::normal_distribution<float> dist(0.0, 1.0);
  for (size_t i = 0; i < N; i++) {
    data[i] = _cvtss_sh(dist(gen) * scale_factor, 0);
  }
#else
  assert(false && "Cannot fill_random due to missing F16C extensions");
#endif
}

// Helper function to allocate aligned memory
float* allocateAlignedArray(size_t N) {
  // Allocate aligned memory (64-byte alignment for AVX-512)
  void* ptr = nullptr;
  if (posix_memalign(&ptr, 64, N * sizeof(float)) != 0) {
    throw std::bad_alloc();
  }
  return static_cast<float*>(ptr);
}

void mem_bench() {
  constexpr size_t N_THREADS = 32;
  constexpr size_t MB_PER_THREAD = 1024;
  constexpr size_t ELS_PER_THREAD = (MB_PER_THREAD * 1024 * 1024) / sizeof(float);
  constexpr size_t N = N_THREADS * ELS_PER_THREAD;

  std::cout << "Using " << N_THREADS << " threads" << std::endl;
  std::cout << "Allocating " << N_THREADS * MB_PER_THREAD << " MB (" << N << " floats)" << std::endl;
  float* data = allocateAlignedArray(N);

  std::cout << "Filling data..." << std::endl;
#pragma omp parallel for num_threads(N_THREADS)
  for (size_t i = 0; i < N_THREADS; i++) {
    fill_random(data + i * ELS_PER_THREAD, ELS_PER_THREAD, (unsigned long)i);
  }
  std::cout << "Running memory bandwidth test..." << std::endl;

  float totalSum = 0.0;
  uint64_t start = get_timestamp_ms();
#pragma omp parallel for simd reduction(+:totalSum) schedule(guided) aligned(data: 64) num_threads(N_THREADS)
  for (size_t i = 0; i < N; i++) {
    totalSum += data[i];
  }
    
  uint64_t end = get_timestamp_ms();
  float elapsed_s = (end - start) / 1000.0;
  float mb_per_s = N_THREADS * MB_PER_THREAD / elapsed_s;

  std::cout << "Total sum: " << totalSum << std::endl;
  std::cout << "Elapsed time: " << elapsed_s << " s" << std::endl;
  std::cout << "Memory bandwidth: " << mb_per_s << " MB/s" << std::endl;
}

// 64 is the typical cache line size
struct alignas(64) ThreadData {
  volatile uint32_t sink;
  char padding[60]; // Ensures 64-byte alignment/padding
};

void mem_bench2_thread(uint32_t* data, size_t start_idx, size_t elements_per_thread, ThreadData* thread_sink) {
  for (size_t i = start_idx; i < start_idx + elements_per_thread; i++) {
    // 32-bit load stored in volatile to prevent optimization
    thread_sink->sink = data[i];
  }
}

void mem_bench2() {
  constexpr size_t N_THREADS = 64;
  constexpr size_t MB_PER_THREAD = 2048;
  constexpr size_t ELS_PER_THREAD = (MB_PER_THREAD * 1024 * 1024) / sizeof(uint32_t);
  constexpr size_t N = N_THREADS * ELS_PER_THREAD;

  std::cout << "Using " << N_THREADS << " threads" << std::endl;
  std::cout << "Allocating " << N_THREADS * MB_PER_THREAD << " MB (" << N << " uint32_t)" << std::endl;
  uint32_t* data = new uint32_t[N];

  std::cout << "Filling data..." << std::endl;
#pragma omp parallel for num_threads(N_THREADS)
  for (size_t i = 0; i < N_THREADS; i++) {
    for (size_t j = 0; j < ELS_PER_THREAD; j++) {
      data[i * ELS_PER_THREAD + j] = i + j;
    }
  }
  std::cout << "Running memory bandwidth test..." << std::endl;

  // Allocate cache-line aligned sinks for each thread
  std::vector<ThreadData> thread_sinks(N_THREADS);

  uint64_t start = get_timestamp_ms();
  std::vector<std::thread> threads;
  
  // Launch threads
  for (size_t i = 0; i < N_THREADS; i++) {
    threads.emplace_back(mem_bench2_thread, 
      data,
      i * ELS_PER_THREAD, 
      ELS_PER_THREAD,
      &thread_sinks[i]
    );
  }
  
  // Wait for all threads to complete
  for (auto& thread : threads) {
    thread.join();
  }
    
  uint64_t end = get_timestamp_ms();
  float elapsed_s = (end - start) / 1000.0;
  float mb_per_s = N_THREADS * MB_PER_THREAD / elapsed_s;

  std::cout << "Elapsed time: " << elapsed_s << " s" << std::endl;
  std::cout << "Memory bandwidth: " << mb_per_s << " MB/s" << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc == 2 && std::string(argv[1]) == "-b") {
    std::cout << "Running memory benchmark" << std::endl;
    mem_bench();
  } else if (argc == 2 && std::string(argv[1]) == "-b2") {
    std::cout << "Running memory benchmark 2" << std::endl;
    mem_bench2();
  } else {
    test_attn();
  }
  std::cout << "All tests passed" << std::endl;
  return 0;
}