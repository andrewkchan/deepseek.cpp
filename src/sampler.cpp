#include "sampler.h"

#include <cfloat>
#include <cstdlib>

Sampler::Sampler(const std::shared_ptr<Config> config, uint64_t seed) {
  vocab_size = config->vocab_size;
  std::srand(seed);
}

float Sampler::sample_prob(int index, const InferenceState& s) {
  const float* logits = s.logits();
  // Find max value to moderate the logits later on for numerical stability
  float max_val = -FLT_MAX;
  for (int i = 0; i < vocab_size; ++i) {
    if (logits[i] > max_val) {
      max_val = logits[i];
    }
  }
  float sum = 0;
  for (int i = 0; i < vocab_size; ++i) {
    sum += expf(logits[i] - max_val);
  }
  return expf(logits[index] - max_val) / sum;
}

int Sampler::sample_argmax(const InferenceState& s) {
  const float* logits = s.logits();
  int argmax = 0;
  float max_val = -FLT_MAX;
  for (int i = 0; i < vocab_size; ++i) {
    if (logits[i] > max_val) {
      max_val = logits[i];
      argmax = i;
    }
  }
  return argmax;
}

int Sampler::sample(const InferenceState& s, float temperature) {
  if (temperature == 0.0) {
    return sample_argmax(s);
  }
  const float* logits = s.logits();
  // Find max value to moderate the logits later on for numerical stability
  float max_val = -FLT_MAX;
  for (int i = 0; i < vocab_size; ++i) {
    if (logits[i] > max_val) {
      max_val = logits[i];
    }
  }
  float sum = 0;
  for (int i = 0; i < vocab_size; ++i) {
    sum += expf((logits[i] - max_val) / temperature);
  }
  float r = std::rand() / (float)RAND_MAX;
  float cumsum = 0;
  for (int i = 0; i < vocab_size; ++i) {
    cumsum += expf((logits[i] - max_val) / temperature) / sum;
    if (cumsum >= r) {
      return i;
    }
  }
  return vocab_size - 1;
}

