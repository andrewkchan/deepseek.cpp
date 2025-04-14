/*
K-quants adapted from llama.cpp

MIT License

Copyright (c) 2023-2024 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#include "quant.h"

static inline int nearest_int(float fval) {
  assert(fabsf(fval) <= 4194303.f);
  float val = fval + 12582912.f;
  int i; memcpy(&i, &val, sizeof(int));
  return (i & 0x007fffff) - 0x00400000;
}

static float make_qkx2_quants(int n, int nmax, const float * __restrict__ x, const float * __restrict__ weights,
    uint8_t * __restrict__ L, float * __restrict__ the_min, uint8_t * __restrict__ Laux,
    float rmin, float rdelta, int nstep, bool use_mad) {
  float min = x[0];
  float max = x[0];
  float sum_w = weights[0];
  float sum_x = sum_w * x[0];
#ifdef HAVE_BUGGY_APPLE_LINKER
  // use 'volatile' to prevent unroll and work around a bug in Apple ld64 1015.7
  for (volatile int i = 1; i < n; ++i) {
#else
  for (int i = 1; i < n; ++i) {
#endif
    if (x[i] < min) min = x[i];
    if (x[i] > max) max = x[i];
    float w = weights[i];
    sum_w += w;
    sum_x += w * x[i];
  }
  if (min > 0) min = 0;
  if (max == min) {
    for (int i = 0; i < n; ++i) L[i] = 0;
    *the_min = -min;
    return 0.f;
  }
  float iscale = nmax/(max - min);
  float scale = 1/iscale;
  float best_mad = 0;
  for (int i = 0; i < n; ++i) {
    int l = nearest_int(iscale*(x[i] - min));
    L[i] = std::max(0, std::min(nmax, l));
    float diff = scale * L[i] + min - x[i];
    diff = use_mad ? fabsf(diff) : diff * diff;
    float w = weights[i];
    best_mad += w * diff;
  }
  if (nstep < 1) {
    *the_min = -min;
    return scale;
  }
  for (int is = 0; is <= nstep; ++is) {
    iscale = (rmin + rdelta*is + nmax)/(max - min);
    float sum_l = 0, sum_l2 = 0, sum_xl = 0;
    for (int i = 0; i < n; ++i) {
      int l = nearest_int(iscale*(x[i] - min));
      l = std::max(0, std::min(nmax, l));
      Laux[i] = l;
      float w = weights[i];
      sum_l += w*l;
      sum_l2 += w*l*l;
      sum_xl += w*l*x[i];
    }
    float D = sum_w * sum_l2 - sum_l * sum_l;
    if (D > 0) {
      float this_scale = (sum_w * sum_xl - sum_x * sum_l)/D;
      float this_min   = (sum_l2 * sum_x - sum_l * sum_xl)/D;
      if (this_min > 0) {
        this_min = 0;
        this_scale = sum_xl / sum_l2;
      }
      float mad = 0;
      for (int i = 0; i < n; ++i) {
        float diff = this_scale * Laux[i] + this_min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        float w = weights[i];
        mad += w * diff;
      }
      if (mad < best_mad) {
        for (int i = 0; i < n; ++i) {
          L[i] = Laux[i];
        }
        best_mad = mad;
        scale = this_scale;
        min = this_min;
      }
    }
  }
  *the_min = -min;
  return scale;
}

void quantize_row_q2_K_ref(const float * __restrict__ x, block_q2_K * __restrict__ y, int64_t k) {
  assert(k % QK_K == 0);
  const int nb = k / QK_K;

  uint8_t L[QK_K];
  uint8_t Laux[16];
  float   weights[16];
  float mins[QK_K/16];
  float scales[QK_K/16];

  const float q4scale = 15.f;

  for (int i = 0; i < nb; i++) {
    float max_scale = 0; // as we are deducting the min, scales are always positive
    float max_min = 0;
    for (int j = 0; j < QK_K/16; ++j) {
      for (int l = 0; l < 16; ++l) weights[l] = fabsf(x[16*j + l]);
      scales[j] = make_qkx2_quants(16, 3, x + 16*j, weights, L + 16*j, &mins[j], Laux, -0.5f, 0.1f, 15, true);
      float scale = scales[j];
      if (scale > max_scale) {
        max_scale = scale;
      }
      float min = mins[j];
      if (min > max_min) {
        max_min = min;
      }
    }

    if (max_scale > 0) {
      float iscale = q4scale/max_scale;
      for (int j = 0; j < QK_K/16; ++j) {
        int l = nearest_int(iscale*scales[j]);
        y[i].scales[j] = l;
      }
      y[i].d = float_to_half(max_scale/q4scale);
    } else {
      for (int j = 0; j < QK_K/16; ++j) y[i].scales[j] = 0;
      y[i].d = float_to_half(0.f);
    }
    if (max_min > 0) {
      float iscale = q4scale/max_min;
      for (int j = 0; j < QK_K/16; ++j) {
        int l = nearest_int(iscale*mins[j]);
        y[i].scales[j] |= (l << 4);
      }
      y[i].dmin = float_to_half(max_min/q4scale);
    } else {
      y[i].dmin = float_to_half(0.f);
    }
    for (int j = 0; j < QK_K/16; ++j) {
      const float d = half_to_float(y[i].d) * (y[i].scales[j] & 0xF);
      if (!d) continue;
      const float dm = half_to_float(y[i].dmin) * (y[i].scales[j] >> 4);
      for (int ii = 0; ii < 16; ++ii) {
        int l = nearest_int((x[16*j + ii] + dm)/d);
        l = std::max(0, std::min(3, l));
        L[16*j + ii] = l;
      }
    }

    for (int j = 0; j < QK_K; j += 128) {
      for (int l = 0; l < 32; ++l) {
        y[i].qs[j/4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
      }
    }

    x += QK_K;
  }
}

void dequantize_row_q2_K(const block_q2_K * __restrict__ x, float * __restrict__ y, int64_t k) {
  assert(k % QK_K == 0);
  const int nb = k / QK_K;

  for (int i = 0; i < nb; i++) {

    const float d = half_to_float(x[i].d);
    const float min = half_to_float(x[i].dmin);

    const uint8_t * q = x[i].qs;

    int is = 0;
    float dl, ml;
    for (int n = 0; n < QK_K; n += 128) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {

        uint8_t sc = x[i].scales[is++];
        dl = d * (sc & 0xF); ml = min * (sc >> 4);
        for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;

        sc = x[i].scales[is++];
        dl = d * (sc & 0xF); ml = min * (sc >> 4);
        for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3)) - ml;

        shift += 2;
      }
      q += 32;
    }
  }
}