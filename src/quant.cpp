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

#include <cassert>

#define GROUP_MAX_EPS 1e-15f

static inline int nearest_int(float fval) {
  assert(fabsf(fval) <= 4194303.f);
  float val = fval + 12582912.f;
  int i; memcpy(&i, &val, sizeof(int));
  return (i & 0x007fffff) - 0x00400000;
}

// some compilers don't provide _mm256_set_m128i, e.g. gcc 7
#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

#if __AVX__ || __AVX2__ || __AVX512F__
// horizontally add 8 floats
static inline float hsum_float_8(const __m256 x) {
  __m128 res = _mm256_extractf128_ps(x, 1);
  res = _mm_add_ps(res, _mm256_castps256_ps128(x));
  res = _mm_add_ps(res, _mm_movehl_ps(res, res));
  res = _mm_add_ss(res, _mm_movehdup_ps(res));
  return _mm_cvtss_f32(res);
}

// shuffles to pick the required scales in dot products
static inline __m256i get_scale_shuffle_q3k(int i) {
  static const uint8_t k_shuffle[128] = {
      0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
      4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,     6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
      8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
    12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
  };
  return _mm256_loadu_si256((const __m256i*)k_shuffle + i);
}
#endif

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

static float make_q3_quants(int n, int nmax, const float * __restrict__ x, int8_t * __restrict__ L, bool do_rmse) {
  float max = 0;
  float amax = 0;
  for (int i = 0; i < n; ++i) {
    float ax = fabsf(x[i]);
    if (ax > amax) { amax = ax; max = x[i]; }
  }
  if (amax < GROUP_MAX_EPS) { // all zero
    for (int i = 0; i < n; ++i) { L[i] = 0; }
    return 0.f;
  }
  float iscale = -nmax / max;
  if (do_rmse) {
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
      int l = nearest_int(iscale * x[i]);
      l = std::max(-nmax, std::min(nmax-1, l));
      L[i] = l;
      float w = x[i]*x[i];
      sumlx += w*x[i]*l;
      suml2 += w*l*l;
    }
    for (int itry = 0; itry < 5; ++itry) {
      int n_changed = 0;
      for (int i = 0; i < n; ++i) {
        float w = x[i]*x[i];
        float slx = sumlx - w*x[i]*L[i];
        if (slx > 0) {
          float sl2 = suml2 - w*L[i]*L[i];
          int new_l = nearest_int(x[i] * sl2 / slx);
          new_l = std::max(-nmax, std::min(nmax-1, new_l));
          if (new_l != L[i]) {
            slx += w*x[i]*new_l;
            sl2 += w*new_l*new_l;
            if (sl2 > 0 && slx*slx*suml2 > sumlx*sumlx*sl2) {
              L[i] = new_l; sumlx = slx; suml2 = sl2;
              ++n_changed;
            }
          }
        }
      }
      if (!n_changed) {
        break;
      }
    }
    for (int i = 0; i < n; ++i) {
        L[i] += nmax;
    }
    return sumlx / suml2;
  }
  for (int i = 0; i < n; ++i) {
      int l = nearest_int(iscale * x[i]);
      l = std::max(-nmax, std::min(nmax-1, l));
      L[i] = l + nmax;
  }
  return 1/iscale;
}

void quantize_row_q3_K_ref(const float * __restrict__ x, block_q3_K * __restrict__ y, int64_t k) {
  assert(k % QK_K == 0);
  const int nb = k / QK_K;

  int8_t L[QK_K];
  float scales[QK_K / 16];

  for (int i = 0; i < nb; i++) {

    float max_scale = 0;
    float amax = 0;
    for (int j = 0; j < QK_K/16; ++j) {
      scales[j] = make_q3_quants(16, 4, x + 16*j, L + 16*j, true);
      float scale = fabsf(scales[j]);
      if (scale > amax) {
        amax = scale; max_scale = scales[j];
      }
    }

    memset(y[i].scales, 0, 12);
    if (max_scale) {
      float iscale = -32.f/max_scale;
      for (int j = 0; j < QK_K/16; ++j) {
        int8_t l = nearest_int(iscale*scales[j]);
        l = std::max(-32, std::min(31, static_cast<int>(l))) + 32;
        if (j < 8) {
          y[i].scales[j] = l & 0xF;
        } else {
          y[i].scales[j-8] |= ((l & 0xF) << 4);
        }
        l >>= 4;
        y[i].scales[j%4 + 8] |= (l << (2*(j/4)));
      }
      y[i].d = float_to_half(1/iscale);
    } else {
      y[i].d = float_to_half(0.f);
    }

    int8_t sc;
    for (int j = 0; j < QK_K/16; ++j) {
      sc = j < 8 ? y[i].scales[j] & 0xF : y[i].scales[j-8] >> 4;
      sc = (sc | (((y[i].scales[8 + j%4] >> (2*(j/4))) & 3) << 4)) - 32;
      float d = half_to_float(y[i].d) * sc;
      if (!d) {
        continue;
      }
      for (int ii = 0; ii < 16; ++ii) {
        int l = nearest_int(x[16*j + ii]/d);
        l = std::max(-4, std::min(3, l));
        L[16*j + ii] = l + 4;
      }
    }

    memset(y[i].hmask, 0, QK_K/8);
    // We put the high-bit for the 1st 8 quants into bit 0, the next 8 into bit 1, etc.
    int m = 0;
    uint8_t hm = 1;
    for (int j = 0; j < QK_K; ++j) {
      if (L[j] > 3) {
        y[i].hmask[m] |= hm;
        L[j] -= 4;
      }
      if (++m == QK_K/8) {
        m = 0; hm <<= 1;
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

void dequantize_row_q3_K(const block_q3_K * __restrict__ x, float * __restrict__ y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    uint32_t aux[4];
    const int8_t * scales = (const int8_t*)aux;

    for (int i = 0; i < nb; i++) {

        const float d_all = half_to_float(x[i].d);

        const uint8_t * __restrict__ q = x[i].qs;
        const uint8_t * __restrict__ hm = x[i].hmask;
        uint8_t m = 1;

        memcpy(aux, x[i].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        int is = 0;
        float dl;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {

                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l+ 0] >> shift) & 3) - ((hm[l+ 0] & m) ? 0 : 4));
                }

                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3) - ((hm[l+16] & m) ? 0 : 4));
                }

                shift += 2;
                m <<= 1;
            }
            q += 32;
        }

    }
}

void ggml_vec_dot_q3_K_q8_K(int n, float * __restrict__ s, const void * __restrict__ vx, const void * __restrict__ vy) {
  assert(n % QK_K == 0);

  const uint32_t kmask1 = 0x03030303;
  const uint32_t kmask2 = 0x0f0f0f0f;

  const block_q3_K * __restrict__ x = (const block_q3_K*)vx;
  const block_q8_K * __restrict__ y = (const block_q8_K*)vy;

  const int nb = n / QK_K;

#if defined __AVX2__

  const __m256i m3 = _mm256_set1_epi8(3);
  const __m256i mone = _mm256_set1_epi8(1);
  const __m128i m32 = _mm_set1_epi8(32);

  __m256 acc = _mm256_setzero_ps();

  uint32_t aux[3];

  for (int i = 0; i < nb; ++i) {

    const float d = y[i].d * half_to_float(x[i].d);

    const uint8_t * __restrict__ q3 = x[i].qs;
    const int8_t  * __restrict__ q8 = y[i].qs;

    // Set up scales
    memcpy(aux, x[i].scales, 12);
    __m128i scales128 = _mm_set_epi32(
            ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4),
            ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4),
            (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4),
            (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4));
    scales128 = _mm_sub_epi8(scales128, m32);
    const __m256i all_scales = _mm256_cvtepi8_epi16(scales128);
    const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
    const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
    const __m256i scales[2] = {MM256_SET_M128I(l_scales, l_scales), MM256_SET_M128I(h_scales, h_scales)};

    // high bit
    const __m256i hbits = _mm256_loadu_si256((const __m256i*)x[i].hmask);

    // integer accumulator
    __m256i sumi = _mm256_setzero_si256();

    int bit = 0;
    int is  = 0;

    for (int j = 0; j < QK_K/128; ++j) {
      // load low 2 bits
      const __m256i q3bits = _mm256_loadu_si256((const __m256i*)q3); q3 += 32;

      // prepare low and high bits
      const __m256i q3l_0 = _mm256_and_si256(q3bits, m3);
      const __m256i q3h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
      ++bit;

      const __m256i q3l_1 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 2), m3);
      const __m256i q3h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
      ++bit;

      const __m256i q3l_2 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 4), m3);
      const __m256i q3h_2 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
      ++bit;

      const __m256i q3l_3 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 6), m3);
      const __m256i q3h_3 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
      ++bit;

      // load Q8 quants
      const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
      const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
      const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
      const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

      // Dot product: we multiply the 2 low bits and 1 high bit part separately, so we can use _mm256_maddubs_epi16,
      // and then subtract. The high bit part has the 2 already subtracted (and so, it is zero if the high bit was not set,
      // and 2 if the high bit was set)
      __m256i q8s_0 = _mm256_maddubs_epi16(q3h_0, q8_0);
      __m256i q8s_1 = _mm256_maddubs_epi16(q3h_1, q8_1);
      __m256i q8s_2 = _mm256_maddubs_epi16(q3h_2, q8_2);
      __m256i q8s_3 = _mm256_maddubs_epi16(q3h_3, q8_3);

      __m256i p16_0 = _mm256_maddubs_epi16(q3l_0, q8_0);
      __m256i p16_1 = _mm256_maddubs_epi16(q3l_1, q8_1);
      __m256i p16_2 = _mm256_maddubs_epi16(q3l_2, q8_2);
      __m256i p16_3 = _mm256_maddubs_epi16(q3l_3, q8_3);

      p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
      p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
      p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
      p16_3 = _mm256_sub_epi16(p16_3, q8s_3);

      // multiply with scales
      p16_0 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 0)), p16_0);
      p16_1 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 1)), p16_1);
      p16_2 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 2)), p16_2);
      p16_3 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 3)), p16_3);

      // accumulate
      p16_0 = _mm256_add_epi32(p16_0, p16_1);
      p16_2 = _mm256_add_epi32(p16_2, p16_3);
      sumi  = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_2));

    }

    // multiply with block scale and accumulate
    acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);

  }

  *s = hsum_float_8(acc);

#else
  // scalar version
  // This function is written like this so the compiler can manage to vectorize most of it
  // Using -Ofast, GCC and clang manage to produce code that is within a factor of 2 or so from the
  // manually vectorized version above. Every other version I tried would run at least 4 times slower.
  // The ideal situation would be if we could just write the code once, and the compiler would
  // automatically produce the best possible set of machine instructions, instead of us having to manually
  // write vectorized versions for AVX, ARM_NEON, etc.

  int8_t  aux8[QK_K];
  int16_t aux16[8];
  float   sums [8];
  int32_t aux32[8];
  memset(sums, 0, 8*sizeof(float));

  uint32_t auxs[4];
  const int8_t * scales = (const int8_t*)auxs;

  float sumf = 0;
  for (int i = 0; i < nb; ++i) {
    const uint8_t * __restrict__ q3 = x[i].qs;
    const uint8_t * __restrict__ hm = x[i].hmask;
    const  int8_t * __restrict__ q8 = y[i].qs;
    memset(aux32, 0, 8*sizeof(int32_t));
    int8_t * __restrict__ a = aux8;
    uint8_t m = 1;
    for (int j = 0; j < QK_K; j += 128) {
      for (int l = 0; l < 32; ++l) a[l] = q3[l] & 3;
      for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
      a += 32; m <<= 1;
      for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 2) & 3;
      for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
      a += 32; m <<= 1;
      for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 4) & 3;
      for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
      a += 32; m <<= 1;
      for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 6) & 3;
      for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
      a += 32; m <<= 1;
      q3 += 32;
    }
    a = aux8;

    memcpy(auxs, x[i].scales, 12);
    uint32_t tmp = auxs[2];
    auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    for (int j = 0; j < QK_K/16; ++j) {
      for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
      for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
      q8 += 8; a += 8;
      for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
      for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
      q8 += 8; a += 8;
    }
    const float d = half_to_float(x[i].d) * y[i].d;
    for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
  }
  for (int l = 0; l < 8; ++l) sumf += sums[l];
  *s = sumf;

#endif

}

void quantize_row_q8_K_ref(const float * __restrict__ x, block_q8_K * __restrict__ y, int64_t k) {
  assert(k % QK_K == 0);
  const int64_t nb = k / QK_K;

  for (int i = 0; i < nb; i++) {

    float max = 0;
    float amax = 0;
    for (int j = 0; j < QK_K; ++j) {
      float ax = fabsf(x[j]);
      if (ax > amax) {
        amax = ax; max = x[j];
      }
    }
    if (!amax) {
      y[i].d = 0;
      memset(y[i].qs, 0, QK_K);
      x += QK_K;
      continue;
    }
    //const float iscale = -128.f/max;
    // We need this change for IQ2_XXS, else the AVX implementation becomes very awkward
    const float iscale = -127.f/max;
    for (int j = 0; j < QK_K; ++j) {
      int v = nearest_int(iscale*x[j]);
      y[i].qs[j] = std::min(127, v);
    }
    for (int j = 0; j < QK_K/16; ++j) {
      int sum = 0;
      for (int ii = 0; ii < 16; ++ii) {
        sum += y[i].qs[j*16 + ii];
      }
      y[i].bsums[j] = sum;
    }
    y[i].d = 1/iscale;
    x += QK_K;
  }
}

void dequantize_row_q8_K(const block_q8_K * __restrict__ x, float * __restrict__ y, int64_t k) {
  assert(k % QK_K == 0);
  const int64_t nb = k / QK_K;

  for (int i = 0; i < nb; i++) {
    for (int j = 0; j < QK_K; ++j) {
      *y++ = x[i].d * x[i].qs[j];
    }
  }
}

void ggml_vec_dot_q2_K_q8_K(
  int n, 
  float * __restrict__ s, 
  const void * __restrict__ vx, 
  const void * __restrict__ vy
) {

  const block_q2_K * __restrict__ x = (const block_q2_K *)vx;
  const block_q8_K * __restrict__ y = (const block_q8_K *)vy;

  const int nb = n / QK_K;

#if defined __AVX2__

  const __m256i m3 = _mm256_set1_epi8(3);
  const __m128i m4 = _mm_set1_epi8(0xF);

  __m256 acc = _mm256_setzero_ps();

  for (int i = 0; i < nb; ++i) {

    const float d = y[i].d * half_to_float(x[i].d);
    const float dmin = -y[i].d * half_to_float(x[i].dmin);

    const uint8_t * __restrict__ q2 = x[i].qs;
    const int8_t  * __restrict__ q8 = y[i].qs;

    const __m128i mins_and_scales = _mm_loadu_si128((const __m128i*)x[i].scales);
    const __m128i scales8 = _mm_and_si128(mins_and_scales, m4);
    const __m128i mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
    const __m256i mins = _mm256_cvtepi8_epi16(mins8);
    const __m256i prod = _mm256_madd_epi16(mins, _mm256_loadu_si256((const __m256i*)y[i].bsums));

    acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&dmin), _mm256_cvtepi32_ps(prod), acc);

    const __m256i all_scales = _mm256_cvtepi8_epi16(scales8);
    const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
    const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
    const __m256i scales[2] = {MM256_SET_M128I(l_scales, l_scales), MM256_SET_M128I(h_scales, h_scales)};

    __m256i sumi = _mm256_setzero_si256();

    for (int j = 0; j < QK_K/128; ++j) {

      const __m256i q2bits = _mm256_loadu_si256((const __m256i*)q2); q2 += 32;

      const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
      const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
      const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
      const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

      const __m256i q2_0 = _mm256_and_si256(q2bits, m3);
      const __m256i q2_1 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 2), m3);
      const __m256i q2_2 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 4), m3);
      const __m256i q2_3 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 6), m3);

      __m256i p0 = _mm256_maddubs_epi16(q2_0, q8_0);
      __m256i p1 = _mm256_maddubs_epi16(q2_1, q8_1);
      __m256i p2 = _mm256_maddubs_epi16(q2_2, q8_2);
      __m256i p3 = _mm256_maddubs_epi16(q2_3, q8_3);

      p0 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(0)), p0);
      p1 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(1)), p1);
      p2 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(2)), p2);
      p3 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(3)), p3);

      p0 = _mm256_add_epi32(p0, p1);
      p2 = _mm256_add_epi32(p2, p3);

      sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p0, p2));
    }

    acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);

  }

  *s = hsum_float_8(acc);
#else
  float sumf = 0;

  for (int i = 0; i < nb; ++i) {

    const uint8_t * q2 = x[i].qs;
    const  int8_t * q8 = y[i].qs;
    const uint8_t * sc = x[i].scales;

    int summs = 0;
    for (int j = 0; j < 16; ++j) {
      summs += y[i].bsums[j] * (sc[j] >> 4);
    }

    const float dall = y[i].d * half_to_float(x[i].d);
    const float dmin = y[i].d * half_to_float(x[i].dmin);

    int isum = 0;
    int is = 0;
    int d;
    for (int k = 0; k < QK_K/128; ++k) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        d = sc[is++] & 0xF;
        int isuml = 0;
        for (int l =  0; l < 16; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
        isum += d * isuml;
        d = sc[is++] & 0xF;
        isuml = 0;
        for (int l = 16; l < 32; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
        isum += d * isuml;
        shift += 2;
        q8 += 32;
      }
      q2 += 32;
    }
    sumf += dall * isum - dmin * summs;
  }
  *s = sumf;
#endif
}