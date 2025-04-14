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

#pragma once

#include "codec.h"

// QK = number of values after dequantization
// QK_K = super-block size

#define QK_K 256

// 2-bit quantization
// weight is represented as x = a * q + b
// 16 blocks of 16 elements each
// Effectively 2.625 bits per weight
typedef struct {
  uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
  uint8_t qs[QK_K/4];      // quants
  union {
    struct {
      uint16_t d;    // super-block scale for quantized scales
      uint16_t dmin; // super-block scale for quantized mins
    };
    uint32_t dm;
  };
} block_q2_K;

// Quantize an array of weights using Q2_K quantization
// - x: pointer to the weights to quantize
// - y: pointer to the quantized weights
// - k: number of weights to quantize (must be a multiple of QK_K)
void quantize_row_q2_K_ref(const float * __restrict__ x, block_q2_K * __restrict__ y, int64_t k);

// Dequantize an array of Q2_K quantized weights
// - x: pointer to the quantized weights
// - y: pointer to the dequantized weights
// - k: number of weights to dequantize (must be a multiple of QK_K)
void dequantize_row_q2_K(const block_q2_K * __restrict__ x, float * __restrict__ y, int64_t k);