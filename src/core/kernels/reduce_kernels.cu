#include "core/kernels/kernels.h"
#include "core/common/cuda/cuda_check.h"
#include <cfloat>
#include <climits>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace hp::core::kernels {

// --- ArgMax ---

// Shared-memory parallel reduction argmax: one block per sequence position
// 256 threads cooperate to find max across the class dim (v6 width 18,710;
// tiny tier 6,906). Class-agnostic — the reduction loop strides num_classes.
__global__ __launch_bounds__(256)
void argmax_kernel(const float *input_probs, int *output_indices,
                              float *output_scores, int total_steps,
                              int num_classes, const uint8_t *class_ok) {
  int step_idx = blockIdx.x;
  if (step_idx >= total_steps)
    return;

  const float *probs_ptr = input_probs + step_idx * num_classes;
  int tid = threadIdx.x;
  int block_size = blockDim.x; // 256

  // Each thread finds local max over its stripe
  float local_max = -FLT_MAX;
  int local_idx = 0;
  for (int c = tid; c < num_classes; c += block_size) {
    if (class_ok && !class_ok[c]) continue;  // class excluded for this page's script
    float val = probs_ptr[c];
    if (val > local_max) {
      local_max = val;
      local_idx = c;
    }
  }

  // Shared memory reduction (strides > 32)
  __shared__ float s_vals[256];
  __shared__ int s_idxs[256];
  s_vals[tid] = local_max;
  s_idxs[tid] = local_idx;
  __syncthreads();

  // Every reduction breaks an exact-value tie toward the LOWER class index
  // (not the lower thread id), so the GPU argmax is deterministic and matches
  // the CPU/AVX2 CTC reference (ctc_decode.cpp), which keeps the lowest index.
  for (int stride = block_size / 2; stride > 32; stride >>= 1) {
    if (tid < stride) {
      if (s_vals[tid + stride] > s_vals[tid] ||
          (s_vals[tid + stride] == s_vals[tid] && s_idxs[tid + stride] < s_idxs[tid])) {
        s_vals[tid] = s_vals[tid + stride];
        s_idxs[tid] = s_idxs[tid + stride];
      }
    }
    __syncthreads();
  }

  // Warp-level reduction (strides <= 32, no syncthreads needed)
  if (tid < 32) {
    // Load from shared to registers
    float wval = s_vals[tid];
    int widx = s_idxs[tid];
    if (s_vals[tid + 32] > wval ||
        (s_vals[tid + 32] == wval && s_idxs[tid + 32] < widx)) {
      wval = s_vals[tid + 32]; widx = s_idxs[tid + 32];
    }

    // Warp shuffle reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
      float other_val = __shfl_down_sync(0xffffffff, wval, offset);
      int other_idx = __shfl_down_sync(0xffffffff, widx, offset);
      if (other_val > wval || (other_val == wval && other_idx < widx)) {
        wval = other_val; widx = other_idx;
      }
    }

    if (tid == 0) {
      output_indices[step_idx] = widx;
      output_scores[step_idx] = wval;
    }
  }
}

void cuda_argmax(const float *input_probs, int *output_indices,
                 float *output_scores, int batch_size, int seq_len,
                 int num_classes, cudaStream_t stream, const uint8_t *class_ok) {
  int total_steps = batch_size * seq_len;
  // One block per step, 256 threads per block for parallel reduction
  argmax_kernel<<<total_steps, 256, 0, stream>>>(
      input_probs, output_indices, output_scores, total_steps, num_classes, class_ok);
  CUDA_CHECK(cudaGetLastError());
}

// --- ArgMax + softmax score from FP16 logits ---

namespace {

struct SoftArg {
  float m;    // running max over all classes
  float s;    // sum of exp(l - m) over all classes
  float bv;   // best allowed logit
  int bi;     // its class (lowest on ties)
};

__device__ __forceinline__ SoftArg merge(SoftArg a, const SoftArg &b) {
  const float M = fmaxf(a.m, b.m);
  float s = 0.f;
  if (a.m != -INFINITY) s += a.s * __expf(a.m - M);
  if (b.m != -INFINITY) s += b.s * __expf(b.m - M);
  a.m = M;
  a.s = s;
  if (b.bv > a.bv || (b.bv == a.bv && b.bi < a.bi)) {
    a.bv = b.bv;
    a.bi = b.bi;
  }
  return a;
}

__device__ __forceinline__ SoftArg shfl_down(const SoftArg &v, int off) {
  return {__shfl_down_sync(0xffffffffu, v.m, off), __shfl_down_sync(0xffffffffu, v.s, off),
          __shfl_down_sync(0xffffffffu, v.bv, off), __shfl_down_sync(0xffffffffu, v.bi, off)};
}

}  // namespace

// One block per row. Each thread runs an online softmax (max, sum) over its
// stride of classes and keeps the best allowed class; warps then the block merge.
__global__ __launch_bounds__(256)
void argmax_logits_kernel(const __half *__restrict__ logits, int *__restrict__ out_idx,
                          float *__restrict__ out_score, int rows, int num_classes,
                          const uint8_t *__restrict__ class_ok) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const __half *r = logits + (size_t)row * num_classes;
  SoftArg a{-INFINITY, 0.f, -INFINITY, INT_MAX};
  for (int c = threadIdx.x; c < num_classes; c += blockDim.x) {
    const float v = __half2float(r[c]);
    if (v > a.m) {
      a.s = a.s * __expf(a.m - v) + 1.f;
      a.m = v;
    } else {
      a.s += __expf(v - a.m);
    }
    if ((!class_ok || class_ok[c]) && v > a.bv) {  // ascending c: strict > keeps the lowest index
      a.bv = v;
      a.bi = c;
    }
  }
  for (int off = 16; off > 0; off >>= 1) a = merge(a, shfl_down(a, off));
  __shared__ SoftArg warp_res[8];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) warp_res[warp] = a;
  __syncthreads();
  if (warp == 0) {
    const int nw = blockDim.x >> 5;
    a = lane < nw ? warp_res[lane] : SoftArg{-INFINITY, 0.f, -INFINITY, INT_MAX};
    for (int off = 16; off > 0; off >>= 1) a = merge(a, shfl_down(a, off));
    if (lane == 0) {
      out_idx[row] = a.bi == INT_MAX ? 0 : a.bi;
      out_score[row] = a.bi == INT_MAX ? 0.f : __expf(a.bv - a.m) / a.s;
    }
  }
}

void cuda_argmax_logits(const uint16_t *logits_f16, int *output_indices, float *output_scores, int rows,
                        int num_classes, cudaStream_t stream, const uint8_t *class_ok) {
  if (rows <= 0) return;
  argmax_logits_kernel<<<rows, 256, 0, stream>>>(reinterpret_cast<const __half *>(logits_f16), output_indices,
                                                 output_scores, rows, num_classes, class_ok);
  CUDA_CHECK(cudaGetLastError());
}

// --- Fused Threshold + float->uint8 (vectorized float4 loads) ---

__global__ __launch_bounds__(256)
void threshold_to_u8_kernel(const float *src, uint8_t *dst, int w,
                                       int h, float thresh) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  int total = w * h;

  if (idx + 3 < total) {
    // Vectorized path: load 4 floats at once via float4
    float4 vals = *reinterpret_cast<const float4*>(src + idx);
    uint8_t r0 = (vals.x > thresh) ? 255 : 0;
    uint8_t r1 = (vals.y > thresh) ? 255 : 0;
    uint8_t r2 = (vals.z > thresh) ? 255 : 0;
    uint8_t r3 = (vals.w > thresh) ? 255 : 0;
    // Pack 4 bytes into one uint32 store
    *reinterpret_cast<uint32_t*>(dst + idx) =
        (uint32_t)r0 | ((uint32_t)r1 << 8) | ((uint32_t)r2 << 16) | ((uint32_t)r3 << 24);
  } else {
    // Tail elements: scalar fallback
    for (int i = idx; i < min(idx + 4, total); i++)
      dst[i] = (src[i] > thresh) ? 255 : 0;
  }
}

void cuda_threshold_to_u8(const float *src, uint8_t *dst, int w, int h,
                           float thresh, cudaStream_t stream) {
  int total = w * h;
  int threads = 256;
  // Each thread processes 4 elements, so divide by 4
  int elements_per_block = threads * 4;
  int blocks = (total + elements_per_block - 1) / elements_per_block;

  threshold_to_u8_kernel<<<blocks, threads, 0, stream>>>(src, dst, w, h,
                                                          thresh);
  CUDA_CHECK(cudaGetLastError());
}

void cuda_batch_threshold_to_u8(const float *src, uint8_t *dst, int w, int h,
                                 int batch_size, float thresh,
                                 cudaStream_t stream) {
  // Treat as flat 1D array: pass total as (w_flat, 1) so kernel sees total = w_flat * 1.
  // Guard the size_t->int narrowing: w*h*batch as int would overflow once
  // per-side limits approach 16384 at batch 8, silently misindexing the
  // threshold writes. Fail loud instead if limits ever grow that far.
  const size_t total = static_cast<size_t>(w) * h * batch_size;
  if (total > static_cast<size_t>(INT_MAX))
    CUDA_CHECK(cudaErrorInvalidValue);
  const int total_pixels = static_cast<int>(total);
  int threads = 256;
  int elements_per_block = threads * 4;
  int blocks = (total_pixels + elements_per_block - 1) / elements_per_block;

  threshold_to_u8_kernel<<<blocks, threads, 0, stream>>>(src, dst, total_pixels, 1,
                                                          thresh);
  CUDA_CHECK(cudaGetLastError());
}

} // namespace hp::core::kernels
