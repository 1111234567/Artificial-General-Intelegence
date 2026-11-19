// 96-row batched FP16 matrix-vector operation with FP32 Tensor Core accumulation.
using half_bits = unsigned short;

__device__ __forceinline__ half_bits rounded_half(float value) {
    half_bits bits;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(bits) : "f"(value));
    return bits;
}

struct TensorRowPair {
    float first, second;
};

template<int row_words = 48>
__device__ __forceinline__ TensorRowPair multiply_row_pair(
    const unsigned* matrix, const unsigned* vector, int warp, int lane
) {
    const int group = lane / 4;
    const int pair = lane % 4;
    const int row = warp * 16 + group;
    const unsigned* upper = matrix + row * row_words + pair;
    const unsigned* lower = upper + 8 * row_words;
    float accum0 = 0, accum1 = 0, accum2 = 0, accum3 = 0;
    #pragma unroll
    for (int k = 0; k < 48; k += 8) {
        const unsigned a0 = upper[k], a1 = lower[k], a2 = upper[k + 4], a3 = lower[k + 4];
        const unsigned b0 = vector[k + pair], b1 = vector[k + pair + 4];
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(accum0), "+f"(accum1), "+f"(accum2), "+f"(accum3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
        );
    }
    return {accum0, accum2};
}

template<int row_words = 48>
__device__ __forceinline__ void multiply_rows(
    const unsigned* matrix, const unsigned* vector, half_bits* result, int warp, int lane
) {
    const auto product = multiply_row_pair<row_words>(matrix, vector, warp, lane);
    if (lane % 4 == 0) {
        const int row = warp * 16 + lane / 4;
        result[row] = rounded_half(product.first);
        result[row + 8] = rounded_half(product.second);
    }
}

extern "C" __global__ void directed_tensor_matvec(
    const unsigned* __restrict__ matrix, const unsigned* __restrict__ vector,
    half_bits* __restrict__ result
) {
    const unsigned actor = blockIdx.x;
    multiply_rows(matrix + actor * 96 * 48, vector + actor * 48,
                  result + actor * 96, threadIdx.x / 32, threadIdx.x % 32);
}
