#pragma once

#include <math.h>

#include "multiple.hpp"
#include "platform.hpp"

/*======================================================*/
//        Declarations for supported operations
/*======================================================*/
// NOTE(jongmin): Current SIMD implementations will assume little-endianness any time
// 8-bit operations are involved. e.g. basic arithmetic will work even with
// big-endian architecture using PLATFORM=SCALAR, but operations like
// VNARROW_TO_8BIT and VSHUFFLE may behave unexpectedly. When in doubt, build
// and run the unit tests and see if they pass.
// Also note that while vectorization supports 32-bit integer, if numbers exceed 16-bit integer,
// things may misbehave wildly during multiplication.
#define SFLOAT typename InstructionSetTrait<S>::FLOAT
#define SUINT8 typename InstructionSetTrait<S>::UINT8
#define SINT32 typename InstructionSetTrait<S>::INT32

namespace lopper {
  template<typename T> T VADD(T op1, T op2);
  template<typename T> T VADD_PAIRWISE(T op1, T op2); // pairwise add
  template<typename T> T VSUB(T op1, T op2);
  template<typename T> T VMUL(T op1, T op2);
  template<typename T> T VDIV(T op1, T op2);
  template<typename T> T VDIV_FAST(T op1, T op2);
  template<typename T> T VMIN(T op1, T op2);
  template<typename T> T VMAX(T op1, T op2);
  template<typename T> T VABS(T op1);
  template<typename T> void VSTORE(float* addr, T op);
  template<typename T> void VSTORE(int32_t* addr, T op);
  template<typename T> void VSTORE(uint8_t* addr, T op);
  template<typename T> void VSTORE_ALIGNED(float* addr, T op) { VSTORE(addr, op); }
  template<typename T> void VSTORE_ALIGNED(int32_t* addr, T op) { VSTORE(addr, op); }
  template<typename T> void VSTORE_ALIGNED(uint8_t* addr, T op) { VSTORE(addr, op); }
  template<InstructionSet S> SFLOAT VLOAD(const float* addr);
  template<InstructionSet S> SINT32 VLOAD(const int32_t* addr);
  template<InstructionSet S> SINT32 VLOAD(const uint8_t* addr);
  template<InstructionSet S> SFLOAT VSET(float op);
  template<InstructionSet S> SINT32 VSET(int32_t op);
  template<InstructionSet S> SINT32 VSET8x16(uint8_t op_a, uint8_t op_b, uint8_t op_c, uint8_t op_d,
                                             uint8_t op_e, uint8_t op_f, uint8_t op_g, uint8_t op_h,
                                             uint8_t op_i, uint8_t op_j, uint8_t op_k, uint8_t op_l,
                                             uint8_t op_m, uint8_t op_n, uint8_t op_o, uint8_t op_p);
  template<InstructionSet S> SFLOAT VTO_FLOAT(SINT32 op1);
  template<InstructionSet S> SINT32 VTO_INT32(SFLOAT op1);
  template<InstructionSet S, size_t I> typename std::enable_if<I < 4u, SINT32>::type VEXPAND_QTR(SINT32 op);
  template<InstructionSet S, size_t I> typename std::enable_if<InstructionSetTrait<S>::num_lanes == 4u &&
                                                               I < 4u, SINT32>::type VEXPAND_BYTE(SINT32 op) {
    return VEXPAND_QTR<S, I>(op);
  }
  template<InstructionSet S> SINT32 VCOLLAPSE_TO_BYTES(SINT32 op1, SINT32 op2, SINT32 op3, SINT32 op4);
  template<typename T> T VBITWISE_OR(T op1, T op2);
  template<typename T> T VBITWISE_AND(T op1, T op2);
  template<InstructionSet S> SINT32 VLT(SFLOAT op1, SFLOAT op2);
  template<InstructionSet S> SINT32 VLT(SINT32 op1, SINT32 op2);
  template<InstructionSet S> SINT32 VEQ(SFLOAT op1, SFLOAT op2);
  template<InstructionSet S> SINT32 VEQ(SINT32 op1, SINT32 op2);
  template<InstructionSet S> SFLOAT VLDEXP(SFLOAT s, SINT32 e); // Compute s * 2^e
  // Note that standard frexp(x) returns (s, iexp) where x=s*2^iexp and 0.5<=s<1. However, VFREXP will return 1<=s<2.
  template<InstructionSet S> SFLOAT VFREXP(SFLOAT x, SINT32& iexp);
  template<InstructionSet S> SFLOAT VLOOKUP_FP_ARRAY(const float* arr, SINT32 index); // Return arr[index] without boundary checking.
  template<InstructionSet S> SINT32 VSELECT(SINT32 mask, SINT32 op1, SINT32 op2); // mask==0?op1:op2
  template<InstructionSet S> SFLOAT VSELECT(SINT32 mask, SFLOAT op1, SFLOAT op2); // mask==0?op1:op2
  template<typename T> uint32_t VNARROW_TO_8BIT(T op); // Takes the least significant 8-bit of each lane and compact.
  template<InstructionSet S> SINT32 VSHUFFLE(SINT32 op1, SINT32 index);
  template<InstructionSet S> SFLOAT VCAST_FLOAT(SINT32 op1);
}

/*======================================================*/
//           Inline implementation for SCALAR
/*======================================================*/
namespace lopper {
  template<> inline float VADD(float op1, float op2) { return op1 + op2; }
  template<> inline float VSUB(float op1, float op2) { return op1 - op2; }
  template<> inline float VMUL(float op1, float op2) { return op1 * op2; }
  template<> inline float VDIV(float op1, float op2) { return op1 / op2; }
  template<> inline float VDIV_FAST(float op1, float op2) { return op1 / op2; }
  template<> inline float VMIN(float op1, float op2) { return op1 > op2 ? op2 : op1; }
  template<> inline float VMAX(float op1, float op2) { return op1 > op2 ? op1 : op2; }
  template<> inline float VABS(float op1) { return fabs(op1); }
  template<> inline void VSTORE(float* addr, float op) { addr[0] = op; }
  template<> inline float VLOAD<SCALAR>(const float* addr) { return addr[0]; }
  template<> inline float VSET<SCALAR>(float op) { return op; }
  template<> inline int32_t VSET8x16<SCALAR>(uint8_t op_a, uint8_t op_b, uint8_t op_c, uint8_t op_d,
                                             uint8_t, uint8_t, uint8_t, uint8_t,
                                             uint8_t, uint8_t, uint8_t, uint8_t,
                                             uint8_t, uint8_t, uint8_t, uint8_t) {
    return (int32_t)((uint32_t)op_a | ((uint32_t)op_b << 8) | ((uint32_t)op_c << 16) | ((uint32_t)op_d << 24));
  }
  template<> inline int32_t VADD(int32_t op1, int32_t op2) { return op1 + op2; }
  template<> inline int32_t VSUB(int32_t op1, int32_t op2) { return op1 - op2; }
  template<> inline int32_t VMUL(int32_t op1, int32_t op2) { return op1 * op2; }
  template<> inline int32_t VDIV(int32_t op1, int32_t op2) { return op1 / op2; }
  template<> inline int32_t VDIV_FAST(int32_t op1, int32_t op2) { return op1 / op2; }
  template<> inline int32_t VMIN(int32_t op1, int32_t op2) { return op1 > op2 ? op2 : op1; }
  template<> inline int32_t VMAX(int32_t op1, int32_t op2) { return op1 > op2 ? op1 : op2; }
  template<> inline int32_t VABS(int32_t op1) { return (op1 + (op1 >> 31)) ^ (op1 >> 31); }
  template<> inline void VSTORE(int32_t* addr, int32_t op) { addr[0] = op; }
  template<> inline void VSTORE(uint8_t* addr, int32_t op) { memcpy(addr, &op, sizeof(int32_t)); }
  template<> inline int32_t VLOAD<SCALAR>(const int32_t* addr) { return addr[0]; }
  template<> inline int32_t VLOAD<SCALAR>(const uint8_t* addr) {
    // NOTE(jongmin): One may be tempted to dereference the pointer as int32_t*, but this is actually illegal.
    int32_t tmp = 0; memcpy((char*)&tmp, addr, sizeof(int32_t)); return tmp;
  }
  template<> inline int32_t VSET<SCALAR>(int32_t op) { return op; }

  template<> inline float VTO_FLOAT<SCALAR>(int32_t op1) { return (float)op1; }
  template<> inline int32_t VTO_INT32<SCALAR>(float op1) { return (int32_t)op1; }
  template<> inline int32_t VEXPAND_QTR<SCALAR, 0>(int32_t op1) { return (op1 >> (0 * 8)) & 0x0ff; }
  template<> inline int32_t VEXPAND_QTR<SCALAR, 1>(int32_t op1) { return (op1 >> (1 * 8)) & 0x0ff; }
  template<> inline int32_t VEXPAND_QTR<SCALAR, 2>(int32_t op1) { return (op1 >> (2 * 8)) & 0x0ff; }
  template<> inline int32_t VEXPAND_QTR<SCALAR, 3>(int32_t op1) { return (op1 >> (3 * 8)) & 0x0ff; }
  template<> inline int32_t VCOLLAPSE_TO_BYTES<SCALAR>(int32_t op1, int32_t op2, int32_t op3, int32_t op4) {
    uint32_t ret = VMIN(VMAX(op1, 0), 255) |
      (VMIN(VMAX(op2, 0), 255) << 8) |
      (VMIN(VMAX(op3, 0), 255) << 16) |
      (((uint32_t)VMIN(VMAX(op4, 0), 255)) << 24);
    return static_cast<int32_t>(ret);
  }
  template<> inline int32_t VBITWISE_OR(int32_t op1, int32_t op2) { return op1 | op2; }
  template<> inline int32_t VBITWISE_AND(int32_t op1, int32_t op2) { return op1 & op2; }
  template<size_t bits> inline int32_t VSHIFTL(int32_t op1) { return op1 << bits; }
  template<size_t bits> inline int32_t VSHIFTR(int32_t op1) { return op1 >> bits; }
  template<> inline int32_t VLT<SCALAR>(float op1, float op2) { return op1 < op2 ? 1 : 0; }
  template<> inline int32_t VLT<SCALAR>(int32_t op1, int32_t op2) { return op1 < op2 ? 1 : 0; }
  template<> inline int32_t VEQ<SCALAR>(float op1, float op2) { return op1 == op2 ? 1 : 0; }
  template<> inline int32_t VEQ<SCALAR>(int32_t op1, int32_t op2) { return op1 == op2 ? 1 : 0; }
  template<> inline float VLDEXP<SCALAR>(float s, int32_t e) { return ldexp(s, e); }
  template<> inline float VFREXP<SCALAR>(float s, int32_t& iexp) {
    float result = frexp((double)s, &iexp); iexp -= 1; return result * 2.f; }
  template<> inline float VLOOKUP_FP_ARRAY<SCALAR>(const float* arr, int32_t index) { return arr[index]; }
  template<> inline float VSELECT<SCALAR>(int32_t mask, float op1, float op2) { return mask==0?op1:op2; }
  template<> inline int32_t VSELECT<SCALAR>(int32_t mask, int32_t op1, int32_t op2) { return mask==0?op1:op2; }
  template<> inline uint32_t VNARROW_TO_8BIT(int32_t op) { return op & 0x0ff; }

  template<size_t C> inline int32_t VGET_LANE(int32_t op) { return op; }
  template<size_t C> inline float VGET_LANE(float op) { return op; }
  template<> inline float VADD_PAIRWISE<>(float op1, float op2) { return op1 + op2; }

  template<> inline int32_t VSHUFFLE<SCALAR>(int32_t op1, int32_t index) {
    uint32_t out = 0u;
    for (size_t i = 0; i < 4; i++) {
      const uint8_t bit = (uint8_t)(index & (0xff << (i * 8)));
      const uint8_t selected = bit < 4 ? (op1 >> (8 * bit) & 0xff) : 0;
      out = out | (selected << (i * 8));
    }
    return out;
  }
  template<> inline float VCAST_FLOAT<SCALAR>(int32_t op1) {
    float ret;
    memcpy(&ret, &op1, sizeof(float));
    return ret;
  }
}

/*======================================================*/
//            Inline implementation for SSE
/*======================================================*/
// NOTE(jongmin): Clang has an open bug (#20670) yielding overzealous alignment warning when making SSE calls
// for loading and storing, so we cast to void* first as a workaround.

#if defined LOPPER_TARGET_SSE4
namespace lopper {
  template<> inline __m128 VADD(__m128 op1, __m128 op2) { return _mm_add_ps(op1, op2); }
  template<> inline __m128 VSUB(__m128 op1, __m128 op2) { return _mm_sub_ps(op1, op2); }
  template<> inline __m128 VMUL(__m128 op1, __m128 op2) { return _mm_mul_ps(op1, op2); }
  template<> inline __m128 VDIV(__m128 op1, __m128 op2) { return _mm_div_ps(op1, op2); }
  template<> inline __m128 VDIV_FAST(__m128 op1, __m128 op2) { return _mm_mul_ps(op1, _mm_rcp_ps(op2)); }
  template<> inline __m128 VMIN(__m128 op1, __m128 op2) { return _mm_min_ps(op1, op2); }
  template<> inline __m128 VMAX(__m128 op1, __m128 op2) { return _mm_max_ps(op1, op2); }
  template<> inline __m128 VABS(__m128 op1) { return _mm_andnot_ps(_mm_set1_ps(-0.f), op1); }
  template<> inline void VSTORE(float* addr, __m128 op) { _mm_storeu_ps(addr, op); }
  template<> inline void VSTORE_ALIGNED(float* addr, __m128 op) { _mm_store_ps(addr, op); }
  template<> inline __m128 VLOAD<SSE>(const float* addr) { return _mm_loadu_ps(addr); }
  template<> inline __m128 VSET<SSE>(float op) { return _mm_set1_ps(op); }

  template<> inline __m128i VADD(__m128i op1, __m128i op2) { return _mm_add_epi32(op1, op2); }
  template<> inline __m128i VSUB(__m128i op1, __m128i op2) { return _mm_sub_epi32(op1, op2); }
  template<> inline __m128i VMUL(__m128i op1, __m128i op2) { return _mm_mullo_epi32(op1, op2); }
  template<> inline __m128i VDIV(__m128i op1, __m128i op2) {
    int32_t op1_array[4] __attribute__((aligned(16)));
    int32_t op2_array[4] __attribute__((aligned(16)));
    _mm_store_si128((__m128i*)(void*)op1_array, op1);
    _mm_store_si128((__m128i*)(void*)op2_array, op2);
    return _mm_set_epi32(op1_array[3]/op2_array[3], op1_array[2]/op2_array[2],
                         op1_array[1]/op2_array[1], op1_array[0]/op2_array[0]);
  }
  template<> inline __m128i VDIV_FAST(__m128i op1, __m128i op2) { return VDIV(op1, op2); }
  template<> inline __m128i VMIN(__m128i op1, __m128i op2) { return _mm_min_epi32(op1, op2); }
  template<> inline __m128i VMAX(__m128i op1, __m128i op2) { return _mm_max_epi32(op1, op2); }
  template<> inline __m128i VABS(__m128i op1) { return _mm_abs_epi32(op1); }
  template<> inline void VSTORE(int32_t* addr, __m128i op) { _mm_storeu_si128((__m128i*)(void*)addr, op); }
  template<> inline void VSTORE_ALIGNED(int32_t* addr, __m128i op) { _mm_store_si128((__m128i*)(void*)addr, op); }
  template<> inline void VSTORE(uint8_t* addr, __m128i op) { _mm_storeu_si128((__m128i*)(void*)addr, op); }
  template<> inline void VSTORE_ALIGNED(uint8_t* addr, __m128i op) { _mm_store_si128((__m128i*)(void*)addr, op); }
  template<> inline __m128i VLOAD<SSE>(const int32_t* addr) { return _mm_loadu_si128((__m128i*)(void*)addr); }
  template<> inline __m128i VLOAD<SSE>(const uint8_t* addr) { return _mm_loadu_si128((__m128i*)(void*)addr); }
  template<> inline __m128i VSET<SSE>(int32_t op) { return _mm_set1_epi32(op); }
  template<> inline __m128i VSET8x16<SSE>(uint8_t op_a, uint8_t op_b, uint8_t op_c, uint8_t op_d,
                                          uint8_t op_e, uint8_t op_f, uint8_t op_g, uint8_t op_h,
                                          uint8_t op_i, uint8_t op_j, uint8_t op_k, uint8_t op_l,
                                          uint8_t op_m, uint8_t op_n, uint8_t op_o, uint8_t op_p) {
    return _mm_setr_epi8(op_a, op_b, op_c, op_d, op_e, op_f, op_g, op_h,
                         op_i, op_j, op_k, op_l, op_m, op_n, op_o, op_p);
  }
  template<> inline __m128 VTO_FLOAT<SSE>(__m128i op1) { return _mm_cvtepi32_ps(op1); }
  template<> inline __m128i VTO_INT32<SSE>(__m128 op1) { return _mm_cvttps_epi32(op1); }
  template<> inline __m128i VEXPAND_QTR<SSE, 0>(__m128i op) { return _mm_cvtepu8_epi32(op); }
  template<> inline __m128i VEXPAND_QTR<SSE, 1>(__m128i op) { return _mm_cvtepu8_epi32(_mm_srli_si128(op, 4)); }
  template<> inline __m128i VEXPAND_QTR<SSE, 2>(__m128i op) { return _mm_cvtepu8_epi32(_mm_srli_si128(op, 8)); }
  template<> inline __m128i VEXPAND_QTR<SSE, 3>(__m128i op) { return _mm_cvtepu8_epi32(_mm_srli_si128(op, 12)); }
  template<> inline __m128i VCOLLAPSE_TO_BYTES<SSE>(__m128i op1, __m128i op2, __m128i op3, __m128i op4) {
    return _mm_packus_epi16(_mm_packs_epi32(op1, op2), _mm_packs_epi32(op3, op4));
  }
  template<> inline __m128i VBITWISE_OR(__m128i op1, __m128i op2) { return _mm_or_si128(op1, op2); }
  template<> inline __m128i VBITWISE_AND(__m128i op1, __m128i op2) { return _mm_and_si128(op1, op2); }
  template<size_t bits> inline __m128i VSHIFTL(__m128i op1) { return _mm_slli_epi32(op1, bits); }
  template<size_t bits> inline __m128i VSHIFTR(__m128i op1) { return _mm_srai_epi32(op1, bits); }
  template<> inline __m128i VLT<SSE>(__m128 op1, __m128 op2) { return _mm_castps_si128(_mm_cmplt_ps(op1, op2)); }
  template<> inline __m128i VLT<SSE>(__m128i op1, __m128i op2) { return _mm_cmplt_epi32(op1, op2); }
  template<> inline __m128i VEQ<SSE>(__m128 op1, __m128 op2) { return _mm_castps_si128(_mm_cmpeq_ps(op1, op2)); }
  template<> inline __m128i VEQ<SSE>(__m128i op1, __m128i op2) { return _mm_cmpeq_epi32(op1, op2); }
  template<> inline __m128 VLDEXP<SSE>(__m128 s, __m128i e) {
    // Need to increase the exponent for the floating-point s by e. This may overflow.
    __m128i MASK_EXPONENT = VSET<SSE>(0x7f800000);
    __m128i exponent = _mm_srai_epi32(_mm_and_si128(_mm_castps_si128(s), MASK_EXPONENT), 23);
    __m128 result = _mm_castsi128_ps(_mm_or_si128(_mm_andnot_si128(MASK_EXPONENT, _mm_castps_si128(s)),
                                                  _mm_slli_epi32(VADD(exponent, e), 23)));
    return result;
  }
  template<> inline __m128 VFREXP<SSE>(__m128 a, __m128i& iexp) {
    __m128i MASK_EXPONENT = VSET<SSE>(0x7f800000);
    __m128 significand = _mm_castsi128_ps(_mm_or_si128(_mm_andnot_si128(MASK_EXPONENT, _mm_castps_si128(a)),
                                                       _mm_slli_epi32(_mm_set1_epi32(127), 23)));
    iexp = VSUB(_mm_srai_epi32(_mm_and_si128(_mm_castps_si128(a), MASK_EXPONENT), 23), VSET<SSE>(127));
    return significand;
  }
  template<> inline __m128 VLOOKUP_FP_ARRAY<SSE>(const float* arr, __m128i index) {
    int32_t indices[4] __attribute__((aligned(16)));
    _mm_store_si128((__m128i*)(void*)indices, index);
    return _mm_set_ps(arr[indices[3]], arr[indices[2]], arr[indices[1]], arr[indices[0]]);
  }
  template<> inline __m128 VSELECT<SSE>(__m128i mask, __m128 op1, __m128 op2) {
    const __m128 is_zero = _mm_cmpeq_ps(_mm_castsi128_ps(mask), _mm_setzero_ps());
    return _mm_or_ps(_mm_and_ps(is_zero, op1),
                     _mm_andnot_ps(is_zero, op2));
  }
  template<> inline __m128i VSELECT<SSE>(__m128i mask, __m128i op1, __m128i op2) {
    const __m128i is_zero = _mm_cmpeq_epi32(mask, _mm_setzero_si128());
    return _mm_or_si128(_mm_and_si128(is_zero, op1),
                        _mm_andnot_si128(is_zero, op2));
  }
  template<size_t C> inline int32_t VGET_LANE(__m128i op) { return _mm_extract_epi32(op, C); }
  template<size_t C> inline float VGET_LANE(__m128 op) { union { int32_t i; float f; } tmp; tmp.i = _mm_extract_ps(op, C); return tmp.f; }
  template<> inline uint32_t VNARROW_TO_8BIT(__m128i op) {
    return (uint32_t)VGET_LANE<0>(_mm_shuffle_epi8(op, _mm_setr_epi8(0, 4, 8, 12, 255u, 255u, 255u, 255u,
                                                                     255u, 255u, 255u, 255u, 255u, 255u, 255u, 255u)));
  }
  template<> inline __m128 VADD_PAIRWISE(__m128 op1, __m128 op2) { return _mm_hadd_ps(op1, op2); }
  template<> inline __m128i VSHUFFLE<SSE>(__m128i op1, __m128i index) { return _mm_shuffle_epi8(op1, index); }
  template<> inline __m128 VCAST_FLOAT<SSE>(__m128i op1) { return _mm_castsi128_ps(op1); }
}
#endif

/*======================================================*/
//            Inline implementation for NEON
/*======================================================*/
#if defined LOPPER_TARGET_NEON
namespace lopper {
  template<> inline float32x4_t VADD(float32x4_t op1, float32x4_t op2) { return vaddq_f32(op1, op2); }
  template<> inline float32x4_t VSUB(float32x4_t op1, float32x4_t op2) { return vsubq_f32(op1, op2); }
  template<> inline float32x4_t VMUL(float32x4_t op1, float32x4_t op2) { return vmulq_f32(op1, op2); }
  template<> inline float32x4_t VDIV(float32x4_t op1, float32x4_t op2) {
    float op1_array[4], op2_array[4];
    vst1q_f32(op1_array, op1);
    vst1q_f32(op2_array, op2);
    op1_array[0] /= op2_array[0];
    op1_array[1] /= op2_array[1];
    op1_array[2] /= op2_array[2];
    op1_array[3] /= op2_array[3];
    return vld1q_f32(op1_array);
  }
  template<> inline float32x4_t VDIV_FAST(float32x4_t op1, float32x4_t op2) { return vmulq_f32(op1, vrecpeq_f32(op2)); }
  template<> inline float32x4_t VMIN(float32x4_t op1, float32x4_t op2) { return vminq_f32(op1, op2); }
  template<> inline float32x4_t VMAX(float32x4_t op1, float32x4_t op2) { return vmaxq_f32(op1, op2); }
  template<> inline float32x4_t VABS(float32x4_t op1) { return vabsq_f32(op1); }
  template<> inline void VSTORE(float* addr, float32x4_t op) { vst1q_f32(addr, op); }
  template<> inline void VSTORE_ALIGNED(float* addr, float32x4_t op) { VSTORE(addr, op); }
  template<> inline float32x4_t VLOAD<NEON>(const float* addr) { return vld1q_f32(addr); }
  template<> inline float32x4_t VSET<NEON>(float op) { return vmovq_n_f32(op); }

  template<> inline int32x4_t VADD(int32x4_t op1, int32x4_t op2) { return vaddq_s32(op1, op2); }
  template<> inline int32x4_t VSUB(int32x4_t op1, int32x4_t op2) { return vsubq_s32(op1, op2); }
  template<> inline int32x4_t VMUL(int32x4_t op1, int32x4_t op2) { return vmulq_s32(op1, op2); }
  template<> inline int32x4_t VDIV(int32x4_t op1, int32x4_t op2) {
    int32_t op1_array[4], op2_array[4];
    vst1q_s32(op1_array, op1);
    vst1q_s32(op2_array, op2);
    op1_array[0] /= op2_array[0];
    op1_array[1] /= op2_array[1];
    op1_array[2] /= op2_array[2];
    op1_array[3] /= op2_array[3];
    return vld1q_s32(op1_array);
  }
  template<> inline int32x4_t VDIV_FAST(int32x4_t op1, int32x4_t op2) { return VDIV(op1, op2); }
  template<> inline int32x4_t VMIN(int32x4_t op1, int32x4_t op2) { return vminq_s32(op1, op2); }
  template<> inline int32x4_t VMAX(int32x4_t op1, int32x4_t op2) { return vmaxq_s32(op1, op2); }
  template<> inline int32x4_t VABS(int32x4_t op1) { return vabsq_s32(op1); }
  template<> inline void VSTORE(int32_t* addr, int32x4_t op) { vst1q_s32(addr, op); }
  template<> inline void VSTORE(uint8_t* addr, int32x4_t op) { vst1q_u8(addr, vreinterpretq_u8_s32(op)); }
  template<> inline int32x4_t VLOAD<NEON>(const int32_t* addr) { return vld1q_s32(addr); }
  template<> inline int32x4_t VLOAD<NEON>(const uint8_t* addr) { return vreinterpretq_s32_u8(vld1q_u8(addr)); }
  template<> inline int32x4_t VSET<NEON>(int32_t op) { return vmovq_n_s32(op); }
  template<> inline int32x4_t VSET8x16<NEON>(uint8_t op_a, uint8_t op_b, uint8_t op_c, uint8_t op_d,
                                             uint8_t op_e, uint8_t op_f, uint8_t op_g, uint8_t op_h,
                                             uint8_t op_i, uint8_t op_j, uint8_t op_k, uint8_t op_l,
                                             uint8_t op_m, uint8_t op_n, uint8_t op_o, uint8_t op_p) {
    const uint32_t literal0 = (uint32_t)op_a | ((uint32_t)op_b << 8) | ((uint32_t)op_c << 16) | ((uint32_t)op_d << 24);
    const uint32_t literal1 = (uint32_t)op_e | ((uint32_t)op_f << 8) | ((uint32_t)op_g << 16) | ((uint32_t)op_h << 24);
    const uint32_t literal2 = (uint32_t)op_i | ((uint32_t)op_j << 8) | ((uint32_t)op_k << 16) | ((uint32_t)op_l << 24);
    const uint32_t literal3 = (uint32_t)op_m | ((uint32_t)op_n << 8) | ((uint32_t)op_o << 16) | ((uint32_t)op_p << 24);
    return vcombine_s32(vcreate_s32(((uint64_t)literal0) | ((uint64_t)literal1 << 32)),
                        vcreate_s32(((uint64_t)literal2) | ((uint64_t)literal3 << 32)));
  }
  template<> inline float32x4_t VTO_FLOAT<NEON>(int32x4_t op1) { return vcvtq_f32_s32(op1); }
  template<> inline int32x4_t VTO_INT32<NEON>(float32x4_t op1) { return vcvtq_s32_f32(op1); }
  inline int32x4_t VEXPAND_QTR_NEON_HELPER(uint32x2_t data) { // lower half of data is valid
    auto u_16x8 = vmovl_u8(vreinterpret_u8_u32(data)); // uint16x8, lower half valid
    return vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(u_16x8)));
  }
  template<> inline int32x4_t VEXPAND_QTR<NEON, 0>(int32x4_t op) { return VEXPAND_QTR_NEON_HELPER(vreinterpret_u32_s32(vget_low_s32(op))); }
  template<> inline int32x4_t VEXPAND_QTR<NEON, 1>(int32x4_t op) { return VEXPAND_QTR_NEON_HELPER(vreinterpret_u32_s32(vrev64_s32(vget_low_s32(op)))); }
  template<> inline int32x4_t VEXPAND_QTR<NEON, 2>(int32x4_t op) { return VEXPAND_QTR_NEON_HELPER(vreinterpret_u32_s32(vget_high_s32(op))); }
  template<> inline int32x4_t VEXPAND_QTR<NEON, 3>(int32x4_t op) { return VEXPAND_QTR_NEON_HELPER(vreinterpret_u32_s32(vrev64_s32(vget_high_s32(op)))); }
  template<> inline int32x4_t VCOLLAPSE_TO_BYTES<NEON>(int32x4_t op1, int32x4_t op2, int32x4_t op3, int32x4_t op4) {
    // TODO(jongmin): Untested and unbenchmarked
    auto t12 = vmaxq_s16(vcombine_s16(vqmovn_s32(op1), vqmovn_s32(op2)), vreinterpretq_s16_s32(VSET<NEON>(0)));
    auto t34 = vmaxq_s16(vcombine_s16(vqmovn_s32(op3), vqmovn_s32(op4)), vreinterpretq_s16_s32(VSET<NEON>(0)));
    return vreinterpretq_s32_u8(vcombine_u8(vqmovn_u16(vreinterpretq_u16_s16(t12)), vqmovn_u16(vreinterpretq_u16_s16(t34))));
  }
  template<> inline int32x4_t VBITWISE_OR(int32x4_t op1, int32x4_t op2) { return vorrq_s32(op1, op2); }
  template<> inline int32x4_t VBITWISE_AND(int32x4_t op1, int32x4_t op2) { return vandq_s32(op1, op2); }
  template<size_t bits> inline int32x4_t VSHIFTL(int32x4_t op1) { return vshlq_n_s32(op1, bits); }
  template<size_t bits> inline int32x4_t VSHIFTR(int32x4_t op1) { return vshrq_n_s32(op1, bits); }
  template<> inline int32x4_t VLT<NEON>(float32x4_t op1, float32x4_t op2) { return vreinterpretq_s32_u32(vcltq_f32(op1, op2)); }
  template<> inline int32x4_t VLT<NEON>(int32x4_t op1, int32x4_t op2) { return vreinterpretq_s32_u32(vcltq_s32(op1, op2)); }
  template<> inline int32x4_t VEQ<NEON>(float32x4_t op1, float32x4_t op2) { return vreinterpretq_s32_u32(vceqq_f32(op1, op2)); }
  template<> inline int32x4_t VEQ<NEON>(int32x4_t op1, int32x4_t op2) { return vreinterpretq_s32_u32(vceqq_s32(op1, op2)); }
  template<> inline float32x4_t VLDEXP<NEON>(float32x4_t s, int32x4_t e) {
    // Need to increase the exponent for the floating-point s by e. This may overflow.
    int32x4_t MASK_EXPONENT = VSET<NEON>(0x7f800000);
    int32x4_t exponent = vshrq_n_s32(vandq_s32(vreinterpretq_s32_f32(s), MASK_EXPONENT), 23);
    float32x4_t result = vreinterpretq_f32_s32(vorrq_s32(vbicq_s32(vreinterpretq_s32_f32(s), MASK_EXPONENT),
                                                        vshlq_n_s32(VADD(exponent, e), 23)));
    return result;
  }
  template<> inline float32x4_t VFREXP<NEON>(float32x4_t a, int32x4_t& iexp) {
    int32x4_t MASK_EXPONENT = VSET<NEON>(0x7f800000);
    float32x4_t significand = vreinterpretq_f32_s32(vorrq_s32(vbicq_s32(vreinterpretq_s32_f32(a), MASK_EXPONENT),
                                                             vshlq_n_s32(VSET<NEON>(127), 23)));
    iexp = VSUB(vshrq_n_s32(vandq_s32(vreinterpretq_s32_f32(a), MASK_EXPONENT), 23), VSET<NEON>(127));
    return significand;
  }
  template<> inline float32x4_t VLOOKUP_FP_ARRAY<NEON>(const float* arr, int32x4_t index) {
    int32_t indices[4];
    VSTORE(indices, index);
    float values[4] = { arr[indices[0]], arr[indices[1]], arr[indices[2]], arr[indices[3]] };
    return VLOAD<NEON>(values);
  }
  template<> inline float32x4_t VSELECT<NEON>(int32x4_t mask, float32x4_t op1, float32x4_t op2) {
    uint32x4_t is_zero  = vceqq_u32(vreinterpretq_u32_s32(mask), vreinterpretq_u32_s32(VSET<NEON>(0)));
    return vreinterpretq_f32_u32(vorrq_u32(vandq_u32(vreinterpretq_u32_f32(op1), is_zero),
                                           vbicq_u32(vreinterpretq_u32_f32(op2), is_zero)));
  }
  template<> inline int32x4_t VSELECT<NEON>(int32x4_t mask, int32x4_t op1, int32x4_t op2) {
    uint32x4_t is_zero  = vceqq_u32(vreinterpretq_u32_s32(mask), vreinterpretq_u32_s32(VSET<NEON>(0)));
    return vreinterpretq_s32_u32(vorrq_u32(vandq_u32(vreinterpretq_u32_s32(op1), is_zero),
                                           vbicq_u32(vreinterpretq_u32_s32(op2), is_zero)));
  }

  template<size_t C> inline int32_t VGET_LANE(int32x4_t op) { return vgetq_lane_s32(op, C); }
  template<size_t C> inline float VGET_LANE(float32x4_t op) { return vgetq_lane_f32(op, C); }
  template<> inline uint32_t VNARROW_TO_8BIT(int32x4_t op) {
    return vget_lane_u32(vreinterpret_u32_u8(vmovn_u16(vcombine_u16(vmovn_u32(vreinterpretq_u32_s32(op)), vcreate_u16(0)))), 0);
  }
  template<> inline float32x4_t VADD_PAIRWISE(float32x4_t op1, float32x4_t op2) {
    return vcombine_f32(vpadd_f32(vget_low_f32(op1), vget_high_f32(op1)),
                        vpadd_f32(vget_low_f32(op2), vget_high_f32(op2)));
  }
  template<> inline int32x4_t VSHUFFLE<NEON>(int32x4_t op1, int32x4_t index) {
    auto index_low = vreinterpret_u8_s32(vget_low_s32(index));
    auto index_hi = vreinterpret_u8_s32(vget_high_s32(index));
    auto lut = (uint8x8x2_t){{vreinterpret_u8_s32(vget_low_s32(op1)), vreinterpret_u8_s32(vget_high_s32(op1))}};
    return vreinterpretq_s32_u8(vcombine_u8(vtbl2_u8(lut, index_low), vtbl2_u8(lut, index_hi)));
  }
  template<> inline float32x4_t VCAST_FLOAT<NEON>(int32x4_t op1) { return vreinterpretq_f32_s32(op1); }
}
#endif

#undef SFLOAT
#undef SINT32
#undef SUINT8
