#pragma once

#include <stdint.h>

#include "platform.hpp"

namespace lopper {

/* Instruction-set-dependent vector types */
template<InstructionSet> struct InstructionSetTrait;
template<> struct InstructionSetTrait<SCALAR> {
  typedef float FLOAT;
  typedef int32_t INT32;
  typedef uint8_t UINT8;
  constexpr static size_t num_lanes = 1;
};
#if defined(LOPPER_TARGET_SSE4)
  template<> struct InstructionSetTrait<SSE> {
    typedef __m128 FLOAT;
    typedef __m128i INT32;
    typedef __m128i UINT8;
    constexpr static size_t num_lanes = 4;
  };
#elif defined(LOPPER_TARGET_NEON)
  template<> struct InstructionSetTrait<NEON> {
    typedef float32x4_t FLOAT;
    typedef int32x4_t INT32;
    typedef uint8x16_t UINT8;
    constexpr static size_t num_lanes = 4;
  };
#endif

/* Instruction-set-dependent scalar and vector types */
template<typename T, InstructionSet S> struct MultipleTrait;

template<typename T> struct MultipleTrait<T, SCALAR> {
  typedef T type;  // the primitive type, e.g. float, int32_t
  typedef T vtype; // the platform-dependent type, e.g. float or __m128, depending on InstructionSet
  constexpr static size_t num_lanes = 1;
};

#ifndef LOPPER_NO_SIMD
template<> struct MultipleTrait<float, LOPPER_TARGET> {
  typedef float type;
  typedef typename InstructionSetTrait<LOPPER_TARGET>::FLOAT vtype;
  constexpr static size_t num_lanes = InstructionSetTrait<LOPPER_TARGET>::num_lanes;
};

template<> struct MultipleTrait<int32_t, LOPPER_TARGET> {
  typedef int32_t type;
  typedef typename InstructionSetTrait<LOPPER_TARGET>::INT32 vtype;
  constexpr static size_t num_lanes = InstructionSetTrait<LOPPER_TARGET>::num_lanes;
};
#endif

#define LOPPER_BITWIDTH (::lopper::InstructionSetTrait<LOPPER_TARGET>::num_lanes * 32u)

} // end namespace lopper
