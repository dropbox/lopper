#pragma once

namespace lopper {

/* An enum of known instruction-set types. */
typedef enum {
  SCALAR, /* no vectorization */
  SSE,
  NEON,
  AVX
} InstructionSet;

} // end namespace lopper

/* Check that one of LOPPER_NO_SIMD, LOPPER_TARGET_SSE4 and LOPPER_TARGET_NEON is defined,
 * and include the appropriate header if needed.
 */
#ifdef LOPPER_NO_SIMD
  #if defined(LOPPER_TARGET_SSE4) || defined(LOPPER_TARGET_NEON) || defined(LOPPER_TARGET_AVX)
    #error "Cannot define vectorization macros with LOPPER_NO_SIMD"
  #endif
#elif defined(__AVX__)
  #ifndef LOPPER_TARGET_AVX
    #define LOPPER_TARGET_AVX
  #endif
#elif defined(__SSE4_2__)
  #ifndef LOPPER_TARGET_SSE4
    #define LOPPER_TARGET_SSE4
  #endif
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
  #ifndef LOPPER_TARGET_NEON
    #define LOPPER_TARGET_NEON
  #endif
#endif

#if (defined(LOPPER_TARGET_SSE4)?1:0) + (defined(LOPPER_TARGET_NEON)?1:0) + (defined(LOPPER_TARGET_AVX)?1:0) > 1
  #error "Cannot define vectorization macros for multiple distinct targets"
#elif defined(LOPPER_TARGET_AVX)
  #include <immintrin.h>
  #define LOPPER_TARGET ::lopper::AVX
#elif defined(LOPPER_TARGET_SSE4)
  #include <smmintrin.h>
  #define LOPPER_TARGET ::lopper::SSE
#elif defined(LOPPER_TARGET_NEON)
  #include <arm_neon.h>
  #define LOPPER_TARGET ::lopper::NEON
#else
  #define LOPPER_TARGET ::lopper::SCALAR
  #ifndef LOPPER_NO_SIMD
    #define LOPPER_NO_SIMD
  #endif
#endif
