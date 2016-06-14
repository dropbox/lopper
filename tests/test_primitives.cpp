#include <cstring>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <gtest/gtest.h>

#include "lopper/primitives.hpp"

using namespace lopper;

#ifndef LOPPER_NO_SIMD
TEST(LopperTestVector, LittleEndianTest) {
  // Make sure we're in little Endian, as some of the routines assume this explicitly.
  // While the corresponding tests will fail, we'll have an explicit test here.
  auto v = VSET<LOPPER_TARGET>(4 | (5 << 8) | (6 << 16) | (7 << 24));
  uint8_t buffer_u8[16] = {0};
  VSTORE(buffer_u8, v);
  ASSERT_EQ(4, buffer_u8[0]); // Big-endian would give 7 here.
}

TEST(LopperTestVector, SetTest) {
  uint8_t buffer_u8[16] = {0};
  int32_t buffer_s32[4] = {0};
  // Try setting a vector from a literal.
  auto v = VSET8x16<LOPPER_TARGET>(0, 1, 2, 3, 4, 5, 6, 7,
                                   8, 9, 10, 11, 12, 13, 14, 15);
  VSTORE(buffer_u8, v);
  for (size_t y = 0; y < 16; y++) {
    ASSERT_EQ((uint8_t)y, buffer_u8[y]);
  }
  // Try setting a vector to a 32-bit input for all lanes.
  v = VSET<LOPPER_TARGET>(12345);
  VSTORE(buffer_s32, v);
  for (size_t y = 0; y < 4; y++) {
    ASSERT_EQ(12345, buffer_s32[y]);
  }
}

TEST(LopperTestVector, ExponentiationTest) {
  // Test VLDEXP and VFREXP.
  std::vector<float> inputs = {-32.1f, 1.0f, 0.4f, 0.001f, 0.9f, 1.98f, 3.99f, 123.45f};
  for (auto& input : inputs) {
    float buffer_in[4] = {input, input + 1, input + 2, input + 3};
    float buffer_fout[4] = {0.f, 0.f, 0.f, 0.f};
    int32_t buffer_iout[4] = {0, 0, 0, 0};
    // Test VLDEXP.
    VSTORE(buffer_fout, VLDEXP<LOPPER_TARGET>(VLOAD<LOPPER_TARGET>(buffer_in), VSET<LOPPER_TARGET>(3)));
    ASSERT_EQ(VLDEXP<SCALAR>(buffer_in[0], 3), buffer_fout[0]);
    ASSERT_EQ(VLDEXP<SCALAR>(buffer_in[1], 3), buffer_fout[1]);
    ASSERT_EQ(VLDEXP<SCALAR>(buffer_in[2], 3), buffer_fout[2]);
    ASSERT_EQ(VLDEXP<SCALAR>(buffer_in[3], 3), buffer_fout[3]);
    // Test VFREXP.
    auto viexp = VSET<LOPPER_TARGET>(0);
    int32_t iexp[4];
    VSTORE(buffer_fout, VFREXP<LOPPER_TARGET>(VLOAD<LOPPER_TARGET>(buffer_in), viexp));
    VSTORE(buffer_iout, viexp);
    ASSERT_EQ(VFREXP<SCALAR>(buffer_in[0], iexp[0]), buffer_fout[0]);
    ASSERT_EQ(VFREXP<SCALAR>(buffer_in[1], iexp[1]), buffer_fout[1]);
    ASSERT_EQ(VFREXP<SCALAR>(buffer_in[2], iexp[2]), buffer_fout[2]);
    ASSERT_EQ(VFREXP<SCALAR>(buffer_in[3], iexp[3]), buffer_fout[3]);
    ASSERT_EQ(iexp[0], buffer_iout[0]);
    ASSERT_EQ(iexp[1], buffer_iout[1]);
    ASSERT_EQ(iexp[2], buffer_iout[2]);
    ASSERT_EQ(iexp[3], buffer_iout[3]);
  }
}

TEST(LopperTestVector, LoadTest) {
  float buffer_in[4] = {1.5f, 2.5f, -3.9f, 100.1f};
  int32_t indices[4] = {1, 2, 3, 0};
  float buffer_out[4];
  typename InstructionSetTrait<LOPPER_TARGET>::FLOAT tmp =
    VLOOKUP_FP_ARRAY<LOPPER_TARGET>(buffer_in, VLOAD<LOPPER_TARGET>(indices));
  ASSERT_EQ(buffer_in[indices[0]], VGET_LANE<0>(tmp));
  ASSERT_EQ(buffer_in[indices[1]], VGET_LANE<1>(tmp));
  ASSERT_EQ(buffer_in[indices[2]], VGET_LANE<2>(tmp));
  ASSERT_EQ(buffer_in[indices[3]], VGET_LANE<3>(tmp));

  VSTORE(buffer_out, tmp);
  ASSERT_EQ(VLOOKUP_FP_ARRAY<SCALAR>(buffer_in, indices[0]), buffer_out[0]);
  ASSERT_EQ(VLOOKUP_FP_ARRAY<SCALAR>(buffer_in, indices[1]), buffer_out[1]);
  ASSERT_EQ(VLOOKUP_FP_ARRAY<SCALAR>(buffer_in, indices[2]), buffer_out[2]);
  ASSERT_EQ(VLOOKUP_FP_ARRAY<SCALAR>(buffer_in, indices[3]), buffer_out[3]);

  for (size_t i = 0; i < 4; i++) {
    ASSERT_EQ(buffer_in[indices[i]], buffer_out[i]);
  }
}

TEST(LopperTestVector, LoadUInt8IntoInt32Test) {
  uint8_t buffer_in[17] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  uint8_t buffer_out[16];
  VSTORE(buffer_out, VLOAD<LOPPER_TARGET>(buffer_in));
  for (size_t i = 0; i < 4 * InstructionSetTrait<LOPPER_TARGET>::num_lanes; i++) {
    ASSERT_EQ(buffer_out[i], buffer_in[i]);
  }
  VSTORE(buffer_out, VLOAD<LOPPER_TARGET>(buffer_in + 1));
  for (size_t i = 0; i < 4 * InstructionSetTrait<LOPPER_TARGET>::num_lanes; i++) {
    ASSERT_EQ(buffer_out[i], buffer_in[i + 1]);
  }
}

// A macro for checking SIMD and serial behaviors against each other
#define VERIFY_ARITHMETIC_OP(OP, arg1, arg2, output)                    \
  VSTORE(output, OP(VLOAD<LOPPER_TARGET>(arg1), VLOAD<LOPPER_TARGET>(arg2))); \
  for (size_t i = 0; i < 4; i++) {                                      \
    ASSERT_EQ(output[i], OP(VLOAD<SCALAR>(arg1+i), VLOAD<SCALAR>(arg2+i))); \
  }

#define VERIFY_UNARY_OP(OP, arg1, output)               \
  VSTORE(output, OP(VLOAD<LOPPER_TARGET>(arg1)));       \
  for (size_t i = 0; i < 4; i++) {                      \
    ASSERT_EQ(output[i], OP(VLOAD<SCALAR>(arg1+i)));    \
  }

TEST(LopperTestVector, FloatingPointMath) {
  float op1[4] = {1.5f, 2.5f, -3.9f, 100.1f};
  float op2[4] = {12.5f, -2.5f, 33.9f, 0.13f};
  float output_sse[4];

  VERIFY_ARITHMETIC_OP(VADD, op1, op2, output_sse);
  ASSERT_EQ(op1[0] + op2[0], output_sse[0]);

  // Pairwise add is tested separately because serial version behaves differently.
  VSTORE(output_sse, VADD_PAIRWISE(VLOAD<LOPPER_TARGET>(op1), VLOAD<LOPPER_TARGET>(op2)));
  ASSERT_EQ(op1[0] + op1[1], output_sse[0]);
  ASSERT_EQ(op1[2] + op1[3], output_sse[1]);
  ASSERT_EQ(op2[0] + op2[1], output_sse[2]);
  ASSERT_EQ(op2[2] + op2[3], output_sse[3]);

  VERIFY_ARITHMETIC_OP(VSUB, op1, op2, output_sse);
  ASSERT_EQ(op1[1] - op2[1], output_sse[1]);

  VERIFY_ARITHMETIC_OP(VMUL, op1, op2, output_sse);
  ASSERT_EQ(op1[1] * op2[1], output_sse[1]);

  VERIFY_ARITHMETIC_OP(VDIV, op1, op2, output_sse);
  ASSERT_EQ(op1[3] / op2[3], output_sse[3]);

  // For VDIV_FAST, equality is not guaratneed, so manually compare.
  VSTORE(output_sse, VDIV_FAST(VLOAD<LOPPER_TARGET>(op1), VLOAD<LOPPER_TARGET>(op2)));
  for (size_t i = 0; i < 4; i++) {
    ASSERT_NEAR(1.0f, output_sse[i] / VDIV_FAST(VLOAD<SCALAR>(op1+i), VLOAD<SCALAR>(op2+i)), 0.01f);
  }

  VERIFY_ARITHMETIC_OP(VMIN, op1, op2, output_sse);
  ASSERT_EQ(std::min(op1[2], op2[2]), output_sse[2]);

  VERIFY_ARITHMETIC_OP(VMAX, op1, op2, output_sse);
  ASSERT_EQ(std::max(op1[2], op2[2]), output_sse[2]);

  VERIFY_UNARY_OP(VABS, op1, output_sse);
  ASSERT_EQ(fabs(op1[2]), output_sse[2]);
}

TEST(LopperTestVector, IntegerMath) {
  int32_t op1[4] = {123, 456, -324, 199};
  int32_t op2[4] = {348, -123, -234, -1000};
  int32_t output_sse[4];

  VERIFY_ARITHMETIC_OP(VADD, op1, op2, output_sse);
  ASSERT_EQ(op1[0] + op2[0], output_sse[0]);

  VERIFY_ARITHMETIC_OP(VSUB, op1, op2, output_sse);
  ASSERT_EQ(op1[1] - op2[1], output_sse[1]);

  VERIFY_ARITHMETIC_OP(VMUL, op1, op2, output_sse);
  ASSERT_EQ(op1[1] * op2[1], output_sse[1]);

  VERIFY_ARITHMETIC_OP(VDIV, op1, op2, output_sse);
  ASSERT_EQ(op1[3] / op2[3], output_sse[3]);

  VERIFY_ARITHMETIC_OP(VDIV_FAST, op1, op2, output_sse);
  ASSERT_EQ(op1[3] / op2[3], output_sse[3]);

  VERIFY_ARITHMETIC_OP(VMIN, op1, op2, output_sse);
  ASSERT_EQ(std::min(op1[2], op2[2]), output_sse[2]);

  VERIFY_ARITHMETIC_OP(VMAX, op1, op2, output_sse);
  ASSERT_EQ(std::max(op1[2], op2[2]), output_sse[2]);

  VERIFY_ARITHMETIC_OP(VBITWISE_OR, op1, op2, output_sse);
  ASSERT_EQ(op1[1] | op2[1], output_sse[1]);

  VERIFY_ARITHMETIC_OP(VBITWISE_AND, op1, op2, output_sse);
  ASSERT_EQ(op1[1] & op2[1], output_sse[1]);

  VERIFY_UNARY_OP(VABS, op1, output_sse);
  ASSERT_EQ(-op1[2], output_sse[2]);
}

TEST(LopperTestVector, BitMath) {
  int32_t input[4] = {123, 456, -324, 199};
  int32_t output[4];

  VSTORE(output, VSHIFTR<2>(VLOAD<LOPPER_TARGET>(input)));
  for (size_t i = 0; i < 4; i++) {
    ASSERT_EQ(input[i] >> 2, output[i]);
    ASSERT_EQ(input[i] >> 2, VSHIFTR<2>(input[i]));
  }

  VSTORE(output, VSHIFTL<2>(VLOAD<LOPPER_TARGET>(input)));
  for (size_t i = 0; i < 4; i++) {
    ASSERT_EQ(input[i] << 2, output[i]);
    ASSERT_EQ(input[i] << 2, VSHIFTL<2>(input[i]));
  }
}

TEST(LopperTestVector, Conversion) {
  float buffer_fp[4] = {100.3f, -38.f, 10.501f, -38.49f};
  int32_t buffer_int[4];

  VSTORE(buffer_int, VTO_INT32<LOPPER_TARGET>(VLOAD<LOPPER_TARGET>(buffer_fp)));
  for (size_t i = 0; i < 4; i++) {
    ASSERT_EQ(buffer_int[i], VTO_INT32<SCALAR>(VLOAD<SCALAR>(buffer_fp+i)));
  }
  ASSERT_EQ(100, buffer_int[0]);
  ASSERT_EQ(-38, buffer_int[1]);
  ASSERT_EQ(10, buffer_int[2]);
  ASSERT_EQ(-38, buffer_int[3]);

  VSTORE(buffer_fp, VTO_FLOAT<LOPPER_TARGET>(VLOAD<LOPPER_TARGET>(buffer_int)));
  for (size_t i = 0; i < 4; i++) {
    ASSERT_EQ(buffer_fp[i], VTO_INT32<SCALAR>(VLOAD<SCALAR>(buffer_int+i)));
  }
  ASSERT_NEAR(100.f, buffer_fp[0], 1e-5);
  ASSERT_NEAR(-38.f, buffer_fp[1], 1e-5);
  ASSERT_NEAR(10.f, buffer_fp[2], 1e-5);
  ASSERT_NEAR(-38.f, buffer_fp[3], 1e-5);
}

TEST(LopperTestVector, ExpandCollapse) {
  int32_t input[4] = {0x12345678, -0x08765432, 0x00112233, 0x01020304};
  int32_t output[4][4];
  int32_t expanded[4];
  expanded[0] = VEXPAND_BYTE<SCALAR, 0>(input[0]);
  expanded[1] = VEXPAND_BYTE<SCALAR, 1>(input[0]);
  expanded[2] = VEXPAND_BYTE<SCALAR, 2>(input[0]);
  expanded[3] = VEXPAND_BYTE<SCALAR, 3>(input[0]);
  ASSERT_EQ(0x78, expanded[0]);
  ASSERT_EQ(0x56, expanded[1]);
  ASSERT_EQ(0x34, expanded[2]);
  ASSERT_EQ(0x12, expanded[3]);

  auto tmp = VLOAD<LOPPER_TARGET>(input);
  VSTORE(output[0], VEXPAND_BYTE<LOPPER_TARGET, 0>(tmp));
  VSTORE(output[1], VEXPAND_BYTE<LOPPER_TARGET, 1>(tmp));
  VSTORE(output[2], VEXPAND_BYTE<LOPPER_TARGET, 2>(tmp));
  VSTORE(output[3], VEXPAND_BYTE<LOPPER_TARGET, 3>(tmp));
  ASSERT_EQ(0x78, output[0][0]);
  ASSERT_EQ(0x56, output[0][1]);
  ASSERT_EQ(0x34, output[0][2]);
  ASSERT_EQ(0x12, output[0][3]);
  ASSERT_EQ(0x33, output[2][0]);
  ASSERT_EQ(0x22, output[2][1]);
  ASSERT_EQ(0x11, output[2][2]);
  ASSERT_EQ(0x00, output[2][3]);

  int32_t collapsed[4];
  VSTORE(collapsed, VCOLLAPSE_TO_BYTES<LOPPER_TARGET>(VLOAD<LOPPER_TARGET>(output[0]),
                                                      VLOAD<LOPPER_TARGET>(output[1]),
                                                      VLOAD<LOPPER_TARGET>(output[2]),
                                                      VLOAD<LOPPER_TARGET>(output[3])));
  for (size_t i = 0; i < 4; i++) {
    ASSERT_EQ(input[i], collapsed[i]);
  }
}

TEST(LopperTestVector, Shuffle) {
  uint8_t indices[16] = {1,0,255u,4,5,6,7,8,9,10,11,12,13,14,255u,0};
  uint8_t data[16] = {100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115};
  uint8_t output[16];

  auto vec_indices = VLOAD<LOPPER_TARGET>(indices);
  auto vec_data = VLOAD<LOPPER_TARGET>(data);
  VSTORE(output, VSHUFFLE<LOPPER_TARGET>(vec_data, vec_indices));

  const size_t num_bytes = 4 * InstructionSetTrait<LOPPER_TARGET>::num_lanes;
  for (size_t i = 0; i < num_bytes; i++) {
    ASSERT_EQ(indices[i] < num_bytes ? data[indices[i]] : 0, output[i]);
  }
}

TEST(LopperTestVector, Comparison) {
  int32_t v0[4] = {1, 2, 3, 4};
  int32_t v1[4] = {2, 1, 4, 4};
  float v2[4] = {1.f, 2.f, 3.f, 4.f};
  float v3[4] = {2.f, 1.f, 4.f, 4.f};
  bool result_nonzero_lt[4] = {true, false, true, false};
  bool result_nonzero_eq[4] = {false, false, false, true};
  int32_t result_computed_lt[4];
  int32_t result_computed_eq[4];

  // Compare int32_t.
  for (size_t i = 0; i < 4; i+=InstructionSetTrait<LOPPER_TARGET>::num_lanes) {
    VSTORE(result_computed_lt + i,
           VLT<LOPPER_TARGET>(VLOAD<LOPPER_TARGET>(v0 + i), VLOAD<LOPPER_TARGET>(v1 + i)));
    VSTORE(result_computed_eq + i,
           VEQ<LOPPER_TARGET>(VLOAD<LOPPER_TARGET>(v0 + i), VLOAD<LOPPER_TARGET>(v1 + i)));
  }
  // Check result.
  for (size_t i = 0; i < 4; i++) {
    ASSERT_EQ(result_nonzero_lt[i], result_computed_lt[i] != 0);
    ASSERT_EQ(result_nonzero_eq[i], result_computed_eq[i] != 0);
  }
  // Compare float32.
  for (size_t i = 0; i < 4; i+=InstructionSetTrait<LOPPER_TARGET>::num_lanes) {
    VSTORE(result_computed_lt + i,
           VLT<LOPPER_TARGET>(VLOAD<LOPPER_TARGET>(v2 + i), VLOAD<LOPPER_TARGET>(v3 + i)));
    VSTORE(result_computed_eq + i,
           VEQ<LOPPER_TARGET>(VLOAD<LOPPER_TARGET>(v2 + i), VLOAD<LOPPER_TARGET>(v3 + i)));
  }
  // Check result.
  for (size_t i = 0; i < 4; i++) {
    ASSERT_EQ(result_nonzero_lt[i], result_computed_lt[i] != 0);
    ASSERT_EQ(result_nonzero_eq[i], result_computed_eq[i] != 0);
  }
}

TEST(LopperTestVector, Select) {
  int32_t selector[4] = {123, 0, -998, 1};
  int32_t lhs1[4] = {0x12345678, 0x21436587, 0x12191919, 0x21212121};
  int32_t rhs1[4] = {0x11111111, 0x22222222, 0x01234567, 0x13131313};
  float lhs2[4] = {123.f, 456.f, -123.f, -456.f};
  float rhs2[4] = {999.f, 888.f, 777.f, -666.f};
  int32_t result_computed1[4];
  float result_computed2[4];
  // Select int32_t.
  for (size_t i = 0; i < 4; i+=InstructionSetTrait<LOPPER_TARGET>::num_lanes) {
    VSTORE(result_computed1 + i,
           VSELECT<LOPPER_TARGET>(VLOAD<LOPPER_TARGET>(selector + i),
                                  VLOAD<LOPPER_TARGET>(lhs1 + i),
                                  VLOAD<LOPPER_TARGET>(rhs1 + i)));
  }
  // Check result.
  for (size_t i = 0; i < 4; i++) {
    ASSERT_EQ(selector[i] == 0 ? lhs1[i] : rhs1[i], result_computed1[i]);
  }
  // Select float.
  for (size_t i = 0; i < 4; i+=InstructionSetTrait<LOPPER_TARGET>::num_lanes) {
    VSTORE(result_computed2 + i,
           VSELECT<LOPPER_TARGET>(VLOAD<LOPPER_TARGET>(selector + i),
                                  VLOAD<LOPPER_TARGET>(lhs2 + i),
                                  VLOAD<LOPPER_TARGET>(rhs2 + i)));
  }
  // Check result.
  for (size_t i = 0; i < 4; i++) {
    ASSERT_EQ(selector[i] == 0 ? lhs2[i] : rhs2[i], result_computed2[i]);
  }
}
#endif
