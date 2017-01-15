#include <cstring>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <gtest/gtest.h>

#include "lopper/primitives.hpp"

using namespace lopper;

class LopperPrimitiveTest : public :: testing::Test {};
template<typename T> class LopperTypedPrimitiveTest : public LopperPrimitiveTest {};
TYPED_TEST_CASE_P(LopperTypedPrimitiveTest);

TEST(LopperTestVector, LittleEndianTest) {
  // Make sure we're in little Endian, as some of the routines assume this explicitly.
  // While the corresponding tests will fail, we'll have an explicit test here.
  auto v = VSET<LOPPER_TARGET>(4 | (5 << 8) | (6 << 16) | (7 << 24));
  uint8_t buffer_u8[LOPPER_BITWIDTH / 8] = {0};
  VSTORE(buffer_u8, v);
  ASSERT_EQ(4, buffer_u8[0]); // Big-endian would give 7 here.
}

TYPED_TEST_P(LopperTypedPrimitiveTest, SetTest) {
  uint8_t buffer_u8[TypeParam::bitwidth / 8] = {0};
  int32_t buffer_s32[TypeParam::num_lanes] = {0};
  // Try setting a vector from a literal.
  auto v = VSET8x16<TypeParam::value>(0, 1, 2, 3, 4, 5, 6, 7,
                                   8, 9, 10, 11, 12, 13, 14, 15);
  VSTORE(buffer_u8, v);
  for (size_t y = 0; y < std::min<size_t>(16, TypeParam::bitwidth / 8); y++) {
    ASSERT_EQ((uint8_t)y, buffer_u8[y]);
  }
  // Try setting a vector to a 32-bit input for all lanes.
  v = VSET<TypeParam::value>(12345);
  VSTORE(buffer_s32, v);
  for (size_t y = 0; y < TypeParam::num_lanes; y++) {
    ASSERT_EQ(12345, buffer_s32[y]);
  }
  // Try setting with VSET4x4 if applicable
  v = VSET4x4<TypeParam::value>(35, 69, -91, -29472);
  VSTORE(buffer_s32, v);
  ASSERT_EQ(35, buffer_s32[0]);
  if (TypeParam::num_lanes >= 4u) {
    ASSERT_EQ(69, buffer_s32[1]);
    ASSERT_EQ(-91, buffer_s32[2]);
    ASSERT_EQ(-29472, buffer_s32[3]);
  }
}

TYPED_TEST_P(LopperTypedPrimitiveTest, ExponentiationTest) {
  // Test VLDEXP and VFREXP.
  std::vector<float> inputs = {-32.1f, 1.0f, 0.4f, 0.001f, 0.9f, 1.98f, 3.99f, 123.45f};
  for (auto& input : inputs) {
    float buffer_in[TypeParam::num_lanes];
    for (size_t i = 0; i < TypeParam::num_lanes; i++) {
      buffer_in[i] = input + i;
    }
    float buffer_fout[TypeParam::num_lanes] = {0.f};
    int32_t buffer_iout[TypeParam::num_lanes] = {0};
    // Test VLDEXP.
    VSTORE(buffer_fout, VLDEXP<TypeParam::value>(VLOAD<TypeParam::value>(buffer_in), VSET<TypeParam::value>(3)));
    for (size_t i = 0; i < TypeParam::num_lanes; i++) {
      ASSERT_EQ(VLDEXP<SCALAR>(buffer_in[i], 3), buffer_fout[i]);
    }
    // Test VFREXP.
    auto viexp = VSET<TypeParam::value>(0);
    int32_t iexp[4];
    VSTORE(buffer_fout, VFREXP<TypeParam::value>(VLOAD<TypeParam::value>(buffer_in), viexp));
    VSTORE(buffer_iout, viexp);
    for (size_t i = 0; i < TypeParam::num_lanes; i++) {
      ASSERT_EQ(VFREXP<SCALAR>(buffer_in[i], iexp[i]), buffer_fout[i]);
    ASSERT_EQ(iexp[i], buffer_iout[i]);
    }
  }
}

TYPED_TEST_P(LopperTypedPrimitiveTest, LoadTest) {
  float buffer_in[8] = {1.5f, 2.5f, -3.9f, 100.1f, -2.1f, 9.9f, 10.f, 1.1f};
  int32_t indices[8] = {1, 2, 3, 4, 5, 6, 7, 0};
  float buffer_out[8];
  typename InstructionSetTrait<TypeParam::value>::FLOAT tmp =
    VLOOKUP_FP_ARRAY<TypeParam::value>(buffer_in, VLOAD<TypeParam::value>(indices));
  int32_t modulus = (int32_t)(TypeParam::num_lanes);
  ASSERT_EQ(buffer_in[indices[0 % modulus]], VGET_LANE<0>(tmp));
  ASSERT_EQ(buffer_in[indices[1 % modulus]], VGET_LANE<1>(tmp));
  ASSERT_EQ(buffer_in[indices[2 % modulus]], VGET_LANE<2>(tmp));
  ASSERT_EQ(buffer_in[indices[3 % modulus]], VGET_LANE<3>(tmp));
  ASSERT_EQ(buffer_in[indices[4 % modulus]], VGET_LANE<4>(tmp));
  ASSERT_EQ(buffer_in[indices[5 % modulus]], VGET_LANE<5>(tmp));
  ASSERT_EQ(buffer_in[indices[6 % modulus]], VGET_LANE<6>(tmp));
  ASSERT_EQ(buffer_in[indices[7 % modulus]], VGET_LANE<7>(tmp));

  VSTORE(buffer_out, tmp);
  for (size_t i = 0; i < TypeParam::num_lanes; i++) {
    ASSERT_EQ(VLOOKUP_FP_ARRAY<SCALAR>(buffer_in, indices[i]), buffer_out[i]);
  }
  for (size_t i = 0; i < TypeParam::num_lanes; i++) {
    ASSERT_EQ(buffer_in[indices[i]], buffer_out[i]);
  }
}

TYPED_TEST_P(LopperTypedPrimitiveTest, LoadUInt8IntoInt32Test) {
  uint8_t buffer_in[32];
  for (int i = 0; i < 32; i++) {
    buffer_in[i] = (uint8_t)i;
  }
  uint8_t buffer_out[TypeParam::bitwidth / 8];
  VSTORE(buffer_out, VLOAD<TypeParam::value>(buffer_in));
  for (size_t i = 0; i < TypeParam::bitwidth / 8; i++) {
    ASSERT_EQ(buffer_out[i], buffer_in[i]);
  }
  VSTORE(buffer_out, VLOAD<TypeParam::value>(buffer_in + 1));
  for (size_t i = 0; i < TypeParam::bitwidth / 8; i++) {
    ASSERT_EQ(buffer_out[i], buffer_in[i + 1]);
  }
}

template<typename T, size_t C, typename TypeParam> void _MultiStoreTestHelper() {
  T buffer_in[C][(TypeParam::bitwidth / 8 / sizeof(T))];
  for (size_t i = 0; i < TypeParam::bitwidth / 8 / sizeof(T); i++) {
    for (size_t c = 0; c < C; c++) {
      buffer_in[c][i] = ((i+0+10*c) | ((i+1+10*c) << 8) | ((i+2+10*c) << 16) | ((i+3+10*c) << 24));
    }
  }
  T buffer_out[(TypeParam::bitwidth / 8 / sizeof(T)) * C];
  switch (C) {
  case 3:
    VSTORE3(buffer_out,
            VLOAD<TypeParam::value>(buffer_in[0%C]),
            VLOAD<TypeParam::value>(buffer_in[1%C]),
            VLOAD<TypeParam::value>(buffer_in[2%C]));
    break;
  case 4:
    VSTORE4(buffer_out,
            VLOAD<TypeParam::value>(buffer_in[0%C]),
            VLOAD<TypeParam::value>(buffer_in[1%C]),
            VLOAD<TypeParam::value>(buffer_in[2%C]),
            VLOAD<TypeParam::value>(buffer_in[3%C]));
    break;
  default:
    ASSERT_FALSE(true);
  }
  for (size_t i = 0; i < TypeParam::bitwidth / 8 / sizeof(T); i++) {
    for (size_t c = 0; c < C; c++) {
      ASSERT_EQ(buffer_in[c][i], buffer_out[i*C+c]);
    }
  }
}

TYPED_TEST_P(LopperTypedPrimitiveTest, MultiStoreTest) {
  _MultiStoreTestHelper<uint8_t, 3, TypeParam>();
  _MultiStoreTestHelper<uint8_t, 4, TypeParam>();
  _MultiStoreTestHelper<int32_t, 3, TypeParam>();
  _MultiStoreTestHelper<int32_t, 4, TypeParam>();
  _MultiStoreTestHelper<float, 3, TypeParam>();
  _MultiStoreTestHelper<float, 4, TypeParam>();
}

// A macro for checking SIMD and serial behaviors against each other
#define VERIFY_ARITHMETIC_OP(OP, arg1, arg2, output)                                         \
  for (size_t i = 0; i < 8; i += TypeParam::num_lanes) {                                     \
    VSTORE(output+i, OP(VLOAD<TypeParam::value>(arg1+i), VLOAD<TypeParam::value>(arg2+i)));  \
    for (size_t j = 0; j < TypeParam::num_lanes; j++) {                                      \
      ASSERT_EQ(output[i+j], OP(VLOAD<SCALAR>(arg1+i+j), VLOAD<SCALAR>(arg2+i+j)));          \
    }                                                                                        \
  }
#define VERIFY_UNARY_OP(OP, arg1, output)                   \
  for (size_t i = 0; i < 8; i += TypeParam::num_lanes) {    \
    VSTORE(output+i, OP(VLOAD<TypeParam::value>(arg1+i)));  \
    for (size_t j = 0; j < TypeParam::num_lanes; j++) {     \
      ASSERT_EQ(output[i+j], OP(VLOAD<SCALAR>(arg1+i+j)));  \
    }                                                        \
  }

TYPED_TEST_P(LopperTypedPrimitiveTest, FloatingPointMath) {
  float input1[8] = {1.5f, 2.5f, -3.9f, 100.1f, -1.5f, -2.5f, 3.9f, -101.1f};
  float input2[8] = {12.5f, -2.5f, 33.9f, 0.13f, 12.5f, -2.5f, 33.9f, 0.13f};
  float output[8];

  for (size_t j = 0; j < 8; j += TypeParam::num_lanes) {
    const float* op1 = input1 + j;
    const float* op2 = input2 + j;
    VERIFY_ARITHMETIC_OP(VADD, op1, op2, output);
    ASSERT_EQ(op1[0] + op2[0], output[0]);

    if (TypeParam::num_lanes >= 4u) {
      // Pairwise add is tested separately because serial version behaves differently.
      VSTORE(output, VADD_PAIRWISE(VLOAD<TypeParam::value>(op1), VLOAD<TypeParam::value>(op2)));
      ASSERT_EQ(op1[0] + op1[1], output[0]);
      ASSERT_EQ(op1[2] + op1[3], output[1]);
      ASSERT_EQ(op2[0] + op2[1], output[2]);
      ASSERT_EQ(op2[2] + op2[3], output[3]);
    }

    VERIFY_ARITHMETIC_OP(VSUB, op1, op2, output);
    ASSERT_EQ(op1[1] - op2[1], output[1]);

    VERIFY_ARITHMETIC_OP(VMUL, op1, op2, output);
    ASSERT_EQ(op1[1] * op2[1], output[1]);

    VERIFY_ARITHMETIC_OP(VDIV, op1, op2, output);
    ASSERT_EQ(op1[3] / op2[3], output[3]);

    // For VDIV_FAST, equality is not guaratneed, so manually compare.
    VSTORE(output, VDIV_FAST(VLOAD<TypeParam::value>(op1), VLOAD<TypeParam::value>(op2)));
    for (size_t i = 0; i < 4; i++) {
      ASSERT_NEAR(1.0f, output[i] / VDIV_FAST(VLOAD<SCALAR>(op1+i), VLOAD<SCALAR>(op2+i)), 0.01f);
    }

    VERIFY_ARITHMETIC_OP(VMIN, op1, op2, output);
    ASSERT_EQ(std::min(op1[2], op2[2]), output[2]);

    VERIFY_ARITHMETIC_OP(VMAX, op1, op2, output);
    ASSERT_EQ(std::max(op1[2], op2[2]), output[2]);

    VERIFY_UNARY_OP(VABS, op1, output);
    ASSERT_EQ(fabs(op1[2]), output[2]);
  }
}

TYPED_TEST_P(LopperTypedPrimitiveTest, IntegerMath) {
  int32_t op1[8] = {123, 456, -324, 199, 124, 451, -124, 7};
  int32_t op2[8] = {348, -123, -234, -1000, 30, -99, 29, -999};
  int32_t output[8];

  VERIFY_ARITHMETIC_OP(VADD, op1, op2, output);
  ASSERT_EQ(op1[0] + op2[0], output[0]);

  VERIFY_ARITHMETIC_OP(VSUB, op1, op2, output);
  ASSERT_EQ(op1[1] - op2[1], output[1]);

  VERIFY_ARITHMETIC_OP(VMUL, op1, op2, output);
  ASSERT_EQ(op1[1] * op2[1], output[1]);

  VERIFY_ARITHMETIC_OP(VDIV, op1, op2, output);
  ASSERT_EQ(op1[3] / op2[3], output[3]);

  VERIFY_ARITHMETIC_OP(VDIV_FAST, op1, op2, output);
  ASSERT_EQ(op1[3] / op2[3], output[3]);

  VERIFY_ARITHMETIC_OP(VMIN, op1, op2, output);
  ASSERT_EQ(std::min(op1[2], op2[2]), output[2]);

  VERIFY_ARITHMETIC_OP(VMAX, op1, op2, output);
  ASSERT_EQ(std::max(op1[2], op2[2]), output[2]);

  VERIFY_ARITHMETIC_OP(VBITWISE_OR, op1, op2, output);
  ASSERT_EQ(op1[1] | op2[1], output[1]);

  VERIFY_ARITHMETIC_OP(VBITWISE_AND, op1, op2, output);
  ASSERT_EQ(op1[1] & op2[1], output[1]);

  VERIFY_UNARY_OP(VABS, op1, output);
  ASSERT_EQ(-op1[2], output[2]);
}

TYPED_TEST_P(LopperTypedPrimitiveTest, BitMath) {
  int32_t input[8] = {123, 456, -324, 199, 273, 119, 827, -9999};
  int32_t output[8];

  constexpr size_t num_lanes = TypeParam::num_lanes;
  for (size_t j = 0; j < 8; j += num_lanes) {
    VSTORE(output, VSHIFTR<2>(VLOAD<TypeParam::value>(input + j)));
    for (size_t i = 0; i < num_lanes; i++) {
      ASSERT_EQ(input[j + i] >> 2, output[i]);
      ASSERT_EQ(input[j + i] >> 2, VSHIFTR<2>(input[j + i]));
    }

    VSTORE(output, VSHIFTL<2>(VLOAD<TypeParam::value>(input + j)));
    for (size_t i = 0; i < num_lanes; i++) {
      ASSERT_EQ(input[j + i] << 2, output[i]);
      ASSERT_EQ(input[j + i] << 2, VSHIFTL<2>(input[j + i]));
    }
  }
}

TYPED_TEST_P(LopperTypedPrimitiveTest, Conversion) {
  float input[8] = {100.3f, -38.f, 10.501f, -38.49f, 88.f, -0.00001f, 9284.2f, -1111.f};
  int32_t reference[8] = {100, -38, 10, -38, 88, 0, 9284, -1111};
  int32_t output_int[8];
  float output_fp[8];

  constexpr size_t num_lanes = TypeParam::num_lanes;
  for (size_t j = 0; j < 8; j += num_lanes) {
    // Convert from float to int.
    VSTORE(output_int + j, VTO_INT32<TypeParam::value>(VLOAD<TypeParam::value>(input + j)));
    for (size_t i = 0; i < num_lanes; i++) {
      ASSERT_EQ(output_int[j + i], VTO_INT32<SCALAR>(VLOAD<SCALAR>(input + j + i)));
      ASSERT_EQ(output_int[j + i], reference[j + i]);
    }
    // Convert from int to float.
    VSTORE(output_fp + j, VTO_FLOAT<TypeParam::value>(VLOAD<TypeParam::value>(reference + j)));
    for (size_t i = 0; i < num_lanes; i++) {
      ASSERT_EQ(output_fp[j + i], VTO_INT32<SCALAR>(VLOAD<SCALAR>(reference + j + i)));
      ASSERT_NEAR(static_cast<float>(reference[j + i]), output_fp[j + i], 1e-5);
    }
  }
}

TYPED_TEST_P(LopperTypedPrimitiveTest, ExpandCollapse) {
  int32_t input[8] = {0x12345678, -0x08765432, 0x00112233, 0x01020304,
                      -0x01020304, 0x09809809, 0x34773613, 0x00098765};
  int32_t expanded[4];
  expanded[0] = VEXPAND_QTR<SCALAR, 0>(input[0]);
  expanded[1] = VEXPAND_QTR<SCALAR, 1>(input[0]);
  expanded[2] = VEXPAND_QTR<SCALAR, 2>(input[0]);
  expanded[3] = VEXPAND_QTR<SCALAR, 3>(input[0]);
  ASSERT_EQ(0x78, expanded[0]);
  ASSERT_EQ(0x56, expanded[1]);
  ASSERT_EQ(0x34, expanded[2]);
  ASSERT_EQ(0x12, expanded[3]);

  auto tmp = VLOAD<TypeParam::value>(input);
  int32_t output[4][8];
  VSTORE(output[0], VEXPAND_QTR<TypeParam::value, 0>(tmp));
  VSTORE(output[1], VEXPAND_QTR<TypeParam::value, 1>(tmp));
  VSTORE(output[2], VEXPAND_QTR<TypeParam::value, 2>(tmp));
  VSTORE(output[3], VEXPAND_QTR<TypeParam::value, 3>(tmp));
  for (int i = 0; i < 4; i++) {
    size_t offset = i * (TypeParam::bitwidth / 8) / 4;
    for (size_t j = 0; j < TypeParam::num_lanes; j++) {
      ASSERT_EQ(reinterpret_cast<uint8_t*>(input)[j + offset], output[i][j]);
    }
  }
  int32_t collapsed[8];
  VSTORE(collapsed, VCOLLAPSE_TO_BYTES<TypeParam::value>(VLOAD<TypeParam::value>(output[0]),
                                                      VLOAD<TypeParam::value>(output[1]),
                                                      VLOAD<TypeParam::value>(output[2]),
                                                      VLOAD<TypeParam::value>(output[3])));
  for (size_t i = 0; i < TypeParam::num_lanes; i++) {
    ASSERT_EQ(input[i], collapsed[i]);
  }
}

TYPED_TEST_P(LopperTypedPrimitiveTest, Shuffle8) {
  uint8_t indices[32], data[32], output[32];
  // Initialize test data.
  for (int i = 0; i < 32; i++) {
    indices[i] = (i * 7 + 15) % 32;
    data[i] = 100 + i;
  }
  indices[8] = indices[15] = indices[24] = 255u; // Set some indices to 0xff.

  const size_t num_bytes = 4u * TypeParam::num_lanes;
  for (size_t j = 0; j < 32; j += num_bytes) {
    auto vec_indices = VLOAD<TypeParam::value>(indices + j);
    auto vec_data = VLOAD<TypeParam::value>(data + j);
    VSTORE(output, VSHUFFLE8<TypeParam::value>(vec_data, vec_indices));

    for (size_t i = 0; i < num_bytes; i++) {
      // NOTE: On NEON, any OOB value gets you zero, whereas on Intel the only LSBs are used.
      if (indices[j + i] == 255u || (indices[j + i] >= 16u && TypeParam::value == NEON)) {
        ASSERT_EQ(0u, output[i]);
      } else {
        ASSERT_EQ(data[j + (indices[j + i] % num_bytes)], output[i]);
      }
    }
  }
}

TYPED_TEST_P(LopperTypedPrimitiveTest, Shuffle32) {
  int32_t indices[8], data[8], output[8];
  // Initialize test data.
  for (int i = 0; i < 8; i++) {
    indices[i] = (i * 3 + 2) % 8;
    data[i] = 100 + i;
  }
  indices[6] = (int32_t)0xffffffff; // Set some indices to -1.

  const size_t num_lanes =  TypeParam::num_lanes;
  for (size_t j = 0; j < 8; j += num_lanes) {
    auto vec_indices = VLOAD<TypeParam::value>(indices + j);
    auto vec_data = VLOAD<TypeParam::value>(data + j);
    VSTORE(output, VSHUFFLE32<TypeParam::value>(vec_data, vec_indices));

    for (size_t i = 0; i < num_lanes; i++) {
      if (indices[j + i] == (int32_t)0xffffffff) {
        ASSERT_EQ(0u, output[i]);
      } else {
        ASSERT_EQ(data[j + (indices[j + i] % num_lanes)], output[i]);
      }
    }
  }
}

TYPED_TEST_P(LopperTypedPrimitiveTest, Comparison) {
  int32_t v0[8] = {1, 2, 3, 4, 10, -9, 30, 11111};
  int32_t v1[8] = {2, 1, 4, 4, 9, -9, 11111, -99999};
  float v2[8] = {1.f, 2.f, 3.f, 4.f, 10.f, -9.f, 30.f, 11111.f};
  float v3[8] = {2.f, 1.f, 4.f, 4.f, 9.f, -9.f, 11111.f, -99999.f};
  bool result_nonzero_lt[8] = {true, false, true, false, false, false, true, false};
  bool result_nonzero_eq[8] = {false, false, false, true, false, true, false, false};
  int32_t result_computed_lt[8];
  int32_t result_computed_eq[8];

  // Compare int32_t.
  for (size_t i = 0; i < 8; i+=TypeParam::num_lanes) {
    VSTORE(result_computed_lt + i,
           VLT<TypeParam::value>(VLOAD<TypeParam::value>(v0 + i), VLOAD<TypeParam::value>(v1 + i)));
    VSTORE(result_computed_eq + i,
           VEQ<TypeParam::value>(VLOAD<TypeParam::value>(v0 + i), VLOAD<TypeParam::value>(v1 + i)));
  }
  // Check result.
  for (size_t i = 0; i < 8; i++) {
    ASSERT_EQ(result_nonzero_lt[i], result_computed_lt[i] != 0);
    ASSERT_EQ(result_nonzero_eq[i], result_computed_eq[i] != 0);
  }
  // Compare float32.
  for (size_t i = 0; i < 8; i+=TypeParam::num_lanes) {
    VSTORE(result_computed_lt + i,
           VLT<TypeParam::value>(VLOAD<TypeParam::value>(v2 + i), VLOAD<TypeParam::value>(v3 + i)));
    VSTORE(result_computed_eq + i,
           VEQ<TypeParam::value>(VLOAD<TypeParam::value>(v2 + i), VLOAD<TypeParam::value>(v3 + i)));
  }
  // Check result.
  for (size_t i = 0; i < 8; i++) {
    ASSERT_EQ(result_nonzero_lt[i], result_computed_lt[i] != 0);
    ASSERT_EQ(result_nonzero_eq[i], result_computed_eq[i] != 0);
  }
}

TYPED_TEST_P(LopperTypedPrimitiveTest, Select) {
  int32_t selector[8] = {123, 0, -998, 1, 0, 0, 11, -35};
  int32_t lhs1[8] = {0x12345678, 0x21436587, 0x12191919, 0x21212121, 0x0, 0x1, 0x2, 0x3};
  int32_t rhs1[8] = {0x11111111, 0x22222222, 0x01234567, 0x13131313, 0x3, 0x2, 0x1, 0x0};
  float lhs2[8] = {123.f, 456.f, -123.f, -456.f, 0.f, 99.9f, 0.001f, -0.001f};
  float rhs2[8] = {999.f, 888.f, 777.f, -666.f, 998.f, -123.f, 1.f, 19383.32f};
  int32_t result_computed1[8];
  float result_computed2[8];
  // Select int32_t.
  for (size_t i = 0; i < 8; i+=TypeParam::num_lanes) {
    VSTORE(result_computed1 + i,
           VSELECT<TypeParam::value>(VLOAD<TypeParam::value>(selector + i),
                                  VLOAD<TypeParam::value>(lhs1 + i),
                                  VLOAD<TypeParam::value>(rhs1 + i)));
  }
  // Check result.
  for (size_t i = 0; i < 8; i++) {
    ASSERT_EQ(selector[i] == 0 ? lhs1[i] : rhs1[i], result_computed1[i]);
  }
  // Select float.
  for (size_t i = 0; i < 8; i+=TypeParam::num_lanes) {
    VSTORE(result_computed2 + i,
           VSELECT<TypeParam::value>(VLOAD<TypeParam::value>(selector + i),
                                  VLOAD<TypeParam::value>(lhs2 + i),
                                  VLOAD<TypeParam::value>(rhs2 + i)));
  }
  // Check result.
  for (size_t i = 0; i < 8; i++) {
    ASSERT_EQ(selector[i] == 0 ? lhs2[i] : rhs2[i], result_computed2[i]);
  }
}

TYPED_TEST_P(LopperTypedPrimitiveTest, InterleaveTest) {
  int32_t v0[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  int32_t v1[8] = {101, 102, 103, 104, 105, 106, 107, 108};
  int32_t result[8];
  for (size_t i = 0; i < 8; i+=TypeParam::num_lanes) {
    const auto _v0 = VLOAD<TypeParam::value>(v0 + i);
    const auto _v1 = VLOAD<TypeParam::value>(v1 + i);
    VSTORE(result, VINTERLEAVE32_LO(_v0, _v1));
    for (size_t j = 0; j < TypeParam::num_lanes; j++) {
      ASSERT_EQ(result[j], (j % 2 == 0 ? v0 : v1)[i+(j>>1)]);
    }
    VSTORE(result, VINTERLEAVE32_HI(_v0, _v1));
    for (size_t j = TypeParam::num_lanes; j < (TypeParam::num_lanes << 1); j++) {
      ASSERT_EQ(result[j - TypeParam::num_lanes], (j % 2 == 0 ? v0 : v1)[i+(j>>1)]);
    }
  }
}

// Instantiate typed tests.
REGISTER_TYPED_TEST_CASE_P(LopperTypedPrimitiveTest,
                           SetTest,
                           ExponentiationTest,
                           LoadTest,
                           LoadUInt8IntoInt32Test,
                           MultiStoreTest,
                           FloatingPointMath,
                           IntegerMath,
                           BitMath,
                           Conversion,
                           ExpandCollapse,
                           Shuffle8,
                           Shuffle32,
                           Comparison,
                           Select,
                           InterleaveTest);

template<InstructionSet S> struct LopperSettingType {
  static constexpr InstructionSet value = S;
  static constexpr size_t num_lanes = InstructionSetTrait<S>::num_lanes;
  static constexpr size_t bitwidth = InstructionSetTrait<S>::num_lanes * 32u;
};
#ifdef LOPPER_NO_SIMD
typedef testing::Types<LopperSettingType<LOPPER_TARGET>> LopperSettingTypes;
#else
typedef testing::Types<LopperSettingType<LOPPER_TARGET>, LopperSettingType<SCALAR>> LopperSettingTypes;
#endif
INSTANTIATE_TYPED_TEST_CASE_P(TypedTest, LopperTypedPrimitiveTest, LopperSettingTypes);
