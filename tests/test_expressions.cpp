#include <cstring>
#include <memory>
#include <stdlib.h>
#include <vector>
#include <gtest/gtest.h>

#include "lopper/lopper.hpp"

/* A simple image container implementation to be used in the tests.
 * It will have a sentinel at either end of a contiguous buffer. */
template<typename T> class Image : public lopper::_Image<T> {
public:
  static constexpr T SENTINEL = T(3);
  Image(const int channels, const int width, const int height) :
    m_channels(channels), m_width(width), m_height(height), m_data(channels * width * height + 2, T(0)) {
    m_data[0] = m_data[channels * width * height + 1] = SENTINEL;
  }
  ~Image() {
    if (m_data[0] != SENTINEL || m_data[m_channels * m_width * m_height + 1] != SENTINEL) {
      throw lopper::LopperException("Sentinel has been overwritten!");
    }
  }
  virtual int getWidth() const { return m_width; }
  virtual int getHeight() const { return m_height; }
  virtual int getChannelCount() const { return m_channels; }
  virtual T* getRowPointer(const size_t y) { return &m_data[(int)y * m_width * m_channels + 1]; }
  virtual const T* getRowPointer(const size_t y) const { return &m_data[(int)y * m_width * m_channels + 1]; }
  /* A convenience function */
  T& operator()(int x, int y, int c=0) { return m_data[y * m_width * m_channels + x * m_channels + c + 1]; }
private:
  const int m_channels, m_width, m_height;
  std::vector<T> m_data;
};

/* A helper to randomize test images. */
template<typename T> void randomize(Image<T>& image) {
  for (int y = 0; y < image.getHeight(); y++) {
    for (int x = 0; x < image.getWidth() * image.getChannels(); x++) {
      image(x, y) = (rand() % 2 == 0) ? T(0) : T(1);
    }
  }
}

/* A helper to check image equality exactly. */
template<typename T> bool areImagesEqual(const Image<T>& image0, const Image<T>& image1) {
  if (image0.getWidth() != image1.getWidth()) return false;
  if (image0.getHeight() != image1.getHeight()) return false;
  if (image0.getChannelCount() != image1.getChannelCount()) return false;
  for (int y = 0; y < image0.getHeight(); y++) {
    if (0 != memcmp(image0.getRowPointer(y), image1.getRowPointer(y),
                    image0.getWidth() * image0.getChannelCount() * (int)sizeof(T))) {
      return false;
    }
  }
  return true;
}

using namespace lopper;

class LopperTest : public :: testing::Test {};
template<typename T> class LopperTypedTest : public LopperTest {};
TYPED_TEST_CASE_P(LopperTypedTest);

TYPED_TEST_P(LopperTypedTest, SizeValidationTest) {
  Image<float> out(1, 100, 100);
  Image<float> in1(1, 100, 101);
  Image<float> in2(1, 101, 100);
  Image<float> in3(1, 200, 50);
  Image<float> in4(1, 100, 100);
  // Expect exception because of dimension mismatch.
  ASSERT_ANY_THROW((ExprEvalSIMD(TypeParam::value, Expr<1>(out) = Expr<1>(in1))));
  ASSERT_ANY_THROW((ExprEvalSIMD(TypeParam::value, Expr<1>(out) = Expr<1>(in2))));
  ASSERT_ANY_THROW((ExprEvalSIMD(TypeParam::value, Expr<1>(out) = Expr<1>(in3))));
  // This should work.
  ExprEvalSIMD(TypeParam::value, Expr<1>(out) = Expr<1>(in4));
  ASSERT_TRUE(areImagesEqual(out, in4));
  // This should also work, since the input image is reindexed.
  ExprEvalSIMD(TypeParam::value, Expr<1>(out) = Expr<1>(in1).reindex([](int y) { return y & ~1; }));
}

TYPED_TEST_P(LopperTypedTest, ConstTest) {
  Image<float> out(1, 100, 100);
  ExprEvalSIMD(TypeParam::value, Expr<1>(out) = Expr<float>(0.5f));
  for (int y = 0; y < out.getHeight(); y++) {
    for (int x = 0; x < out.getWidth() * out.getChannelCount(); x++) {
      ASSERT_EQ(0.5f, out(x, y));
    }
  }
}

TYPED_TEST_P(LopperTypedTest, RowConstTest) {
  Image<int32_t> out(1, 100, 100);
  ExprEvalSIMD(TypeParam::value, Expr<1>(out) = Expr<int32_t>([](const int y) -> int32_t { return (int32_t)y * 2; }));
  for (int y = 0; y < out.getHeight(); y++) {
    for (int x = 0; x < out.getWidth() * out.getChannelCount(); x++) {
      ASSERT_EQ((int32_t)(y * 2), out(x, y));
    }
  }
}

TYPED_TEST_P(LopperTypedTest, OffsetTest) {
  const int width = 123, height = 234;
  Image<uint8_t> in(1, width, height);
  Image<uint8_t> out(1, width, height);
  // Try translating vertically by 1
  ExprEvalSIMD(TypeParam::value, Expr<1>(out) = Expr<1>(in).offset(0, -1));
  for (int y = 0; y < out.getHeight(); y++) {
    int y_src = std::max<int>(0, y - 1);
    for (int x = 0; x < out.getWidth(); x++) {
      ASSERT_EQ(in(x, y_src), out(x, y));
    }
  }
  // Try translating vertically by -1
  ExprEvalSIMD(TypeParam::value, Expr<1>(out) = Expr<1>(in).offset(0, 1));
  for (int y = 0; y < out.getHeight(); y++) {
    int y_src = std::min<int>(in.getHeight() - 1, y + 1);
    for (int x = 0; x < out.getWidth(); x++) {
      ASSERT_EQ(in(x, y_src), out(x, y));
    }
  }
  // Try translating vertically twice, for a total of 5 rows.
  ExprEvalSIMD(TypeParam::value, Expr<1>(out) = Expr<1>(in).offset(0, -3).offset(0, -2));
  for (int y = 0; y < out.getHeight(); y++) {
    int y_src = std::max<int>(0, y - 5);
    for (int x = 0; x < out.getWidth(); x++) {
      ASSERT_EQ(in(x, y_src), out(x, y));
    }
  }
  // Try using translation within an expression
  ExprEvalSIMD(TypeParam::value, Expr<1>(out) = Expr<1>(in).offset(0, -3).offset(0, -2) + 100);
  for (int y = 0; y < out.getHeight(); y++) {
    int y_src = std::max<int>(0, y - 5);
    for (int x = 0; x < out.getWidth(); x++) {
      ASSERT_EQ(in(x, y_src) + 100, out(x, y));
    }
  }
  // Try translating horizontally by -1
  ExprEvalSIMD(TypeParam::value, Expr<1>(out) = Expr<1>(in).offset(-1, 0));
  for (int y = 0; y < out.getHeight(); y++) {
    for (int x = 0; x < out.getWidth(); x++) {
      int x_src = std::max<int>(0, x - 1);
      ASSERT_EQ(in(x_src, y), out(x, y));
    }
  }
  // Try translating horizontally by -30
  ExprEvalSIMD(TypeParam::value, Expr<1>(out) = Expr<1>(in).offset(-30, 0));
  for (int y = 0; y < out.getHeight(); y++) {
    for (int x = 0; x < out.getWidth(); x++) {
      int x_src = std::max<int>(0, x - 30);
      ASSERT_EQ(in(x_src, y), out(x, y));
    }
  }
  // Try translating horizontally by 1
  ExprEvalSIMD(TypeParam::value, Expr<1>(out) = Expr<1>(in).offset(1, 0));
  for (int y = 0; y < out.getHeight(); y++) {
    for (int x = 0; x < out.getWidth(); x++) {
      int x_src = std::min<int>(in.getWidth() - 1, x + 1);
      ASSERT_EQ(in(x_src, y), out(x, y));
    }
  }
  // Try translating horizontally by 30
  ExprEvalSIMD(TypeParam::value, Expr<1>(out) = Expr<1>(in).offset(30, 0));
  for (int y = 0; y < out.getHeight(); y++) {
    for (int x = 0; x < out.getWidth(); x++) {
      int x_src = std::min<int>(in.getWidth() - 1, x + 30);
      ASSERT_EQ(in(x_src, y), out(x, y));
    }
  }
  // Try translating horizontally twice, to cancel out
  ExprEvalSIMD(TypeParam::value, Expr<1>(out) = Expr<1>(in).offset(-3, 0).offset(3, 0));
  ASSERT_TRUE(areImagesEqual(in, out));
  // Try translating two images separately, to check interaction.
  ExprEvalSIMD(TypeParam::value, Expr<1>(out) = Expr<1>(in).offset(2, 0) - Expr<1>(in).offset(1, 0));
  for (int y = 0; y < out.getHeight(); y++) {
    for (int x = 0; x < out.getWidth(); x++) {
      auto expected = in(std::min(x+2, in.getWidth()-1), y) - in(std::min(x+1, in.getWidth()-1), y);
      ASSERT_EQ(expected, out(x, y));
    }
  }
}

TYPED_TEST_P(LopperTypedTest, GradientTest) {
  const int width = 80, height = 50;
  Image<uint8_t> input(1, width, height);
  Image<uint8_t> reference_forward_dx(1, width, height);
  Image<uint8_t> reference_forward_dy(1, width, height);
  Image<uint8_t> reference_backward_dx(1, width, height);
  Image<uint8_t> reference_backward_dy(1, width, height);
  for (int y = 0; y < input.getHeight(); y++) {
    for (int x = 0; x < input.getWidth(); x++) {
      input(x, y) = x + x + y; // di/dx = 2, di/dy = 1
      reference_forward_dx(x, y) = (x == input.getWidth() - 1) ? 0 : 2;
      reference_forward_dy(x, y) = (y == input.getHeight() - 1) ? 0 : 1;
      reference_backward_dx(x, y) = (x == 0) ? 0 : 2;
      reference_backward_dy(x, y) = (y == 0) ? 0 : 1;
    }
  }
  Image<uint8_t> output(1, width, height);
  ExprEvalSIMD(TypeParam::value, Expr<1>(output) = Expr<1>(input).offset(1, 0) - Expr<1>(input));
  ASSERT_TRUE(areImagesEqual(reference_forward_dx, output));
  ExprEvalSIMD(TypeParam::value, Expr<1>(output) = Expr<1>(input) - Expr<1>(input).offset(-1, 0));
  ASSERT_TRUE(areImagesEqual(reference_backward_dx, output));
  ExprEvalSIMD(TypeParam::value, Expr<1>(output) = Expr<1>(input).offset(0, 1) - Expr<1>(input));
  ASSERT_TRUE(areImagesEqual(reference_forward_dy, output));
  ExprEvalSIMD(TypeParam::value, Expr<1>(output) = Expr<1>(input) - Expr<1>(input).offset(0, -1));
  ASSERT_TRUE(areImagesEqual(reference_backward_dy, output));
}

TYPED_TEST_P(LopperTypedTest, ReindexTest) {
  const int width = 123, height = 233;
  Image<uint8_t> input(1, width, height);
  Image<uint8_t> output(1, width, height);
  // Try flipping vertically.
  auto flipper = [height](int x) { return (int)height - 1 - x; };
  auto even_rows = [](int x) { return x - (x&1); };
  ExprEvalSIMD(TypeParam::value, Expr<1>(output) = Expr<1>(input).reindex(flipper));
  for (int y = 0; y < input.getHeight(); y++) {
    for (int x = 0; x < input.getWidth(); x++) {
      ASSERT_EQ(input(x, height - 1 - y), output(x, y));
    }
  }
  // Try flipping vertically twice, which should equal the input.
  ExprEvalSIMD(TypeParam::value, Expr<1>(output) = Expr<1>(input).reindex(flipper).reindex(flipper));
  ASSERT_TRUE(areImagesEqual(output, input));
  // Try dropping the least significant bit and then flipping vertically.
  // These two do not transpose because height is odd, so it will test the composition ordering.
  //  Original   after flip   after even_rows
  //     0           4             4
  //     1           3             4  ("1" gets mapped to "0", which is really row 4 of original)
  //     2   ====>   2   ====>     2
  //     3           1             2
  //     4           0             0
  ExprEvalSIMD(TypeParam::value, Expr<1>(output) = Expr<1>(input).reindex(flipper).reindex(even_rows));
  for (int y = 0; y < input.getHeight(); y++) {
    for (int x = 0; x < input.getWidth(); x++) {
      ASSERT_EQ(input(x, (height - 1 - y) & ~1), output(x, y));
    }
  }
}

TYPED_TEST_P(LopperTypedTest, ReindexOffsetTest) {
  // The input has the following form:
  //   1  2  3  4
  //   5  6  7  8
  //   9 10 11 12
  Image<uint8_t> input(1, 4, 3);
  Image<uint8_t> output(1, 4, 3);
  for (int y = 0; y < 3; y++) {
    for (int x = 0; x < 4; x++) {
      input(x, y) = y * 4 + x + 1;
    }
  }
  // If we flip vertically, and then move it down by 1, the image would look as follows:
  //   9 10 11 12
  //   9 10 11 12
  //   5  6  7  8
  auto flipper = [](int x) { return 2 - x; };
  {
    ExprEvalSIMD(TypeParam::value, Expr<1>(output) = Expr<1>(input).reindex(flipper).offset(0, 1));
    for (int y = 0; y < 3; y++) {
      for (int x = 0; x < 4; x++) {
        ASSERT_EQ((y == 2 ? 5 : 9) + x, output(x, y));
      }
    }
  }
  // If we move the image down by 1, and then flip vertically, the image would look as follows:
  //   5  6  7  8
  //   1  2  3  4
  //   1  2  3  4
  {
    ExprEvalSIMD(TypeParam::value, Expr<1>(output) = Expr<1>(input).offset(0, 1).reindex(flipper));
    for (int y = 0; y < 3; y++) {
      for (int x = 0; x < 4; x++) {
        ASSERT_EQ((y == 0 ? 5 : 1) + x, output(x, y));
      }
    }
  }
}

TYPED_TEST_P(LopperTypedTest, MinimumMaximumTest) {
  Image<uint8_t> in1(1, 100, 100);
  Image<uint8_t> in2(1, 100, 100);
  Image<uint8_t> in3(1, 100, 100);
  Image<uint8_t> out1(1, 100, 100);
  Image<uint8_t> out2(1, 100, 100);
  // Test ExprMin and ExprMax
  ExprEval(Expr<1>(out1) = ExprMin(Expr<1>(in1), Expr<1>(in2)));
  ExprEval(Expr<1>(out2) = ExprMax(Expr<1>(in1), Expr<1>(in2)));
  for (int y = 0; y < in1.getHeight(); y++) {
    for (int x = 0; x < in1.getWidth(); x++) {
      ASSERT_EQ(std::min(in1(x, y), in2(x, y)), out1(x, y));
      ASSERT_EQ(std::max(in1(x, y), in2(x, y)), out2(x, y));
    }
  }
  // Test ExprMin and ExprMax chaining
  ExprEval(Expr<1>(out1) = ExprMin(Expr<1>(in1), Expr<1>(in2), Expr<1>(in3)));
  ExprEval(Expr<1>(out2) = ExprMax(Expr<1>(in1), Expr<1>(in2), Expr<1>(in3)));
  for (int y = 0; y < in1.getHeight(); y++) {
    for (int x = 0; x < in1.getWidth(); x++) {
      ASSERT_EQ(std::min(std::min(in1(x, y), in2(x, y)), in3(x, y)), out1(x, y));
      ASSERT_EQ(std::max(std::max(in1(x, y), in2(x, y)), in3(x, y)), out2(x, y));
    }
  }
}

TYPED_TEST_P(LopperTypedTest, CacheTest) {
  Image<float> in(1, 100, 100);
  Image<float> out(1, 100, 100);
  // Compute some expressin using caches.
  ExprPrepareContext();
  auto i = ExprCache(Expr<1>(in));
  auto j = ExprCache(i + i);
  ExprEvalWithContextSIMD(TypeParam::value, Expr<1>(out) = j * j);
  // Manually compute the same.
  for (int y = 0; y < in.getHeight(); y++) {
    for (int x = 0; x < in.getWidth(); x++) {
      auto i = in(x, y);
      auto j = i + i;
      EXPECT_NEAR(j * j, out(x, y), 1e-6);
    }
  }
}

TYPED_TEST_P(LopperTypedTest, ThreeChannelTest) {
  Image<uint8_t> in(3, 100, 100);
  for (int y = 0; y < in.getHeight(); y++) {
    for (int x = 0; x < in.getWidth(); x++) {
      in(x, y, 0) = 99;
      in(x, y, 1) = 11;
      in(x, y, 2) = 222;
    }
  }
  Image<int32_t> out(1, 100, 100);
  // Try computing R + G * B
  ExprPrepareContext();
  auto rgb = ExprCache(Expr<3>(in));
  auto r = rgb.get<0>();
  auto g = rgb.get<1>();
  auto b = rgb.get<2>();
  auto tmp = ExprCache(r + (g * b));
  ExprEvalWithContextSIMD(TypeParam::value, Expr<1>(out) = tmp + tmp);
  for (int y = 0; y < in.getHeight(); y++) {
    for (int x = 0; x < in.getWidth(); x++) {
      int32_t r = in(x, y, 0);
      int32_t g = in(x, y, 1);
      int32_t b = in(x, y, 2);
      ASSERT_EQ((r + g * b) * 2, out(x, y, 0));
    }
  }
}

TYPED_TEST_P(LopperTypedTest, RerunTest) {
  Image<float> in(1, 100, 100);
  Image<float> out1(1, 100, 100);
  Image<float> out2(1, 100, 100);
  ExprPrepareContext();
  auto a = ExprCache(Expr<1>(in));
  auto b = ExprCache(a + a);
  ExprEvalWithContextSIMD(TypeParam::value, Expr<1>(out1) = b * b);
  ExprEvalWithContextSIMD(TypeParam::value, Expr<1>(out2) = b * b);
  ASSERT_TRUE(areImagesEqual(out1, out2));
}

TYPED_TEST_P(LopperTypedTest, ScopeTest) {
  Image<float> in(1, 100, 100);
  Image<float> out1(1, 100, 100);
  Image<float> out2(1, 100, 100);
  {
    ExprPrepareContext();
    auto a = ExprCache(Expr<1>(in));
    auto b = ExprCache(a + a);
    ExprEvalWithContextSIMD(TypeParam::value, Expr<1>(out1) = b * b);
  }
  {
    ExprPrepareContext();
    auto a = ExprCache(Expr<1>(in));
    auto b = ExprCache(a * a);
    ExprEvalWithContextSIMD(TypeParam::value, Expr<1>(out2) = b + b);
  }
  for (int y = 0; y < in.getHeight(); y++) {
    for (int x = 0; x < in.getWidth(); x++) {
      auto val = in(x, y);
      EXPECT_NEAR((val + val) * (val + val), out1(x, y), 1e-6);
      EXPECT_NEAR((val * val) + (val * val), out2(x, y), 1e-6);
    }
  }
}

TYPED_TEST_P(LopperTypedTest, RGBToHSVTest) {
  Image<uint8_t> in(3, 55, 80);
  Image<uint8_t> out(3, 55, 80);
  in(0, 0, 0) = 255u; in(0, 0, 1) = 0u; in(0, 0, 2) = 0u; // #ff0000
  in(30, 40, 0) = 0u; in(30, 40, 1) = 255u; in(30, 40, 2) = 0u; // #00ff00
  in(54, 79, 0) = 0u; in(54, 79, 1) = 0u; in(54, 79, 2) = 255u; // #0000ff
  in(1, 0, 0) = 0u; in(1, 0, 1) = 0u; in(1, 0, 2) = 0u; // #000000
  ExprPrepareContext();
  auto rgb = ExprCache(Expr<3>(in));
  auto r = rgb.get<0>();
  auto g = rgb.get<1>();
  auto b = rgb.get<2>();
  auto cmax = ExprCache(ExprMax(r, ExprMax(g, b)));
  auto delta = ExprCache(cmax - ExprMin(r, ExprMin(g, b)));
  auto i1 = ExprCache(ExprIf(r == cmax, g, ExprIf(g == cmax, b, r)));
  auto q = Expr<float>(i1 + i1 - (r + g + b) + cmax) / Expr<float>(delta); // compute (??-??)/delta
  auto offset = ExprIf(r == cmax, Expr<float>(0.f), ExprIf(g == cmax, Expr<float>(2.f), Expr<float>(4.f)));
  auto h_fp = ExprCache(ExprIf(delta == 0, Expr<float>(0.f), (q + offset) * (256.f / 6.f)));
  auto s = ExprIf(cmax < Expr<1>(), Expr<0>(), Expr<int32_t>(Expr<float>(delta * 255) / Expr<float>(cmax) + 0.5f));
  auto h = Expr<int32_t>(ExprIf(h_fp < -0.5f, h_fp + 256.5f, h_fp + 0.5f));
  ExprEvalWithContext(Expr<3>(out) = std::make_tuple(h, s, cmax));
  // #ff0000 should map to (0, 255, 255)
  ASSERT_EQ(0u, out(0, 0, 0));
  ASSERT_EQ(255u, out(0, 0, 1));
  ASSERT_EQ(255u, out(0, 0, 2));
  // #00ff00 should map to roughly (85, 255, 255)
  ASSERT_EQ(85u, out(30, 40, 0));
  ASSERT_EQ(255u, out(30, 40, 1));
  ASSERT_EQ(255u, out(30, 40, 2));
  // #0000ff should map to roughly (171, 255, 255)
  ASSERT_EQ(171u, out(54, 79, 0));
  ASSERT_EQ(255u, out(54, 79, 1));
  ASSERT_EQ(255u, out(54, 79, 2));
  // #000000 should map to (???, ???, 0)
  ASSERT_EQ(0u, out(1, 0, 1));
  ASSERT_EQ(0u, out(1, 0, 2));
}

// Instantiate typed tests.
REGISTER_TYPED_TEST_CASE_P(LopperTypedTest,
                           SizeValidationTest,
                           ConstTest,
                           RowConstTest,
                           OffsetTest,
                           GradientTest,
                           ReindexTest,
                           MinimumMaximumTest,
                           ReindexOffsetTest,
                           CacheTest,
                           ThreeChannelTest,
                           RerunTest,
                           RGBToHSVTest,
                           ScopeTest);

template<InstructionSet S> struct LopperSettingType {
  static constexpr InstructionSet value = S;
};
#ifdef LOPPER_NO_SIMD
typedef testing::Types<LopperSettingType<LOPPER_TARGET>> LopperSettingTypes;
#else
typedef testing::Types<LopperSettingType<LOPPER_TARGET>, LopperSettingType<SCALAR>> LopperSettingTypes;
#endif
INSTANTIATE_TYPED_TEST_CASE_P(TypedTest, LopperTypedTest, LopperSettingTypes);
