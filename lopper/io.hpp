#pragma once

#include <exception>
#include <functional>
#include <stdint.h>
#include <string>
#include <type_traits>
#include <utility>

#include "context.hpp"
#include "expression.hpp"
#include "image.hpp"
#include "multiple.hpp"
#include "primitives.hpp"

namespace lopper {
namespace internal {

// We currently support arithmetic in int32 or float, so we'll cast incoming pixels to one of these types.
template<typename T> struct IOTypeTrait;
template<> struct IOTypeTrait<uint8_t> { typedef int32_t type; };
template<> struct IOTypeTrait<int32_t> { typedef int32_t type; };
template<> struct IOTypeTrait<float> { typedef float type; };

// MultipleIO<T, S> is the shorthand for the platform-specific vector type (pertaining to S)
// that represents primitives that are stored in memory as type T. e.g. MultipleIO<uint8_t, SSE> --> __mm128i
template<typename T, InstructionSet S> using MultipleIO =
  typename MultipleTrait<typename IOTypeTrait<T>::type, S>::vtype;

// MultipleIOTuple<T, C, S> is the shorthand for the tuple consisting of C copies of MultipleIO<T, S>.
template<size_t C> struct _MultipleIOTuple {
  template<typename T, InstructionSet S> using type =
    decltype(std::tuple_cat(std::declval<typename _MultipleIOTuple<C-1>::template type<T, S>>(),
                            std::declval<std::tuple<MultipleIO<T, S>>>()));
};
template<> struct _MultipleIOTuple<1> {
  template<typename T, InstructionSet S> using type = std::tuple<MultipleIO<T, S>>;
};
template<typename T, size_t C, InstructionSet S> using MultipleIOTuple =
  typename _MultipleIOTuple<C>::template type<T, S>;

// Forward declaration for pixel loading/storing utilities.
template<InstructionSet S> struct _PixelLoader {};
template<InstructionSet S> struct _PixelStorer {};

/*=============================== Machinery for reading pixels ===============================*/

template<> struct _PixelLoader<SCALAR> {
  // Loads a single Multiple from the address specified by the pointer.
  template<typename T> inline static MultipleIO<T, SCALAR> load(const T* ptr) {
    return (typename IOTypeTrait<T>::type)ptr[0];
  }
  // Returns the number of bytes accessed in each read operation.
  template<typename T, size_t C> static constexpr size_t bytesPerOp() { return sizeof(T) * C; }
  // Returns MultipleIO<T, C, S>, e.g. C copies of vectors, loaded from the given pointer
  template<typename T, size_t C> inline static SFINAE<(C>1), MultipleIOTuple<T, C, SCALAR>> load(const T* ptr) {
    return std::tuple_cat(std::make_tuple(MultipleIO<T, SCALAR>(ptr[0])), load<T, C-1>(ptr + 1));
  }
  template<typename T, size_t C> inline static SFINAE<(C==1), MultipleIOTuple<T, C, SCALAR>> load(const T* ptr) {
    return std::make_tuple(MultipleIO<T, SCALAR>(ptr[0]));
  }
};

#ifndef LOPPER_NO_SIMD
// TODO(jongmin): NEON could probably use vld3 instructions to avoid shuffling manually.
template<> struct _PixelLoader<LOPPER_TARGET> {
  template<typename T> inline static MultipleIO<T, LOPPER_TARGET> load(const T* ptr);
  template<typename T, size_t C> static constexpr size_t bytesPerOp();
  template<typename T, size_t C> inline static MultipleIOTuple<T, C, LOPPER_TARGET> load(const T* ptr);
};

// Specialization for int32_t
template<> inline constexpr size_t _PixelLoader<LOPPER_TARGET>::bytesPerOp<int32_t, 1>() { return 16; }
template<> inline constexpr size_t _PixelLoader<LOPPER_TARGET>::bytesPerOp<int32_t, 2>() { return 32; }
template<> inline constexpr size_t _PixelLoader<LOPPER_TARGET>::bytesPerOp<int32_t, 3>() { return 48; }
template<> inline Multiple<int32_t, LOPPER_TARGET> _PixelLoader<LOPPER_TARGET>::load<int32_t>(const int32_t* ptr) {
  return VLOAD<LOPPER_TARGET>(ptr);
}
template<> inline MultipleIOTuple<int32_t, 3, LOPPER_TARGET>
_PixelLoader<LOPPER_TARGET>::load<int32_t, 3 >(const int32_t* ptr) {
  // We want to go from [R0 G0 B0 R1] [G1 B1 R2 G2] [B2 R3 G3 B3] to [R0..R3] [G0..G3] [B0..B3]
  const typename InstructionSetTrait<LOPPER_TARGET>::INT32 _deshuffler00
  = VSET8x16<LOPPER_TARGET>(0, 1, 2, 3, 12, 13, 14, 15, 128, 128, 128, 128, 128, 128, 128, 128);
  const typename InstructionSetTrait<LOPPER_TARGET>::INT32 _deshuffler01
  = VSET8x16<LOPPER_TARGET>(128, 128, 128, 128, 128, 128, 128, 128, 8, 9, 10, 11, 128, 128, 128, 128);
  const typename InstructionSetTrait<LOPPER_TARGET>::INT32 _deshuffler02
  = VSET8x16<LOPPER_TARGET>(128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 4, 5, 6, 7);
  const typename InstructionSetTrait<LOPPER_TARGET>::INT32 _deshuffler10
  = VSET8x16<LOPPER_TARGET>(4, 5, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128);
  const typename InstructionSetTrait<LOPPER_TARGET>::INT32 _deshuffler11
  = VSET8x16<LOPPER_TARGET>(128, 128, 128, 128, 0, 1, 2, 3, 12, 13, 14, 15, 128, 128, 128, 128);
  const typename InstructionSetTrait<LOPPER_TARGET>::INT32 _deshuffler12
  = VSET8x16<LOPPER_TARGET>(128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 8, 9, 10, 11);
  const typename InstructionSetTrait<LOPPER_TARGET>::INT32 _deshuffler20
  = VSET8x16<LOPPER_TARGET>(8, 9, 10, 11, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128);
  const typename InstructionSetTrait<LOPPER_TARGET>::INT32 _deshuffler21
  = VSET8x16<LOPPER_TARGET>(128, 128, 128, 128, 4, 5, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128);
  const typename InstructionSetTrait<LOPPER_TARGET>::INT32 _deshuffler22
  = VSET8x16<LOPPER_TARGET>(128, 128, 128, 128, 128, 128, 128, 128, 0, 1, 2, 3, 12, 13, 14, 15);

  const auto in0 = VLOAD<LOPPER_TARGET>(ptr);
  const auto in1 = VLOAD<LOPPER_TARGET>(ptr + 4);
  const auto in2 = VLOAD<LOPPER_TARGET>(ptr + 8);
  const auto out0 = VBITWISE_OR(VBITWISE_OR(VSHUFFLE<LOPPER_TARGET>(in0, _deshuffler00),
                                            VSHUFFLE<LOPPER_TARGET>(in1, _deshuffler01)),
                                VSHUFFLE<LOPPER_TARGET>(in2, _deshuffler02));
  const auto out1 = VBITWISE_OR(VBITWISE_OR(VSHUFFLE<LOPPER_TARGET>(in0, _deshuffler10),
                                            VSHUFFLE<LOPPER_TARGET>(in1, _deshuffler11)),
                                VSHUFFLE<LOPPER_TARGET>(in2, _deshuffler12));
  const auto out2 = VBITWISE_OR(VBITWISE_OR(VSHUFFLE<LOPPER_TARGET>(in0, _deshuffler20),
                                            VSHUFFLE<LOPPER_TARGET>(in1, _deshuffler21)),
                                VSHUFFLE<LOPPER_TARGET>(in2, _deshuffler22));
  return std::make_tuple(out0, out1, out2);
}

// Specialization for uint8_t
template<> inline constexpr size_t _PixelLoader<LOPPER_TARGET>::bytesPerOp<uint8_t, 1>() { return 16; }
template<> inline constexpr size_t _PixelLoader<LOPPER_TARGET>::bytesPerOp<uint8_t, 2>() { return 16; }
template<> inline constexpr size_t _PixelLoader<LOPPER_TARGET>::bytesPerOp<uint8_t, 3>() { return 16; }
template<> inline Multiple<int32_t, LOPPER_TARGET> _PixelLoader<LOPPER_TARGET>::load<uint8_t>(const uint8_t* ptr) {
  return VEXPAND_BYTE<LOPPER_TARGET, 0>(VLOAD<LOPPER_TARGET>(ptr));
}
template<> inline MultipleIOTuple<uint8_t, 2, LOPPER_TARGET>
_PixelLoader<LOPPER_TARGET>::load<uint8_t, 2 >(const uint8_t* ptr) {
  const typename InstructionSetTrait<LOPPER_TARGET>::INT32 _deshuffler0
    = VSET8x16<LOPPER_TARGET>(0, 255, 255, 255, 2, 255, 255, 255, 4, 255, 255, 255, 6, 255, 255, 255);
  const typename InstructionSetTrait<LOPPER_TARGET>::INT32 _deshuffler1
    = VSET8x16<LOPPER_TARGET>(1, 255, 255, 255, 3, 255, 255, 255, 5, 255, 255, 255, 7, 255, 255, 255);
  const auto in = VLOAD<LOPPER_TARGET>(ptr);
  return std::make_tuple(VSHUFFLE<LOPPER_TARGET>(in, _deshuffler0),
                         VSHUFFLE<LOPPER_TARGET>(in, _deshuffler1));
}
template<> inline MultipleIOTuple<uint8_t, 3, LOPPER_TARGET>
_PixelLoader<LOPPER_TARGET>::load<uint8_t, 3 >(const uint8_t* ptr) {
  const typename InstructionSetTrait<LOPPER_TARGET>::INT32 _deshuffler0
    = VSET8x16<LOPPER_TARGET>(0, 255, 255, 255, 3, 255, 255, 255, 6, 255, 255, 255, 9, 255, 255, 255);
  const typename InstructionSetTrait<LOPPER_TARGET>::INT32 _deshuffler1
    = VSET8x16<LOPPER_TARGET>(1, 255, 255, 255, 4, 255, 255, 255, 7, 255, 255, 255, 10, 255, 255, 255);
  const typename InstructionSetTrait<LOPPER_TARGET>::INT32 _deshuffler2
    = VSET8x16<LOPPER_TARGET>(2, 255, 255, 255, 5, 255, 255, 255, 8, 255, 255, 255, 11, 255, 255, 255);
  const auto in = VLOAD<LOPPER_TARGET>(ptr);
  return std::make_tuple(VSHUFFLE<LOPPER_TARGET>(in, _deshuffler0),
                         VSHUFFLE<LOPPER_TARGET>(in, _deshuffler1),
                         VSHUFFLE<LOPPER_TARGET>(in, _deshuffler2));
}

// Specialization for float
template<> inline constexpr size_t _PixelLoader<LOPPER_TARGET>::bytesPerOp<float, 1>() { return 16; }
template<> inline constexpr size_t _PixelLoader<LOPPER_TARGET>::bytesPerOp<float, 2>() { return 32; }
template<> inline constexpr size_t _PixelLoader<LOPPER_TARGET>::bytesPerOp<float, 3>() { return 48; }
template<> inline Multiple<float, LOPPER_TARGET> _PixelLoader<LOPPER_TARGET>::load<float>(const float* ptr) {
  return VLOAD<LOPPER_TARGET>(ptr);
}
template<> inline MultipleIOTuple<float, 3, LOPPER_TARGET>
_PixelLoader<LOPPER_TARGET>::load<float, 3>(const float* ptr) {
  // We want to go from [R0 G0 B0 R1] [G1 B1 R2 G2] [B2 R3 G3 B3] to [R0..R3] [G0..G3] [B0..B3]
  const typename InstructionSetTrait<LOPPER_TARGET>::INT32 _deshuffler00
  = VSET8x16<LOPPER_TARGET>(0, 1, 2, 3, 12, 13, 14, 15, 128, 128, 128, 128, 128, 128, 128, 128);
  const typename InstructionSetTrait<LOPPER_TARGET>::INT32 _deshuffler01
  = VSET8x16<LOPPER_TARGET>(128, 128, 128, 128, 128, 128, 128, 128, 8, 9, 10, 11, 128, 128, 128, 128);
  const typename InstructionSetTrait<LOPPER_TARGET>::INT32 _deshuffler02
  = VSET8x16<LOPPER_TARGET>(128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 4, 5, 6, 7);
  const typename InstructionSetTrait<LOPPER_TARGET>::INT32 _deshuffler10
  = VSET8x16<LOPPER_TARGET>(4, 5, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128);
  const typename InstructionSetTrait<LOPPER_TARGET>::INT32 _deshuffler11
  = VSET8x16<LOPPER_TARGET>(128, 128, 128, 128, 0, 1, 2, 3, 12, 13, 14, 15, 128, 128, 128, 128);
  const typename InstructionSetTrait<LOPPER_TARGET>::INT32 _deshuffler12
  = VSET8x16<LOPPER_TARGET>(128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 8, 9, 10, 11);
  const typename InstructionSetTrait<LOPPER_TARGET>::INT32 _deshuffler20
  = VSET8x16<LOPPER_TARGET>(8, 9, 10, 11, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128);
  const typename InstructionSetTrait<LOPPER_TARGET>::INT32 _deshuffler21
  = VSET8x16<LOPPER_TARGET>(128, 128, 128, 128, 4, 5, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128);
  const typename InstructionSetTrait<LOPPER_TARGET>::INT32 _deshuffler22
  = VSET8x16<LOPPER_TARGET>(128, 128, 128, 128, 128, 128, 128, 128, 0, 1, 2, 3, 12, 13, 14, 15);
  const auto in0 = VLOAD<LOPPER_TARGET>((int32_t*)ptr);
  const auto in1 = VLOAD<LOPPER_TARGET>((int32_t*)ptr + 4);
  const auto in2 = VLOAD<LOPPER_TARGET>((int32_t*)ptr + 8);
  const auto out0 = VBITWISE_OR(VBITWISE_OR(VSHUFFLE<LOPPER_TARGET>(in0, _deshuffler00),
                                            VSHUFFLE<LOPPER_TARGET>(in1, _deshuffler01)),
                                VSHUFFLE<LOPPER_TARGET>(in2, _deshuffler02));
  const auto out1 = VBITWISE_OR(VBITWISE_OR(VSHUFFLE<LOPPER_TARGET>(in0, _deshuffler10),
                                            VSHUFFLE<LOPPER_TARGET>(in1, _deshuffler11)),
                                VSHUFFLE<LOPPER_TARGET>(in2, _deshuffler12));
  const auto out2 = VBITWISE_OR(VBITWISE_OR(VSHUFFLE<LOPPER_TARGET>(in0, _deshuffler20),
                                            VSHUFFLE<LOPPER_TARGET>(in1, _deshuffler21)),
                                VSHUFFLE<LOPPER_TARGET>(in2, _deshuffler22));
  return std::make_tuple(VCAST_FLOAT<LOPPER_TARGET>(out0), VCAST_FLOAT<LOPPER_TARGET>(out1), VCAST_FLOAT<LOPPER_TARGET>(out2));
}
#endif

/*=============================== Machinery for writing pixels ===============================*/

template<> struct _PixelStorer<SCALAR> {
  // Returns the number of bytes accessed in each write operation.
  template<typename T, size_t C> constexpr static size_t bytesPerOp() {
    return sizeof(T) * C;
  }
  template<typename T> static void store(T* ptr, const MultipleIO<T, SCALAR>& val) {
    *ptr = (T)val;
  }
  template<typename T> static void store3(T* ptr,
                                          const MultipleIO<T, SCALAR>& val0,
                                          const MultipleIO<T, SCALAR>& val1,
                                          const MultipleIO<T, SCALAR>& val2) {
    ptr[0] = (T)val0;
    ptr[1] = (T)val1;
    ptr[2] = (T)val2;
  }
};

#ifndef LOPPER_NO_SIMD
// XXX: Use vld3 for NEON.
template<> struct _PixelStorer<LOPPER_TARGET> {
  template<typename T, size_t C> constexpr static size_t bytesPerOp() {
    // At the moment, all supported formats and platforms write over 16 bytes.
    return 16;
  }
  template<typename T> static void store(T* ptr, const MultipleIO<T, LOPPER_TARGET>& val) {
    VSTORE(ptr, val);
  }
  template<typename T> static void store3(T* ptr,
                                          const MultipleIO<T, LOPPER_TARGET>& val0,
                                          const MultipleIO<T, LOPPER_TARGET>& val1,
                                          const MultipleIO<T, LOPPER_TARGET>& val2);
};

template<> inline void _PixelStorer<LOPPER_TARGET>::store<uint8_t>(uint8_t* ptr,
                                                                   const Multiple<int32_t, LOPPER_TARGET>& val) {
  const typename InstructionSetTrait<LOPPER_TARGET>::INT32 _deshuffler0 =
    VSET8x16<LOPPER_TARGET>(0, 4, 8, 12, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128);
  VSTORE(ptr, VSHUFFLE<LOPPER_TARGET>(val, _deshuffler0));
}

template<> inline void _PixelStorer<LOPPER_TARGET>::store3<uint8_t>(uint8_t* ptr,
                                                                    const Multiple<int32_t, LOPPER_TARGET>& val0,
                                                                    const Multiple<int32_t, LOPPER_TARGET>& val1,
                                                                    const Multiple<int32_t, LOPPER_TARGET>& val2) {
  const typename InstructionSetTrait<LOPPER_TARGET>::INT32 _deshuffler0 =
    VSET8x16<LOPPER_TARGET>(0, 128, 128, 4, 128, 128, 8, 128, 128, 12, 128, 128, 128, 128, 128, 128);
  const typename InstructionSetTrait<LOPPER_TARGET>::INT32 _deshuffler1 =
    VSET8x16<LOPPER_TARGET>(128, 0, 128, 128, 4, 128, 128, 8, 128, 128, 12, 128, 128, 128, 128, 128);
  const typename InstructionSetTrait<LOPPER_TARGET>::INT32 _deshuffler2 =
    VSET8x16<LOPPER_TARGET>(128, 128, 0, 128, 128, 4, 128, 128, 8, 128, 128, 12, 128, 128, 128, 128);
  VSTORE(ptr, VBITWISE_OR(VBITWISE_OR(VSHUFFLE<LOPPER_TARGET>(val0, _deshuffler0),
                                      VSHUFFLE<LOPPER_TARGET>(val1, _deshuffler1)),
                          VSHUFFLE<LOPPER_TARGET>(val2, _deshuffler2)));
}

template<> inline void _PixelStorer<LOPPER_TARGET>::store3<int32_t>(int32_t* ptr,
                                                                    const Multiple<int32_t, LOPPER_TARGET>& val0,
                                                                    const Multiple<int32_t, LOPPER_TARGET>& val1,
                                                                    const Multiple<int32_t, LOPPER_TARGET>& val2) {
  const typename InstructionSetTrait<LOPPER_TARGET>::INT32 _deshuffler0 =
    VSET8x16<LOPPER_TARGET>(0, 1, 2, 3, 128, 128, 128, 128, 128, 128, 128, 128, 4, 5, 6, 7);
  const typename InstructionSetTrait<LOPPER_TARGET>::INT32 _deshuffler1 =
    VSET8x16<LOPPER_TARGET>(128, 128, 128, 128, 0, 1, 2, 3, 128, 128, 128, 128, 128, 128, 128, 128);
  const typename InstructionSetTrait<LOPPER_TARGET>::INT32 _deshuffler2 =
    VSET8x16<LOPPER_TARGET>(128, 128, 128, 128, 128, 128, 128, 128, 0, 1, 2, 3, 128, 128, 128, 128);
  // Write out the first 16 bytes.
  auto out0 = VBITWISE_OR(VBITWISE_OR(VSHUFFLE<LOPPER_TARGET>(val0, _deshuffler0),
                                      VSHUFFLE<LOPPER_TARGET>(val1, _deshuffler1)),
                          VSHUFFLE<LOPPER_TARGET>(val2, _deshuffler2));
  VSTORE(ptr, out0);
  // Write out the second 16 bytes.
  // NOTE: 0x01010101 is a hack because vectors are meant to be 32-bit int, but we really want 8-bit addition.
#ifndef VSET8
#define VSET8(x) VSET<LOPPER_TARGET>(0x01010101 * (x))
#endif
  auto out1 = VBITWISE_OR(VBITWISE_OR(VSHUFFLE<LOPPER_TARGET>(val1, VADD(VSET8(4), _deshuffler0)),
                                      VSHUFFLE<LOPPER_TARGET>(val2, VADD(VSET8(4), _deshuffler1))),
                          VSHUFFLE<LOPPER_TARGET>(val0, VADD(VSET8(8), _deshuffler2)));
  VSTORE(ptr + 4, out1);
  // Write out the third 16 bytes.
  auto out2 = VBITWISE_OR(VBITWISE_OR(VSHUFFLE<LOPPER_TARGET>(val2, VADD(VSET8(8), _deshuffler0)),
                                      VSHUFFLE<LOPPER_TARGET>(val0, VADD(VSET8(12), _deshuffler1))),
                          VSHUFFLE<LOPPER_TARGET>(val1, VADD(VSET8(12), _deshuffler2)));
  VSTORE(ptr + 8, out2);
}
#endif

/*=============================== Expressions for writing memory ===============================*/

template<typename T, typename ... E> struct _ExprSave : public NaryExpr<E...> {
  _ExprSave(const std::shared_ptr<_Image<T>>& image, const E&... e)
    : NaryExpr<E...>(e...), _image(image) {
    if (image->getChannelCount() != sizeof...(E)) { throw LopperException("Invalid number of channels"); }
  }
  virtual ~_ExprSave() {}
  virtual int getWidth() const { return _image->getWidth(); }
  virtual int getHeight() const { return _image->getHeight(); }
  virtual size_t getSIMDClearance() const = 0;
protected:
  std::shared_ptr<_Image<T>> _image;
};

template<typename T, typename E> struct _ExprSave1 : public _ExprSave<T, E> {
  _ExprSave1(const std::shared_ptr<_Image<T>>& image, const E& e) : _ExprSave<T, E>(image, e) {}

  virtual size_t getSIMDClearance() const {
    const size_t bytes_written = _PixelStorer<LOPPER_TARGET>::template bytesPerOp<T, 1>();
    const size_t bytes_per_pixel = sizeof(T);
    return (bytes_written + (bytes_per_pixel - 1)) / bytes_per_pixel;
  }

  void prepareRow(const int y) const {
    _ptr = this->_image->getRowPointer(y);
    this->UnaryExpr<typename E::type, E>::prepareRow(y);
  }

  template<InstructionSet S, size_t U, typename ... Cxt> inline
  MultipleIO<T, S> eval(const int x, const Cxt& ... args) const {
    const auto v = this->_e0.template eval<S, U>(x, args...);
    _PixelStorer<S>::template store<T>(_ptr + x, v);
    return v;
  }
private:
  mutable T* _ptr;
};

template<typename T, typename E0, typename E1, typename E2> struct _ExprSave3 : public _ExprSave<T, E0, E1, E2> {
  _ExprSave3(std::shared_ptr<_Image<T>>& image, const E0& e0, const E1& e1, const E2& e2)
    : _ExprSave<T, E0, E1, E2>(image, e0, e1, e2) {}

  virtual size_t getSIMDClearance() const {
    const size_t bytes_written = _PixelStorer<LOPPER_TARGET>::template bytesPerOp<T, 3>();
    const size_t bytes_per_pixel = sizeof(T) * 3;
    return (bytes_written + (bytes_per_pixel - 1)) / bytes_per_pixel;
  }

  void prepareRow(const int y) const {
    _ptr = this->_image->getRowPointer(y);
    this->TernaryExpr<typename E0::type, E0, E1, E2>::prepareRow(y);
  }

  template<InstructionSet S, size_t U, typename ... Cxt> inline
  Multiple<typename E0::type, S> eval(const int x, const Cxt& ... args) const {
    const auto v0 = this->_e0.template eval<S, U>(x, args...);
    const auto v1 = this->_e1.template eval<S, U>(x, args...);
    const auto v2 = this->_e2.template eval<S, U>(x, args...);
    _PixelStorer<S>::template store3<T>(_ptr + x * 3, v0, v1, v2);
    return v0;
  }
private:
  mutable T* _ptr;
};

/*=============================== Expressions for reading memory ===============================*/

// Machinery for reading or writing from image. By default, _ExprImage instance behaves as a reader,
// but one can instantiate writers via calling assignment operators, as long as the image hasn't been reindexed.
// DefaultX, DefaultY indicate whether the image has nontrivial reindexing horizontally or vertically, respectively.

template<typename T, bool DefaultX=true, bool DefaultY=true>
struct _ExprImage1 : public NullaryExpr<typename IOTypeTrait<T>::type> {
  _ExprImage1(std::shared_ptr<_Image<T>> image) : _ExprImage1(image, 0, [](int y) { return y; }) {}

  virtual int getWidth() const { return _image->getWidth(); }
  virtual int getHeight() const { return DefaultY ? _image->getHeight() : -1; }
  virtual size_t getSIMDClearance() const {
    const size_t bytes_read = _PixelLoader<LOPPER_TARGET>::template bytesPerOp<T, 1>();
    const size_t bytes_per_pixel = sizeof(T);
    return (bytes_read + (bytes_per_pixel - 1)) / bytes_per_pixel;
  }

  void prepareRow(const int y) const {
    const int new_y = std::min(std::max<int>(_ymap(y), 0), this->_image->getHeight() - 1);
    _ptr = this->_image->getRowPointer(new_y);
    _width = this->_image->getWidth(); // cache this to avoid calling the virtual function in eval(...).
  }

  template<InstructionSet S, size_t U, typename ... Cxt> inline
  MultipleIO<T, S> eval(const int x, const Cxt& ...) const {
    if (DefaultX) {
      return _PixelLoader<S>::template load<T>(_ptr + x);
    } else if (S != SCALAR) {
      // _execute(...) guarantees that the SIMD portion of the loop will only access valid areas even with translation.
      return _PixelLoader<S>::template load<T>(_ptr + x + _dx);
    } else {
      const int offset = std::min(std::max(0, (int)x + _dx), _width - 1);
      return _PixelLoader<S>::template load<T>(_ptr + offset);
    }
  }

  // Returns the expression representing a read-only image, with the given offset to be used during indexing.
  _ExprImage1<T, false, false> offset(int dx, int dy) const {
    auto ymap_local = this->_ymap;
    return _ExprImage1<T, false, false>(this->_image, _dx + dx,
                                        [ymap_local, dy](int y) { return ymap_local(y) + dy; });
  }

  // Returns the expression representing a read-only image, with the given row remapping.
  // That is, it corresponds logically to an image whose y-th row is the func(y)-th row of the source.
  _ExprImage1<T, DefaultX, false> reindex(const std::function<int(int)>& func) const {
    auto ymap_local = this->_ymap;
    return _ExprImage1<T, DefaultX, false>(this->_image, _dx,
                                           [ymap_local, func](int y) { return func(ymap_local(y)); });
  }

  virtual int getHorizontalOffset() const { return _dx; }

  // NOTE: It would be nice to use C++14's decltype(auto) here, but in the meantime we define operators
  // to enable Expr(image) += ... and other syntactic sugars.
  template<typename E> auto operator+=(const E& e) ->
    _ExprSave1<T, decltype(std::declval<_ExprImage1<T, DefaultX, DefaultY>>() + std::declval<E>())> {
    _check_writability();
    auto final_exp = *this + e;
    return _ExprSave1<T, decltype(final_exp)>(this->_image, final_exp);
  }

  template<typename E> auto operator-=(const E& e) ->
    _ExprSave1<T, decltype(std::declval<_ExprImage1<T, DefaultX, DefaultY>>() - std::declval<E>())> {
    _check_writability();
    auto final_exp = *this - e;
    return _ExprSave1<T, decltype(final_exp)>(this->_image, final_exp);
  }

  template<typename E> auto operator*=(const E& e) ->
    _ExprSave1<T, decltype(std::declval<_ExprImage1<T, DefaultX, DefaultY>>() * std::declval<E>())> {
    _check_writability();
    auto final_exp = *this * e;
    return _ExprSave1<T, decltype(final_exp)>(this->_image, final_exp);
  }

  template<typename E> auto operator=(const E& e) -> _ExprSave1<T, E> {
    _check_writability();
    return _ExprSave1<T, E>(this->_image, e);
  }

  _ExprSave1<T, _ExprImage1<T, DefaultX, DefaultY>> operator=(const _ExprImage1<T, DefaultX, DefaultY>& e) {
    _check_writability();
    return _ExprSave1<T, _ExprImage1<T, DefaultX, DefaultY>>(this->_image, e);
  }
private:
  _ExprImage1(const std::shared_ptr<_Image<T>>& image, int dx, const std::function<int(int)>& ymap) :
    _image(image), _dx(dx), _ymap(ymap) {
    if (_image->getChannelCount() != 1) { throw LopperException("Invalid number of channels"); }
  }
  void _check_writability() const {
    static_assert(DefaultX && DefaultY, "Cannot write into a translated image expression");
  }
  std::shared_ptr<_Image<T>> _image;
  int _dx;
  mutable int _width;
  std::function<int(int)> _ymap;
  mutable T* _ptr;
  template<typename, bool, bool> friend struct _ExprImage1;
};

template<typename T> struct _ExprImage3 : public NullaryExpr<typename IOTypeTrait<T>::type> {
  _ExprImage3(std::shared_ptr<_Image<T>> image) : _image(image) {
    if (_image->getChannelCount() != 3) { throw LopperException("Invalid number of channels"); }
  }

  virtual int getWidth() const { return _image->getWidth(); }
  virtual int getHeight() const { return _image->getHeight(); }
  virtual size_t getSIMDClearance() const {
    const size_t bytes_read = _PixelLoader<LOPPER_TARGET>::template bytesPerOp<T, 3>();
    const size_t bytes_per_pixel = sizeof(T);
    return (bytes_read + (bytes_per_pixel - 1)) / bytes_per_pixel;
  }

  void prepareRow(const int y) const { _ptr = _image->getRowPointer(y); }

  template<InstructionSet S, size_t U, typename ... Cxt>
  MultipleIOTuple<T, 3, S> inline eval(const int x, const Cxt& ... ) const {
    return _PixelLoader<S>::template load<T, 3>(_ptr + x * 3);
  }

  template<typename E0, typename E1, typename E2> auto operator=(const std::tuple<E0, E1, E2>& t) ->
    _ExprSave3<T, E0, E1, E2> {
      return _ExprSave3<T, E0, E1, E2>(_image, std::get<0>(t), std::get<1>(t), std::get<2>(t));
    }

private:
  mutable T* _ptr;
  std::shared_ptr<_Image<T>> _image;
};

} // end namespace internal

using namespace lopper::internal;
// Caller-friendly wrappers to generate expressions from images, assuming that images will stay on the stack.
template<size_t C, typename T, typename F = SFINAE<C == 1, _ExprImage1<T>>>
  _ExprImage1<T> Expr(::lopper::_Image<T>& image) {
  return _ExprImage1<T>(std::shared_ptr<::lopper::_Image<T>>(std::shared_ptr<::lopper::_Image<T>>(),
                                                             dynamic_cast<::lopper::_Image<T>*>(&image)));
}
template<size_t C, typename T, typename F = SFINAE<C == 3, _ExprImage3<T>>>
  _ExprImage3<T> Expr(::lopper::_Image<T>& image) {
  return _ExprImage3<T>(std::shared_ptr<::lopper::_Image<T>>(std::shared_ptr<::lopper::_Image<T>>(),
                                                             dynamic_cast<::lopper::_Image<T>*>(&image)));
}

} // end namespace lopper
