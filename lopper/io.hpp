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
#include "util.hpp"

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

// _DataLoader is the helper class for loading primitives of size D bytes in C interleaved channels.
// It's intentionally abstracted out so that _PixelLoader can apply logic like clamping if necessary.
template<size_t D, size_t C> struct _DataLoader /* <LOPPER_TARGET> */ {
  inline static constexpr size_t bytesPerOp();
};
template<> struct _PixelLoader<LOPPER_TARGET> {
  template<typename T, size_t C> static constexpr size_t bytesPerOp() {
    return _DataLoader<sizeof(T), C>::bytesPerOp();
  }
  template<typename T> inline static MultipleIO<T, LOPPER_TARGET> load(const T* ptr) {
    return _DataLoader<sizeof(T), 1>::template load<T>(ptr);
  }
  template<typename T, size_t C> inline static MultipleIOTuple<T, C, LOPPER_TARGET> load(const T* ptr) {
    return _DataLoader<sizeof(T), C>::template load<T>(ptr);
  }
};

template<> struct _DataLoader<1, 1> {
  inline static constexpr size_t bytesPerOp() { return LOPPER_BITWIDTH >> 3; }
  template<typename T> inline static MultipleIO<T, LOPPER_TARGET> load(const T* ptr) {
    // T is expected to be uint8_t
    static_assert(std::is_same<T, uint8_t>::value, "Expect uint8_t");
    return VEXPAND_QTR<LOPPER_TARGET, 0>(VLOAD<LOPPER_TARGET>(ptr));
  }
};

template<> struct _DataLoader<1, 2> {
  inline static constexpr size_t bytesPerOp() { return LOPPER_BITWIDTH >> 3; }
  template<typename T> inline static MultipleIOTuple<T, 2, LOPPER_TARGET> load(const T* ptr) {
    // T is expected to be uint8_t
    static_assert(std::is_same<T, uint8_t>::value, "Expect uint8_t");
    const auto in = VLOAD<LOPPER_TARGET>(ptr);
    const auto deshuffler = VSET4x8<LOPPER_TARGET>(0xff01ff00,
                                                   0xff03ff02,
                                                   0xff05ff04,
                                                   0xff07ff06,
                                                   0xff09ff08,
                                                   0xff0bff0a,
                                                   0xff0dff0c,
                                                   0xff0fff0e);
    const auto tmp = VSHUFFLE8<LOPPER_TARGET>(in, deshuffler);
    return std::make_tuple(VBITWISE_AND(tmp, VSET<LOPPER_TARGET>(255)), VSHIFTR<16>(tmp));
  }
};

template<> struct _DataLoader<1, 3> {
  inline static constexpr size_t bytesPerOp() { return LOPPER_BITWIDTH >> 3; }
  template<typename T> inline static MultipleIOTuple<T, 3, LOPPER_TARGET> load(const T* ptr) {
    // T is expected to be uint8_t
    static_assert(std::is_same<T, uint8_t>::value, "Expect uint8_t");
    constexpr size_t num_lanes = InstructionSetTrait<LOPPER_TARGET>::num_lanes;
    const auto in = VLOAD<LOPPER_TARGET>(ptr);
    if (num_lanes == 4u) {
      const auto deshuffler0 = VSET4x4<LOPPER_TARGET>(0xffffff00, 0xffffff03, 0xffffff06, 0xffffff09);
      const auto deshuffler1 = VSET4x4<LOPPER_TARGET>(0xffffff01, 0xffffff04, 0xffffff07, 0xffffff0a);
      const auto deshuffler2 = VSET4x4<LOPPER_TARGET>(0xffffff02, 0xffffff05, 0xffffff08, 0xffffff0b);
      const auto in = VLOAD<LOPPER_TARGET>(ptr);
      return std::make_tuple(VSHUFFLE8<LOPPER_TARGET>(in, deshuffler0),
                             VSHUFFLE8<LOPPER_TARGET>(in, deshuffler1),
                             VSHUFFLE8<LOPPER_TARGET>(in, deshuffler2));
    } else if (num_lanes == 8u) {
      // [R0 G0 B0 R1 G1 B1 ... R7 G7 B7 _ ... _]
      const auto deshuffler = VSET4x8<LOPPER_TARGET>(0xff020100,
                                                     0xff050403,
                                                     0xff080706,
                                                     0xff0b0a09,
                                                     0xff0e0d0c,
                                                     0xff11100f,
                                                     0xff141312,
                                                     0xff171615);
      const auto tmp = VSHUFFLE8<LOPPER_TARGET>(in, deshuffler);
      return std::make_tuple(VBITWISE_AND(tmp, VSET<LOPPER_TARGET>(255)),
                             VBITWISE_AND(VSHIFTR<8>(tmp), VSET<LOPPER_TARGET>(255)),
                             VSHIFTR<16>(tmp));
    }
  }
};

template<> struct _DataLoader<1, 4> {
  inline static constexpr size_t bytesPerOp() { return LOPPER_BITWIDTH >> 3; }
  template<typename T> inline static MultipleIOTuple<T, 4, LOPPER_TARGET> load(const T* ptr) {
    // T is expected to be uint8_t
    static_assert(std::is_same<T, uint8_t>::value, "Expect uint8_t");
    const auto in = VLOAD<LOPPER_TARGET>(ptr);
    const auto deshuffler = VSET4x8<LOPPER_TARGET>(0x03020100,
                                                   0x07060504,
                                                   0x0b0a0908,
                                                   0x0f0e0d0c,
                                                   0x13121110,
                                                   0x17161514,
                                                   0x1b1a1918,
                                                   0x1f1e1d1c);
    const auto tmp = VSHUFFLE8<LOPPER_TARGET>(in, deshuffler);
    return std::make_tuple(VBITWISE_AND(tmp, VSET<LOPPER_TARGET>(255)),
                           VBITWISE_AND(VSHIFTR<8>(tmp), VSET<LOPPER_TARGET>(255)),
                           VBITWISE_AND(VSHIFTR<16>(tmp), VSET<LOPPER_TARGET>(255)),
                           VBITWISE_AND(VSHIFTR<24>(tmp), VSET<LOPPER_TARGET>(255)));
  }
};

template<> struct _DataLoader<4, 1> {
  inline static constexpr size_t bytesPerOp() { return LOPPER_BITWIDTH >> 3; }
  template<typename T> inline static MultipleIO<T, LOPPER_TARGET> load(const T* ptr) {
    return VLOAD<LOPPER_TARGET>(ptr);
  }
};

template<> struct _DataLoader<4, 3> {
  inline static constexpr size_t bytesPerOp() { return (LOPPER_BITWIDTH >> 3) * 3; }
  template<typename T> inline static MultipleIOTuple<T, 3, LOPPER_TARGET> load(const T* ptr) {
    return VLOAD3<LOPPER_TARGET>(ptr);
  }
};

template<> struct _DataLoader<4, 4> {
  inline static constexpr size_t bytesPerOp() { return LOPPER_BITWIDTH >> 1; }
  template<typename T> inline static MultipleIOTuple<T, 4, LOPPER_TARGET> load(const T* ptr) {
    return VLOAD4<LOPPER_TARGET>(ptr);
  }
};
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
  template<typename T> static void store(T* ptr,
                                         const MultipleIO<T, SCALAR>& val0,
                                         const MultipleIO<T, SCALAR>& val1,
                                         const MultipleIO<T, SCALAR>& val2) {
    ptr[0] = (T)val0;
    ptr[1] = (T)val1;
    ptr[2] = (T)val2;
  }
  template<typename T> static void store(T* ptr,
                                         const MultipleIO<T, SCALAR>& val0,
                                         const MultipleIO<T, SCALAR>& val1,
                                         const MultipleIO<T, SCALAR>& val2,
                                         const MultipleIO<T, SCALAR>& val3) {
    ptr[0] = (T)val0;
    ptr[1] = (T)val1;
    ptr[2] = (T)val2;
    ptr[3] = (T)val3;
  }
};

#ifndef LOPPER_NO_SIMD

template<size_t D, size_t C> struct _DataStorer /* <LOPPER_TARGET> */ {
  inline static constexpr size_t bytesPerOp();
};
template<> struct _PixelStorer<LOPPER_TARGET> {
  template<typename T, size_t C> static constexpr size_t bytesPerOp() {
    return _DataStorer<sizeof(T), C>::bytesPerOp();
  }
  template<typename T> static void store(T* ptr, const MultipleIO<T, LOPPER_TARGET>& val) {
    _DataStorer<sizeof(T), 1>::store(ptr, val);
  }
  template<typename T> static void store(T* ptr,
                                         const MultipleIO<T, LOPPER_TARGET>& val0,
                                         const MultipleIO<T, LOPPER_TARGET>& val1,
                                         const MultipleIO<T, LOPPER_TARGET>& val2) {
    _DataStorer<sizeof(T), 3>::store(ptr, val0, val1, val2);
  }
  template<typename T> static void store(T* ptr,
                                         const MultipleIO<T, LOPPER_TARGET>& val0,
                                         const MultipleIO<T, LOPPER_TARGET>& val1,
                                         const MultipleIO<T, LOPPER_TARGET>& val2,
                                         const MultipleIO<T, LOPPER_TARGET>& val3) {
    _DataStorer<sizeof(T), 4>::store(ptr, val0, val1, val2, val3);
  }
};

template<> struct _DataStorer<1, 1> {
  inline static constexpr size_t bytesPerOp() { return LOPPER_BITWIDTH >> 3; }
  template<typename T> inline static void store(T* ptr, const MultipleIO<T, LOPPER_TARGET>& val) {
    static_assert(std::is_same<T, uint8_t>::value, "Expect uint8_t");
    constexpr size_t num_lanes = InstructionSetTrait<LOPPER_TARGET>::num_lanes;
    static_assert(num_lanes == 4u || num_lanes == 8u, "Expect 4 or 8 lanes");
    const auto deshuffler =
      VSET8x16<LOPPER_TARGET>(0, 4, 8, 12,
                              num_lanes == 8u ? 16 : 128,
                              num_lanes == 8u ? 20 : 128,
                              num_lanes == 8u ? 24 : 128,
                              num_lanes == 8u ? 28 : 128,
                              128, 128, 128, 128, 128, 128, 128, 128);
    VSTORE(ptr, VSHUFFLE8<LOPPER_TARGET>(val, deshuffler));
  }
};

template<> struct _DataStorer<1, 3> {
  inline static constexpr size_t bytesPerOp() { return LOPPER_BITWIDTH >> 3; }
  template<typename T> inline static void store(T* ptr,
                                                const MultipleIO<T, LOPPER_TARGET>& val0,
                                                const MultipleIO<T, LOPPER_TARGET>& val1,
                                                const MultipleIO<T, LOPPER_TARGET>& val2) {
    static_assert(std::is_same<T, uint8_t>::value, "Expect uint8_t");
    constexpr size_t num_lanes = InstructionSetTrait<LOPPER_TARGET>::num_lanes;
    static_assert(num_lanes == 4u || num_lanes == 8u, "Expect 4 or 8 lanes");
    if (num_lanes == 4u) {
      const auto deshuffler0 = VSET4x4<LOPPER_TARGET>(0x04ffff00, 0xff08ffff, 0xffff0cff, 0xffffffff);
      const auto deshuffler1 = VSET4x4<LOPPER_TARGET>(0xffff00ff, 0x08ffff04, 0xff0cffff, 0xffffffff);
      const auto deshuffler2 = VSET4x4<LOPPER_TARGET>(0xff00ffff, 0xffff04ff, 0x0cffff08, 0xffffffff);
      VSTORE(ptr, VBITWISE_OR(VBITWISE_OR(VSHUFFLE<LOPPER_TARGET>(val0, deshuffler0),
                                          VSHUFFLE<LOPPER_TARGET>(val1, deshuffler1)),
                              VSHUFFLE<LOPPER_TARGET>(val2, deshuffler2)));
    } else if (num_lanes == 8u) {
      const auto deshuffler0 = VSET4x8<LOPPER_TARGET>(0x04ffff00, 0xff08ffff, 0xffff0cff, 0x14ffff10,
                                                      0xff18ffff, 0xffff1cff, -1, -1);
      const auto deshuffler1 = VSET4x8<LOPPER_TARGET>(0xffff00ff, 0x08ffff04, 0xff0cffff, 0xffff10ff,
                                                      0x18ffff14, 0xff1cffff, -1, -1);
      const auto deshuffler2 = VSET4x8<LOPPER_TARGET>(0xff00ffff, 0xffff04ff, 0x0cffff08, 0xff10ffff,
                                                      0xffff14ff, 0x1cffff18, -1, -1);
      VSTORE(ptr, VBITWISE_OR(VBITWISE_OR(VSHUFFLE8<LOPPER_TARGET>(val0, deshuffler0),
                                          VSHUFFLE8<LOPPER_TARGET>(val1, deshuffler1)),
                              VSHUFFLE8<LOPPER_TARGET>(val2, deshuffler2)));
    }
  }
};

template<> struct _DataStorer<1, 4> {
  inline static constexpr size_t bytesPerOp() { return LOPPER_BITWIDTH >> 3; }
  template<typename T> inline static void store(T* ptr,
                                                const MultipleIO<T, LOPPER_TARGET>& val0,
                                                const MultipleIO<T, LOPPER_TARGET>& val1,
                                                const MultipleIO<T, LOPPER_TARGET>& val2,
                                                const MultipleIO<T, LOPPER_TARGET>& val3) {
    static_assert(std::is_same<T, uint8_t>::value, "Expect uint8_t");
    const auto mask = VSET<LOPPER_TARGET>(255);
    const auto val0_masked = VBITWISE_AND(val0, mask);
    const auto val1_masked = VSHIFTL<8>(VBITWISE_AND(val1, mask));
    const auto val2_masked = VSHIFTL<16>(VBITWISE_AND(val2, mask));
    const auto val3_masked = VSHIFTL<24>(VBITWISE_AND(val3, mask));
    VSTORE(ptr, VBITWISE_OR(VBITWISE_OR(val0_masked, val1_masked), VBITWISE_OR(val2_masked, val3_masked)));
  }
};

template<> struct _DataStorer<4, 1> {
  inline static constexpr size_t bytesPerOp() { return LOPPER_BITWIDTH >> 3; }
  template<typename T> inline static void store(T* ptr, const MultipleIO<T, LOPPER_TARGET>& val) {
    VSTORE(ptr, val);
  }
};

template<> struct _DataStorer<4, 3> {
  inline static constexpr size_t bytesPerOp() { return (LOPPER_BITWIDTH >> 3) * 3; }
  template<typename T> inline static void store(T* ptr,
                                                const MultipleIO<T, LOPPER_TARGET>& val0,
                                                const MultipleIO<T, LOPPER_TARGET>& val1,
                                                const MultipleIO<T, LOPPER_TARGET>& val2) {
    VSTORE3(ptr, val0, val1, val2);
  }
};


template<> struct _DataStorer<4, 4> {
  inline static constexpr size_t bytesPerOp() { return (LOPPER_BITWIDTH >> 3) * 4; }
  template<typename T> inline static void store(T* ptr,
                                                const MultipleIO<T, LOPPER_TARGET>& val0,
                                                const MultipleIO<T, LOPPER_TARGET>& val1,
                                                const MultipleIO<T, LOPPER_TARGET>& val2,
                                                const MultipleIO<T, LOPPER_TARGET>& val3) {
    VSTORE4(ptr, val0, val1, val2, val3);
  }
};

#endif

/*=============================== Expressions for writing memory ===============================*/

template<typename T, typename ... E> struct _ExprSaveBase : public NaryExpr<E...> {
  _ExprSaveBase(const std::shared_ptr<_Image<T>>& image, const E&... e)
    : NaryExpr<E...>(e...), _image(image) {
    if (image->getChannelCount() != sizeof...(E)) { throw LopperException("Invalid number of channels"); }
  }
  virtual ~_ExprSaveBase() {}
  virtual int getWidth() const { return _image->getWidth(); }
  virtual int getHeight() const { return _image->getHeight(); }
  virtual size_t getSIMDClearance() const = 0;
protected:
  std::shared_ptr<_Image<T>> _image;
};

template<typename T, typename E> struct _ExprSave1 : public _ExprSaveBase<T, E> {
  _ExprSave1(const std::shared_ptr<_Image<T>>& image, const E& e) : _ExprSaveBase<T, E>(image, e) {}

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

template<typename T, typename E, typename ... Es> struct _ExprSaveN : public _ExprSaveBase<T, E, Es...> {
  _ExprSaveN(std::shared_ptr<_Image<T>>& image, const E& e0, const Es& ... es)
    : _ExprSaveBase<T, E, Es...>(image, e0, es...) {}

  virtual size_t getSIMDClearance() const {
    const size_t bytes_written = _PixelStorer<LOPPER_TARGET>::template bytesPerOp<T, 1 + sizeof...(Es)>();
    const size_t bytes_per_pixel = sizeof(T) * (1 + sizeof...(Es));
    return (bytes_written + (bytes_per_pixel - 1)) / bytes_per_pixel;
  }

  void prepareRow(const int y) const {
    _ptr = this->_image->getRowPointer(y);
    this->_NaryExpr<1 + sizeof...(Es)>::template type<void, E, Es...>::prepareRow(y);
  }

  template<InstructionSet S, size_t U, typename ... Cxt> inline
  SFINAE<sizeof...(Es) == 2u, Multiple<typename E::type, S>> eval(const int x, const Cxt& ... args) const {
    const auto v0 = this->_e0.template eval<S, U>(x, args...);
    const auto v1 = this->_e1.template eval<S, U>(x, args...);
    const auto v2 = this->_e2.template eval<S, U>(x, args...);
    _PixelStorer<S>::template store<T>(_ptr + x * 3, v0, v1, v2);
    return v0;
  }

  template<InstructionSet S, size_t U, typename ... Cxt> inline
  SFINAE<sizeof...(Es) == 3u, Multiple<typename E::type, S>> eval(const int x, const Cxt& ... args) const {
    const auto v0 = this->_e0.template eval<S, U>(x, args...);
    const auto v1 = this->_e1.template eval<S, U>(x, args...);
    const auto v2 = this->_e2.template eval<S, U>(x, args...);
    const auto v3 = this->_e3.template eval<S, U>(x, args...);
    _PixelStorer<S>::template store<T>(_ptr + x * 4, v0, v1, v2, v3);
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

template<typename T, size_t C> struct _ExprImage : public NullaryExpr<typename IOTypeTrait<T>::type> {
  _ExprImage(std::shared_ptr<_Image<T>> image) : _image(image) {
    if (_image->getChannelCount() != C) { throw LopperException("Invalid number of channels"); }
  }

  virtual int getWidth() const { return _image->getWidth(); }
  virtual int getHeight() const { return _image->getHeight(); }
  virtual size_t getSIMDClearance() const {
    const size_t bytes_read = _PixelLoader<LOPPER_TARGET>::template bytesPerOp<T, C>();
    const size_t bytes_per_pixel = sizeof(T);
    return (bytes_read + (bytes_per_pixel - 1)) / bytes_per_pixel;
  }

  void prepareRow(const int y) const { _ptr = _image->getRowPointer(y); }

  template<InstructionSet S, size_t U, typename ... Cxt>
  MultipleIOTuple<T, C, S> inline eval(const int x, const Cxt& ... ) const {
    return _PixelLoader<S>::template load<T, C>(_ptr + x * C);
  }

  template<typename ... E> auto operator=(const std::tuple<E...>& t) ->
    SFINAE<(sizeof...(E) == C && sizeof...(E) == 3), _ExprSaveN<T, E...>> {
    return _ExprSaveN<T, E...>(_image, std::get<0>(t), std::get<1>(t), std::get<2>(t));
  }

  template<typename ... E> auto operator=(const std::tuple<E...>& t) ->
    SFINAE<(sizeof...(E) == C && sizeof...(E) == 4), _ExprSaveN<T, E...>> {
    return _ExprSaveN<T, E...>(_image, std::get<0>(t), std::get<1>(t), std::get<2>(t), std::get<3>(t));
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
template<size_t C, typename T, typename F = SFINAE<C == 2 || C == 3 || C == 4, _ExprImage<T, C>>>
  _ExprImage<T, C> Expr(::lopper::_Image<T>& image) {
  return _ExprImage<T, C>(std::shared_ptr<::lopper::_Image<T>>(std::shared_ptr<::lopper::_Image<T>>(),
                                                               dynamic_cast<::lopper::_Image<T>*>(&image)));
}

} // end namespace lopper
