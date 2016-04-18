#pragma once

#include "multiple.hpp"

namespace lopper {
namespace internal {

template<typename T, ::lopper::InstructionSet S> using Multiple = typename ::lopper::MultipleTrait<T, S>::vtype;

// _Storage is an internal class used by the evaluation context for storing the result of
// intermediate computations. It implements a simple interface (store() and get()).
template<typename T> struct _Storage {
  // At the time of declaration, we don't know whether we will use SIMD or not, so we provide containers for both.
  Multiple<T, LOPPER_TARGET> _val_vector;
  Multiple<T, SCALAR> _val_scalar;
  template<InstructionSet S> const Multiple<T, S>& get() const;
  template<InstructionSet S> void store(const Multiple<T, S>& v);
};

template<> template<> inline const Multiple<float, SCALAR>& _Storage<float>::get<SCALAR>() const { return _val_scalar; }
template<> template<> inline const Multiple<int32_t, SCALAR>& _Storage<int32_t>::get<SCALAR>() const { return _val_scalar; }
template<> template<> inline void _Storage<float>::store<SCALAR>(const Multiple<float, SCALAR>& v) { _val_scalar = v; }
template<> template<> inline void _Storage<int32_t>::store<SCALAR>(const Multiple<int32_t, SCALAR>& v) { _val_scalar = v; }

#ifndef LOPPER_NO_SIMD
template<> template<> inline const Multiple<float, LOPPER_TARGET>& _Storage<float>::get<LOPPER_TARGET>() const { return _val_vector; }
template<> template<> inline const Multiple<int32_t, LOPPER_TARGET>& _Storage<int32_t>::get<LOPPER_TARGET>() const { return _val_vector; }
template<> template<> inline void _Storage<float>::store<LOPPER_TARGET>(const Multiple<float, LOPPER_TARGET>& v) { _val_vector = v; }
template<> template<> inline void _Storage<int32_t>::store<LOPPER_TARGET>(const Multiple<int32_t, LOPPER_TARGET>& v) { _val_vector = v; }
#endif

} // end namespace internal
} // end namespace lopper
