#pragma once

#include <stdint.h>
#include <type_traits>

#include "expression.hpp"
#include "multiple.hpp"
#include "primitives.hpp"
#include "storage.hpp"
#include "util.hpp"

namespace lopper {
using namespace lopper::internal;

// An expression representing a constant.
template<typename T> struct ExprConst : public NullaryExpr<T> {
  const T _val;
  ExprConst(const T& val) : _val(val) {}
  void prepareRow(const int) const {}
  template<InstructionSet S, size_t U, typename ... Cxt> inline
  Multiple<T, S> eval(const int, const Cxt& ... ) const {
    return VSET<S>(_val);
  }
};

// An expression representing a constexpr.
template<int32_t V> struct ExprConstTemplate : public NullaryExpr<int32_t> {
  void prepareRow(const int) const {}
  template<InstructionSet S, size_t U, typename ... Cxt> inline
  Multiple<int32_t, S> eval(const int, const Cxt& ... ) const {
    return VSET<S>(V);
  }
};

// An expression representing a variable that depends only on the row index.
template<typename T, typename F> struct ExprRowConst : public NullaryExpr<T> {
  mutable _Storage<T> _storage;
  ExprRowConst(const F& func) : _func(func) {}
  void prepareRow(const int y) const {
    auto val = _func(y);
    _storage.template store<SCALAR>(val);
    _storage.template store<LOPPER_TARGET>(VSET<LOPPER_TARGET>(val));
  }
  template<InstructionSet S, size_t U, typename ... Cxt> inline
  Multiple<T, S> eval(const int, const Cxt& ... ) const {
    return _storage.template get<S>();
  }
private:
  const std::function<T(int)> _func;
};

// Allow Expr<...>(constant).
template<typename T, typename T_> // require explicit specification of type T
SFINAE<std::is_same<T, T_>::value, ExprConst<T_>> Expr(const T_ val) {
  return ExprConst<T_>(val);
}

template<int32_t V> ExprConstTemplate<V> Expr() {
  return ExprConstTemplate<V>();
}

// Allow Expr<...>(std::function<T(int)>) that varies by row.
template<typename T, typename F, typename T_ = typename std::result_of<F(const int)>::type>
SFINAE<std::is_same<T, T_>::value, ExprRowConst<T_, F>> Expr(const F& func) {
  return ExprRowConst<T_, F>(func);
}

} // end namespace lopper
