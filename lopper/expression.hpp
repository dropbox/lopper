#pragma once

#include "multiple.hpp"
#include "util.hpp"

/*======================================== Expression types ========================================*/
// Expressions are basic building blocks that represents some symbolic expression.
// They should implement these three methods.
//  1) analyze(...);
//  2) prepareRow(const int y);
//  3) eval(const int x, ...);
//
// Prior to evaluation, analyze(...) may be called with an object that has a templatized operator(),
// and the children expressions will be fed to this operator. This also happens recursively, so it effectively
// provides a vehicle to run an analysis on the expression tree.
//
// During evaluation, prepareRow(y) will be called before processing each row of index y,
// and then for each pixel of each row, eval(...) will be called.
//
// Expressions should inherit from NullaryExpr, UnaryExpr, BinaryExpr or TernaryExpr.
// They provide the wiring for dispatching the above methods to subexpressions. It is up to the programmer
// to provide a kernel that combines the return values of eval(...) from the subexpressions.
// If a kernel is not provided, eval(...) should be implemented separately.
//
// There are some special expressions that allow reading and writing from memory as well.

namespace lopper {
namespace internal {

template<typename T, ::lopper::InstructionSet S> using Multiple = typename ::lopper::MultipleTrait<T, S>::vtype;

/* Base class for all expressions. */
struct _ExprBase {
  virtual ~_ExprBase() {}
};

template<typename T> struct _Expr : public _ExprBase {
  typedef T type;
  typedef T lopper_expr_type; // A sentinel with a fairly unique name for SFINAE
  // Should implement the aforementioned methods (analyze, prepareRow, eval),
  // but we do not declare them here since we're not using runtime polymorphism.
  // Since expressions are supposed to be inhereit from NullaryExpr, UnaryExpr, BinaryExpr or TernaryExpr,
  // some of these methods will be provided. Implement the missing ones or overload as appropriate.
};

// Represents a nullary expression that evaluates into Multiple<T> without depending on a subexpression.
template<typename T, typename K = void> struct NullaryExpr : public _Expr<T> {
  template<typename F> void analyze(F& func) const {
    func(this);
  }
  // prepareRow and eval are not implemented so that the compiler will fail loudly, should one
  // forget to implement them in a new subclass.
};

// Represents a unary expression that takes an input E0 and evaluates into Multiple<T>, e.g. negation.
template<typename T, typename E0, typename K = void> struct UnaryExpr : public _Expr<T> {
  const E0 _e0;
  UnaryExpr(const E0& e0) : _e0(e0) {}
  template<typename F> void analyze(F& func) const {
    func(_e0);
    _e0.analyze(func);
  }
  void prepareRow(const int y) const {
    _e0.prepareRow(y);
  }
  template<InstructionSet S, size_t U, typename ... Cxt> Multiple<T, S> inline eval(const int x, const Cxt& ... args) const {
    const auto v = _e0.template eval<S, U>(x, args...);
    return K::template eval<S>(v);
  }
};

// Represents a binary expression that evaluates into Multiple<T>, e.g. addition.
template<typename T, typename E0, typename E1, typename K = void> struct BinaryExpr : public _Expr<T> {
  const E0 _e0;
  const E1 _e1;
  BinaryExpr(const E0& e0, const E1& e1) : _e0(e0), _e1(e1) {}
  template<typename F> void analyze(F& func) const {
    func(_e0);
    func(_e1);
    _e0.analyze(func);
    _e1.analyze(func);
  }
  void prepareRow(const int y) const {
    _e0.prepareRow(y);
    _e1.prepareRow(y);
  }
  template<InstructionSet S, size_t U, typename ... Cxt> Multiple<T, S> inline eval(const int x, const Cxt& ... args) const {
    const auto v0 = _e0.template eval<S, U>(x, args...);
    const auto v1 = _e1.template eval<S, U>(x, args...);
    return K::template eval<S>(v0, v1);
  }
};

// Represents a ternary expression that evaluates into Multiple<T>.
template<typename T, typename E0, typename E1, typename E2, typename K = void> struct TernaryExpr : public _Expr<T> {
  const E0 _e0;
  const E1 _e1;
  const E2 _e2;
  TernaryExpr(const E0& e0, const E1& e1, const E2& e2) : _e0(e0), _e1(e1), _e2(e2) {}
  template<typename F> void analyze(F& func) const {
    func(_e0);
    func(_e1);
    func(_e2);
    _e0.analyze(func);
    _e1.analyze(func);
    _e2.analyze(func);
  }
  void prepareRow(const int y) const {
    _e0.prepareRow(y);
    _e1.prepareRow(y);
    _e2.prepareRow(y);
  }
  template<InstructionSet S, size_t U, typename ... Cxt> Multiple<T, S> inline eval(const int x, const Cxt& ... args) const {
    const auto v0 = _e0.template eval<S, U>(x, args...);
    const auto v1 = _e1.template eval<S, U>(x, args...);
    const auto v2 = _e2.template eval<S, U>(x, args...);
    return K::template eval<S>(v0, v1, v2);
  }
};

// Machinery to figure out the base expression type automatically, for non-nullary expressions.
// Preferably UnaryExpr, BinaryExpr, TernaryExpr should be used explicitly, but for some expressions
// that are variadic, NaryExpr<E0, ...> can be used.
template<size_t N> struct _NaryExpr {};
template<> struct _NaryExpr<1> {
  template<typename K, typename ... E> using type = UnaryExpr<typename NthTypeOf<0, E...>::type, E..., K>;
};
template<> struct _NaryExpr<2> {
  template<typename K, typename ... E> using type = BinaryExpr<typename NthTypeOf<0, E...>::type, E..., K>;
};
template<> struct _NaryExpr<3> {
  template<typename K, typename ... E> using type = TernaryExpr<typename NthTypeOf<0, E...>::type, E..., K>;
};
template<typename ... E> using NaryExpr =
  typename _NaryExpr<sizeof...(E)>::template type<void, E...>;
template<typename K, typename ... E> using NaryExprWithKernel =
  typename _NaryExpr<sizeof...(E)>::template type<K, E...>;

} // end namespace internal
} // end namespace lopper
