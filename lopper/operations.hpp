#pragma once

#include <stdint.h>

#include "expression.hpp"
#include "util.hpp"

namespace lopper {
namespace internal {

/*=============================== UnaryExpr ===============================*/

template<typename T_out, typename T_in> struct _OperationTypeConvert {
  template<InstructionSet S> static Multiple<T_out, S> eval(const Multiple<T_in, S>& in) {
    return (T_out)in;
  }
};

template<> template<> inline Multiple<float, LOPPER_TARGET>
_OperationTypeConvert<float, int32_t>::eval<LOPPER_TARGET>(const Multiple<int32_t, LOPPER_TARGET>& in) {
  return VTO_FLOAT<LOPPER_TARGET>(in);
}

template<> template<> inline Multiple<int32_t, LOPPER_TARGET>
_OperationTypeConvert<int32_t, float>::eval<LOPPER_TARGET>(const Multiple<float, LOPPER_TARGET>& in) {
  return VTO_INT32<LOPPER_TARGET>(in);
}

template<typename T> struct _OperationAbs {
  typedef T type;
  template<InstructionSet S> static Multiple<T, S> eval(const Multiple<T, S>& in0) {
    return VABS(in0);
  }
};

template<typename T, typename E> struct _ExprTypeConvert :
  public UnaryExpr<T, E, _OperationTypeConvert<T, typename E::type>> {
  _ExprTypeConvert(const E& e) : UnaryExpr<T, E, _OperationTypeConvert<T, typename E::type>>(e) {}
};

// In theory, T could be inferred from J::eval<S> methods, but this is internal.
template<typename T, typename E0, typename J> struct _ExprLambda1 : public UnaryExpr<T, E0, J> {
  _ExprLambda1(const E0& e0) : UnaryExpr<T, E0, J>(e0) {}
};

template<typename T, typename J, typename E0> _ExprLambda1<T, E0, J> ExprLambda(const E0& exp0) {
  return _ExprLambda1<T, E0, J>(exp0);
}

/*=============================== BinaryExpr ===============================*/

// For these kernels, the default eval will assume scalars. We'll specialize the case in which S = LOPPER_TARGET.
template<typename T> struct _OperationAdd {
  typedef T type;
  template<InstructionSet S> static Multiple<T, S> eval(const Multiple<T, S>& in0,
                                                        const Multiple<T, S>& in1) {
    return in0 + in1;
  }
};

template<typename T> struct _OperationSubtract {
  typedef T type;
  template<InstructionSet S> static Multiple<T, S> eval(const Multiple<T, S>& in0,
                                                        const Multiple<T, S>& in1) {
    return in0 - in1;
  }
};

template<typename T> struct _OperationMultiply {
  typedef T type;
  template<InstructionSet S> static Multiple<T, S> eval(const Multiple<T, S>& in0,
                                                        const Multiple<T, S>& in1) {
    return in0 * in1;
  }
};

template<typename T> struct _OperationDivide { // Caller should take care to not divide by zero.
  typedef T type;
  template<InstructionSet S> static Multiple<T, S> eval(const Multiple<T, S>& in0,
                                                        const Multiple<T, S>& in1) {
    return in0 / in1;
  }
};

template<typename T, size_t bits> struct _OperationShiftLeft {
  typedef int32_t type;
  template<InstructionSet S> static Multiple<int32_t, S> eval(const Multiple<int32_t, S>& in0) {
    return VSHIFTL<bits>(in0);
  }
};

template<typename T, size_t bits> struct _OperationShiftRight {
  typedef int32_t type;
  template<InstructionSet S> static Multiple<int32_t, S> eval(const Multiple<int32_t, S>& in0) {
    return VSHIFTR<bits>(in0);
  }
};

template<typename T> struct _OperationLessThan {
  typedef int32_t type;
  template<InstructionSet S> static Multiple<int32_t, S> eval(const Multiple<T, S>& in0,
                                                              const Multiple<T, S>& in1) {
    return in0 < in1 ? 0xffffffff : 0;
  }
};

template<typename T> struct _OperationEqual {
  typedef int32_t type;
  template<InstructionSet S> static Multiple<int32_t, S> eval(const Multiple<T, S>& in0,
                                                              const Multiple<T, S>& in1) {
    return in0 == in1 ? 0xffffffff : 0;
  }
};

template<typename T> struct _OperationMin {
  typedef T type;
  template<InstructionSet S> static Multiple<T, S> eval(const Multiple<T, S>& in0,
                                                        const Multiple<T, S>& in1) {
    return in0 < in1 ? in0 : in1;
  }
};

template<typename T> struct _OperationMax {
  typedef T type;
  template<InstructionSet S> static Multiple<T, S> eval(const Multiple<T, S>& in0,
                                                        const Multiple<T, S>& in1) {
    return in0 > in1 ? in0 : in1;
  }
};

template<typename T> struct _OperationBitwiseAnd {
  typedef int32_t type;
  template<InstructionSet S> static Multiple<int32_t, S> eval(const Multiple<int32_t, S>& in0,
                                                              const Multiple<int32_t, S>& in1) {
    return VBITWISE_AND(in0, in1);
  }
};

template<typename T> struct _OperationBitwiseOr {
  typedef int32_t type;
  template<InstructionSet S> static Multiple<int32_t, S> eval(const Multiple<int32_t, S>& in0,
                                                              const Multiple<int32_t, S>& in1) {
    return VBITWISE_OR(in0, in1);
  }
};

template<> template<> inline Multiple<float, LOPPER_TARGET>
_OperationAdd<float>::eval<LOPPER_TARGET>(const Multiple<float, LOPPER_TARGET>& in0,
                                       const Multiple<float, LOPPER_TARGET>& in1) {
  return VADD(in0, in1);
}

template<> template<> inline Multiple<int32_t, LOPPER_TARGET>
_OperationAdd<int32_t>::eval<LOPPER_TARGET>(const Multiple<int32_t, LOPPER_TARGET>& in0,
                                         const Multiple<int32_t, LOPPER_TARGET>& in1) {
  return VADD(in0, in1);
}

template<> template<> inline Multiple<float, LOPPER_TARGET>
_OperationSubtract<float>::eval<LOPPER_TARGET>(const Multiple<float, LOPPER_TARGET>& in0,
                                            const Multiple<float, LOPPER_TARGET>& in1) {
  return VSUB(in0, in1);
}

template<> template<> inline Multiple<int32_t, LOPPER_TARGET>
_OperationSubtract<int32_t>::eval<LOPPER_TARGET>(const Multiple<int32_t, LOPPER_TARGET>& in0,
                                              const Multiple<int32_t, LOPPER_TARGET>& in1) {
  return VSUB(in0, in1);
}

template<> template<> inline Multiple<float, LOPPER_TARGET>
_OperationMultiply<float>::eval<LOPPER_TARGET>(const Multiple<float, LOPPER_TARGET>& in0,
                                            const Multiple<float, LOPPER_TARGET>& in1) {
  return VMUL(in0, in1);
}

template<> template<> inline Multiple<int32_t, LOPPER_TARGET>
_OperationMultiply<int32_t>::eval<LOPPER_TARGET>(const Multiple<int32_t, LOPPER_TARGET>& in0,
                                              const Multiple<int32_t, LOPPER_TARGET>& in1) {
  return VMUL(in0, in1);
}

template<> template<> inline Multiple<float, LOPPER_TARGET>
_OperationDivide<float>::eval<LOPPER_TARGET>(const Multiple<float, LOPPER_TARGET>& in0,
                                          const Multiple<float, LOPPER_TARGET>& in1) {
  return VDIV_FAST(in0, in1);
}

template<> template<> inline Multiple<int32_t, LOPPER_TARGET>
_OperationLessThan<float>::eval<LOPPER_TARGET>(const Multiple<float, LOPPER_TARGET>& in0,
                                            const Multiple<float, LOPPER_TARGET>& in1) {
  return VLT<LOPPER_TARGET>(in0, in1);
}

template<> template<> inline Multiple<int32_t, LOPPER_TARGET>
_OperationLessThan<int32_t>::eval<LOPPER_TARGET>(const Multiple<int32_t, LOPPER_TARGET>& in0,
                                              const Multiple<int32_t, LOPPER_TARGET>& in1) {
  return VLT<LOPPER_TARGET>(in0, in1);
}

template<> template<> inline Multiple<int32_t, LOPPER_TARGET>
_OperationEqual<int32_t>::eval<LOPPER_TARGET>(const Multiple<int32_t, LOPPER_TARGET>& in0,
                                           const Multiple<int32_t, LOPPER_TARGET>& in1) {
  return VEQ<LOPPER_TARGET>(in0, in1);
}

template<> template<> inline Multiple<float, LOPPER_TARGET>
_OperationMin<float>::eval<LOPPER_TARGET>(const Multiple<float, LOPPER_TARGET>& in0,
                                       const Multiple<float, LOPPER_TARGET>& in1) {
  return VMIN(in0, in1);
}

template<> template<> inline Multiple<int32_t, LOPPER_TARGET>
_OperationMin<int32_t>::eval<LOPPER_TARGET>(const Multiple<int32_t, LOPPER_TARGET>& in0,
                                         const Multiple<int32_t, LOPPER_TARGET>& in1) {
  return VMIN(in0, in1);
}

template<> template<> inline Multiple<float, LOPPER_TARGET>
_OperationMax<float>::eval<LOPPER_TARGET>(const Multiple<float, LOPPER_TARGET>& in0,
                                       const Multiple<float, LOPPER_TARGET>& in1) {
  return VMAX(in0, in1);
}

template<> template<> inline Multiple<int32_t, LOPPER_TARGET>
_OperationMax<int32_t>::eval<LOPPER_TARGET>(const Multiple<int32_t, LOPPER_TARGET>& in0,
                                         const Multiple<int32_t, LOPPER_TARGET>& in1) {
  return VMAX(in0, in1);
}

/*=============================== TernaryExpr ===============================*/

template<typename T> struct _OperationIf {
  typedef T type;
  template<InstructionSet S> static Multiple<T, S> eval(const Multiple<int32_t, S>& in0,
                                                        const Multiple<T, S>& in1,
                                                        const Multiple<T, S>& in2) {
    return in0 ? in1 : in2;
  }
};

template<> template<> inline Multiple<float, LOPPER_TARGET>
_OperationIf<float>::eval<LOPPER_TARGET>(const Multiple<int32_t, LOPPER_TARGET>& in0,
                                      const Multiple<float, LOPPER_TARGET>& in1,
                                      const Multiple<float, LOPPER_TARGET>& in2) {
  return VSELECT<LOPPER_TARGET>(in0, in2, in1);
}

template<> template<> inline Multiple<int32_t, LOPPER_TARGET>
_OperationIf<int32_t>::eval<LOPPER_TARGET>(const Multiple<int32_t, LOPPER_TARGET>& in0,
                                        const Multiple<int32_t, LOPPER_TARGET>& in1,
                                        const Multiple<int32_t, LOPPER_TARGET>& in2) {
  return VSELECT<LOPPER_TARGET>(in0, in2, in1);
}

} // end namespace internal

template<typename T, typename E0>
const SFINAE<std::is_same<T, typename E0::lopper_expr_type>::value, E0> Expr(const E0& exp0) {
  return exp0;
}

template<typename T, typename E0, typename F = typename E0::lopper_expr_type>
SFINAE<(std::is_same<float, T>::value || std::is_same<int32_t, T>::value) && !std::is_same<T, F>::value,
       _ExprTypeConvert<T, E0>> Expr(const E0& exp0) {
  return _ExprTypeConvert<T, E0>(exp0);
}

template<typename E0> _ExprLambda1<typename E0::type, E0, _OperationAbs<typename E0::type>> ExprAbs(const E0& exp0) {
  return _ExprLambda1<typename E0::type, E0, _OperationAbs<typename E0::type>>(exp0);
}

template<size_t bits, typename E0> _ExprLambda1<typename E0::type, E0, _OperationShiftLeft<typename E0::type, bits>> ExprShiftLeft(const E0& exp0) {
  return _ExprLambda1<typename E0::type, E0, _OperationShiftLeft<typename E0::type, bits>>(exp0);
}

template<size_t bits, typename E0> _ExprLambda1<typename E0::type, E0, _OperationShiftRight<typename E0::type, bits>> ExprShiftRight(const E0& exp0) {
  return _ExprLambda1<typename E0::type, E0, _OperationShiftRight<typename E0::type, bits>>(exp0);
}

#define DECLARE_BINARY_KERNEL(KERNEL, NAME) template<typename E0, typename E1> \
  SFINAE<std::is_same<typename E0::type, typename E1::type>::value, \
         BinaryExpr<typename KERNEL<typename E0::lopper_expr_type>::type, E0, E1, \
                    KERNEL<typename E0::type>>> NAME(const E0& exp0, const E1& exp1) { \
    typedef typename E0::type T_in;                                     \
    typedef typename KERNEL<T_in>::type T_out;                          \
    return BinaryExpr<T_out, E0, E1, KERNEL<T_in>>(exp0, exp1); }

#define DECLARE_BINARY_KERNEL_WITH_CONST(KERNEL, NAME) template<typename E0> \
  SFINAE<std::is_same<typename E0::type, typename E0::lopper_expr_type>::value, \
         BinaryExpr<typename KERNEL<typename E0::lopper_expr_type>::type, E0, \
                    ExprConst<typename E0::type>, KERNEL<typename E0::type>>> \
    NAME(const E0& exp, const typename E0::type& val) {                 \
    typedef typename E0::type T_in;                                     \
    return NAME(exp, ExprConst<T_in>(val)); }

DECLARE_BINARY_KERNEL(_OperationAdd, operator+);
DECLARE_BINARY_KERNEL(_OperationSubtract, operator-);
DECLARE_BINARY_KERNEL(_OperationMultiply, operator*);
DECLARE_BINARY_KERNEL(_OperationDivide, operator/);
DECLARE_BINARY_KERNEL(_OperationLessThan, operator<);
DECLARE_BINARY_KERNEL(_OperationEqual, operator==);
DECLARE_BINARY_KERNEL(_OperationMin, ExprMin);
DECLARE_BINARY_KERNEL(_OperationMax, ExprMax);
DECLARE_BINARY_KERNEL(_OperationBitwiseAnd, operator&);
DECLARE_BINARY_KERNEL(_OperationBitwiseOr, operator|);

DECLARE_BINARY_KERNEL_WITH_CONST(_OperationAdd, operator+);
DECLARE_BINARY_KERNEL_WITH_CONST(_OperationSubtract, operator-);
DECLARE_BINARY_KERNEL_WITH_CONST(_OperationMultiply, operator*);
DECLARE_BINARY_KERNEL_WITH_CONST(_OperationLessThan, operator<);
DECLARE_BINARY_KERNEL_WITH_CONST(_OperationEqual, operator==);
DECLARE_BINARY_KERNEL_WITH_CONST(_OperationMin, ExprMin);
DECLARE_BINARY_KERNEL_WITH_CONST(_OperationMax, ExprMax);
DECLARE_BINARY_KERNEL_WITH_CONST(_OperationBitwiseAnd, operator&);
DECLARE_BINARY_KERNEL_WITH_CONST(_OperationBitwiseOr, operator|);

/* Provides a kernel that chains a given binary kernel, like the reduction operator */
template<typename K> struct _ChainedOperation {
  typedef typename K::type type;
  template<InstructionSet S> static Multiple<type, S> eval(const Multiple<type, S>& in0,
                                                           const Multiple<type, S>& in1) {
    return K::template eval<S>(in0, in1);
  }
  template<InstructionSet S, typename ... E> static Multiple<type, S> eval(const Multiple<type, S>& in0,
                                                                           const Multiple<type, S>& in1,
                                                                           const Multiple<type, S>& in2,
                                                                           const E&... e) {
    return _ChainedOperation<K>::template eval<S>(K::template eval<S>(in0, in1), in2, e...);
  }
};

#define DECLARE_VARIADIC_OP_FOR_BINARY_KERNEL(KERNEL, NAME) template<typename ... E> \
  SFINAE<(sizeof...(E) > 2), NaryExprWithKernel<_ChainedOperation<KERNEL<typename NthTypeOf<0, E...>::type>>, E...>> NAME(const E&... exp) { \
    return NaryExprWithKernel<_ChainedOperation<KERNEL<typename NthTypeOf<0, E...>::type>>, E...>(exp...); }

DECLARE_VARIADIC_OP_FOR_BINARY_KERNEL(_OperationMin, ExprMin);
DECLARE_VARIADIC_OP_FOR_BINARY_KERNEL(_OperationMax, ExprMax);

// TODO: In theory, T could be inferred from J::eval<S> methods.
template<typename T, typename J, typename E0, typename E1>
BinaryExpr<T, E0, E1, J> ExprLambda(const E0& exp0, const E1& exp1) {
  return BinaryExpr<T, E0, E1, J>(exp0, exp1);
}

template<typename E0, typename E1, typename E2>
TernaryExpr<typename E1::type, E0, E1, E2, _OperationIf<typename E1::type>>
  ExprIf(const E0& exp0, const E1& exp1, const E2& exp2) {
  typedef typename E1::type T;
  return TernaryExpr<T, E0, E1, E2, _OperationIf<T>>(exp0, exp1, exp2);
}

template<typename T, typename J, typename E0, typename E1, typename E2>
TernaryExpr<T, E0, E1, E2, J> ExprLambda(const E0& exp0, const E1& exp1, const E2& exp2) {
  return TernaryExpr<T, E0, E1, E2, J>(exp0, exp1, exp2);
}

} // end namespace lopper
