#pragma once

#include <algorithm>
#include <set>
#include <type_traits>
#include <vector>

#include "expression.hpp"
#include "multiple.hpp"
#include "util.hpp"

namespace lopper {
namespace internal {

/*======================================== Caching support  ========================================*/
// Some expressions interact with the context that's carried around during evaluation.

template<typename T, size_t UID, size_t C> struct _ExprContextTupleReader : public NullaryExpr<T> {
  void prepareRow(const size_t) const {}
  template<InstructionSet S, size_t U, typename ... Cxt>
  const Multiple<T, S> inline eval(const size_t, const Cxt& ... args) const {
    return std::get<C>(Selector<UID - U - 1>(args...));
  }
};

template<typename T, size_t UID> struct _ExprContextReader : public NullaryExpr<T> {
  void prepareRow(const size_t) const {}
  template<InstructionSet S, size_t U, typename ... Cxt>
  const Multiple<T, S> inline eval(const size_t, const Cxt& ... args) const {
    return Selector<UID - U - 1>(args...);
  }
  template<size_t C_new> _ExprContextTupleReader<T, UID, C_new> get() const {
    return _ExprContextTupleReader<T, UID, C_new>();
  }
};

/*=============================== Execution helpers ===============================*/
#define LOPPER_VALIDATE_RECURSIVE_CASE(X, index) SFINAE<(std::tuple_size<X>::value > index)>
#define LOPPER_VALIDATE_BASE_CASE(X, index) SFINAE<(std::tuple_size<X>::value == index)>

template<size_t index, typename DepInfo, typename F> LOPPER_VALIDATE_RECURSIVE_CASE(DepInfo, index)
_dependency_analyze(const DepInfo& depinfo, F& func) {
  func(std::get<index>(depinfo));
  std::get<index>(depinfo).analyze(func);
  _dependency_analyze<index + 1>(depinfo, func);
}

template<size_t index, typename DepInfo> LOPPER_VALIDATE_RECURSIVE_CASE(DepInfo, index)
_dependency_prepareRow(const DepInfo& depinfo, const int y) {
  std::get<index>(depinfo).prepareRow(y);
  _dependency_prepareRow<index + 1>(depinfo, y);
}

template<InstructionSet S, size_t DepOffset, size_t index, typename DepInfo, typename ... Cxt>
inline LOPPER_VALIDATE_RECURSIVE_CASE(DepInfo, index)
_dependency_eval(const DepInfo& depinfo, const int x, const Cxt& ... args) {
  _dependency_eval<S, DepOffset, index + 1>(depinfo, x, args...,
                                            std::get<index>(depinfo).template eval<S, DepOffset>(x, args...));
}

// The helpers will be called recursively, so they need a base case with index equaling the size of DepInfo.
template<size_t index, typename DepInfo, typename F> LOPPER_VALIDATE_BASE_CASE(DepInfo, index)
_dependency_analyze(const DepInfo&, F&) {}

template<size_t index, typename DepInfo> LOPPER_VALIDATE_BASE_CASE(DepInfo, index)
_dependency_prepareRow(const DepInfo&, const int) {}

template<InstructionSet S, size_t DepOffset, size_t index, typename DepInfo, typename ... Cxt>
inline LOPPER_VALIDATE_BASE_CASE(DepInfo, index)
_dependency_eval(const DepInfo&, const int, const Cxt& ...) {}

#undef LOPPER_VALIDATE_RECURSIVE_CASE
#undef LOPPER_VALIDATE_BASE_CASE

/* A helper struct that returns std::true_type or std::false_type depending on whether the given template type
 * has an instance method of the given name and return type.
 */
template<typename E> struct _ExprPropertyChecker {
  #define LOPPER_DECLARE_PROPERTY_CHECK(name, return_type) \
    template<typename C> static std::false_type name (...); \
    template<typename C> static typename std::is_same<return_type, decltype(std::declval<C>().name())>::type \
    name (decltype(&C::name));

  LOPPER_DECLARE_PROPERTY_CHECK(getWidth, int);
  LOPPER_DECLARE_PROPERTY_CHECK(getHeight, int);
  LOPPER_DECLARE_PROPERTY_CHECK(getSIMDClearance, size_t);
  LOPPER_DECLARE_PROPERTY_CHECK(getHorizontalOffset, int);
  #undef LOPPER_DECLARE_PROPERTY_CHECK
};

// A simple container to hold statistics as the expression tree is traversed.
struct _DimensionChecker {
  _DimensionChecker() {
    clearances.push_back(0u);
    offsets.push_back(0);
  }
  std::set<int> widths, heights;
  std::vector<int> offsets;
  std::vector<size_t> clearances;

  // By default, which assumes that the properties are missing, the property checks will be no-ops.
  template<typename HasProperty> struct _Handler {
    #define LOPPER_DECLARE_PROPERTY_CHECK_HANDLER(name) \
      template<typename E> static void name(_DimensionChecker&, const E&)
    LOPPER_DECLARE_PROPERTY_CHECK_HANDLER(getWidth) {}
    LOPPER_DECLARE_PROPERTY_CHECK_HANDLER(getHeight) {}
    LOPPER_DECLARE_PROPERTY_CHECK_HANDLER(getSIMDClearance) {}
    LOPPER_DECLARE_PROPERTY_CHECK_HANDLER(getHorizontalOffset) {}
    #undef LOPPER_DECLARE_PROPERTY_CHECK_HANDLER
  };
  template<typename E> void operator()(const E& e);
};

template<> struct _DimensionChecker::_Handler<std::true_type> {
  #define LOPPER_DECLARE_PROPERTY_CHECK_HANDLER(name) \
    template<typename E> static void name(_DimensionChecker& state, const E& e)
  LOPPER_DECLARE_PROPERTY_CHECK_HANDLER(getWidth) {
    state.widths.insert(e.getWidth());
  }
  LOPPER_DECLARE_PROPERTY_CHECK_HANDLER(getHeight) {
    // XXX: This is a hack because reindexed or translated _ExprImage still declares getHeight(),
    // which should be ignored. Hence it returns -1 at the moment to signal that it's to be ignored.
    auto h = e.getHeight();
    if (h != -1) {
      state.heights.insert(h);
    }
  }
  LOPPER_DECLARE_PROPERTY_CHECK_HANDLER(getSIMDClearance) {
    state.clearances.push_back(e.getSIMDClearance());
  }
  LOPPER_DECLARE_PROPERTY_CHECK_HANDLER(getHorizontalOffset) {
    state.offsets.push_back(e.getHorizontalOffset());
  }
  #undef LOPPER_DECLARE_PROPERTY_CHECK_HANDLER
};

template<typename E> void _DimensionChecker::operator()(const E& e) {
  #define LOPPER_EXECUTE_PROPERTY_CHECK(name) \
    _Handler<decltype(_ExprPropertyChecker<E>::template name<E>(nullptr))>::name(*this, e);
  LOPPER_EXECUTE_PROPERTY_CHECK(getWidth);
  LOPPER_EXECUTE_PROPERTY_CHECK(getHeight);
  LOPPER_EXECUTE_PROPERTY_CHECK(getSIMDClearance);
  LOPPER_EXECUTE_PROPERTY_CHECK(getHorizontalOffset);
  #undef LOPPER_EXECUTE_PROPERTY_CHECK
}

} // end namespace internal

using namespace lopper::internal;

template<size_t UID, typename Z> _ExprContextReader<typename Z::type, UID> ExprContextReader(const Z&) {
  return ::lopper::internal::_ExprContextReader<typename Z::type, UID>();
}

template<bool EnableSIMD, size_t DepOffset, typename DepInfo> void _execute(const DepInfo& depinfo) {
  // Ensure that the image dimensions are consistent, and figure out clearance.
  ::lopper::internal::_DimensionChecker checker;
  _dependency_analyze<0>(depinfo, checker);
  if (checker.widths.size() != 1u || checker.heights.size() != 1u) {
    throw LopperException("Image dimensions are not well-defined");
  }
  int width = *checker.widths.begin(), height = *checker.heights.begin();
  const auto translation_minmax = std::minmax_element(std::begin(checker.offsets), std::end(checker.offsets));
  const int clearance = (int)*std::max_element(std::begin(checker.clearances), std::end(checker.clearances));
  for (int y = 0; y < height; y++) {
    int i = 0;
    _dependency_prepareRow<0>(depinfo, y);
    const int cutoff = std::min<int>(-*translation_minmax.first, width);
    for (; i < cutoff; i++) {
      _dependency_eval<SCALAR, DepOffset, 0>(depinfo, i);
    }
    if (EnableSIMD) {
      constexpr int step = (int)InstructionSetTrait<LOPPER_TARGET>::num_lanes;
      const int cutoff_SIMD = std::max<int>(0, width - clearance + 1 - *translation_minmax.second);
      for (; i < cutoff_SIMD; i += step) {
        _dependency_eval<LOPPER_TARGET, DepOffset, 0>(depinfo, i);
      }
    }
    { // Handle stragglers.
      for (; i < width; i++) {
        _dependency_eval<SCALAR, DepOffset, 0>(depinfo, i);
      }
    }
  }
}

} // end namespace lopper
