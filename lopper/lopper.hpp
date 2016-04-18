#pragma once

#include <cmath>
#include <functional>
#include <memory>
#include <utility>
#include <tuple>
#include <type_traits>

#include "constants.hpp"
#include "context.hpp"
#include "expression.hpp"
#include "io.hpp"
#include "multiple.hpp"
#include "operations.hpp"
#include "platform.hpp"

#define MACRO_COMBINER_HELPER(a, b) a##b
#define MACRO_COMBINER(a, b) MACRO_COMBINER_HELPER(a, b)
#define DEC_1 0
#define DEC_2 1
#define DEC_3 2
#define DEC_4 3
#define DEC_5 4
#define DEC_6 5
#define DEC_7 6
#define DEC_8 7
#define DEC_9 8
#define DEC_10 9
#define DEC_11 10
#define DEC_12 11
#define DEC_13 12
#define DEC_14 13
#define DEC_15 14
#define DEC_16 15
#define DEC_17 16
#define DEC_18 17
#define DEC_19 18
#define DEC_20 19
#define DEC_21 20
#define DEC_22 21
#define DEC_23 22
#define DEC_24 23
#define DEC_25 24
#define DEC_26 25
#define DEC_27 26
#define DEC_28 27
#define DEC_29 28
#define DEC_30 29
#define DEC_31 30
#define DEC_32 31
#define DEC_33 32
#define DEC_34 33
#define DEC_35 34
#define DEC_36 35
#define DEC_37 36
#define DEC_38 37
#define DEC_39 38
#define DECREMENT(X) MACRO_COMBINER(DEC_, X)

#define CURRENT_DEPINFO(UID) MACRO_COMBINER(reserved_depinfo, UID)
#define PREVIOUS_DEPINFO(UID) MACRO_COMBINER(reserved_depinfo, DECREMENT(UID))
#define DEPINFO_CHAINING(UID, ...) auto CURRENT_DEPINFO(UID) = \
    std::tuple_cat(PREVIOUS_DEPINFO(UID), std::make_tuple(__VA_ARGS__)); \
  (void)CURRENT_DEPINFO(UID)

#define ExprPrepareContextHelper(UID) constexpr size_t reserved_depoffset = (size_t)(UID); \
  auto CURRENT_DEPINFO(UID) = std::make_tuple();
#define ExprPrepareContext() ExprPrepareContextHelper(__COUNTER__)

#define ExprCacheHelper(UID, ...) ExprContextReader<UID>(__VA_ARGS__); DEPINFO_CHAINING(UID, __VA_ARGS__)
#define ExprCache(...) ExprCacheHelper(__COUNTER__, __VA_ARGS__)

#define ExprEvalWithContextHelper(UID, S, ...) DEPINFO_CHAINING(UID, __VA_ARGS__); \
  lopper::_execute<S, reserved_depoffset>(CURRENT_DEPINFO(UID))
#define ExprEvalWithContext(...) ExprEvalWithContextHelper(__COUNTER__, true, __VA_ARGS__)
#define ExprEvalWithContextSIMD(S, ...) ExprEvalWithContextHelper(__COUNTER__, S, __VA_ARGS__)
#define ExprEval(...) lopper::_execute<true, 0>(std::make_tuple(__VA_ARGS__))
#define ExprEvalSIMD(S, ...) lopper::_execute<S, 0>(std::make_tuple(__VA_ARGS__))
