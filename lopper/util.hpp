#pragma once

#include <exception>
#include <string>
#include <tuple>

namespace lopper {

class LopperException : public std::exception {
public:
  LopperException(const char* msg) : m_msg(msg) {}
  virtual ~LopperException() throw () {}
  virtual const char* what() const noexcept { return m_msg.c_str(); }
private:
  std::string m_msg;
};

namespace internal {

template<bool condition, typename T = void> using SFINAE = typename std::enable_if<condition, T>::type;

template<size_t N, typename ... Ctx> using NthTypeOf =
  typename std::tuple_element<N, std::tuple<Ctx...>>::type;

template<size_t N, typename F, typename ... Cxt> inline SFINAE<N==0, F>
Selector(F arg, const Cxt& ... ) { return arg; }

template<size_t N, typename F, typename ... Cxt> inline SFINAE<(N>0), NthTypeOf<N, F, Cxt...>>
Selector(F, const Cxt& ... args) { return Selector<N-1>(args...); }

} // end namespace internal
} // end namespace lopper
