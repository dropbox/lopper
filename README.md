Lopper
======
[![Build status](https://travis-ci.org/dropbox/lopper.svg?branch=master)](https://travis-ci.org/dropbox/lopper)

Lopper is a lightweight C++ template meta-programming framework for making vectorizing image-processing code easy, synthesized during Dropbox's 2016 hack week. It was inspired by many (more general) predecessors like Eigen and Halide, but the core focus was in enabling precise control over inlining of primitive computation. It can target platforms that support SSE (4.2) or NEON instructions. Note that Lopper does some very naughty things, like placing variables on the stack without being explicit about it, so take care when using it.

Requirements
------------

Lopper consists only of C++ header files, so no pre-compilation is necessary. Simply include "lopper/lopper.hpp" and compile your code with C++11-standard-compliant compiler. You will need CMake to build the unit test.

Usage
-----

The code snippets below assume that `using namespace lopper` is in effect.

Images can be wrapped into an expression as follows:
```
auto a = Expr<1>(image); // image must be a 1-channel image.
auto rgb = ExprCache(Expr<3>(image)); // multi-channel images must first be wrapped by ExprCache.
auto r = rgb.get<0>();
```

One can combine expressions to form other expressions:
```
auto c = a + a * a;
auto d = c + 3;
auto e = c * d; // implicitly expand to (a + a * a) * ((a + a * a) + 3)
...
```

In general, every expression will be inlined during evaluation. In order to reuse values, use `ExprCache`; you must assign the resulting expression to a variable, owing to macro expansion that happens, and must first declare `ExprPrepareContext` as shown below, but it can be very powerful in controlling exactly what arithmetic happens in the evaluation.
```
ExprPrepareContext();
auto c = ExprCache(a + a * a);
auto d = c + 3;
auto e = c * d; // implicitly equivalent to c = a + a * a; e = c * (c + 3)
...
```

To trigger evaluation, use the `ExprEval` macro on an assignment operation if you haven't inlined anything:
```
auto a = Expr<1>(image1) + Expr<1>(image2);
ExprEval(Expr<1>(image3) = a * a);
```

On the other hand, you must use `ExprEvalWithContext` macro otherwise:
```
ExprPrepareContext();
auto a = ExprCache(Expr<1>(image1) + Expr<1>(image2));
ExprEvalWithContext(Expr<1>(image3) = a * a);
```

Lopper supports rudimentary index manipulation, without providing the full functional expansion (e.g. Halide). Any expression that's instantiated directly from a single-channel image can call `reindex` or `offset`.
```
Expr<1>(image1).offset(1, 0) - Expr<1>(image1); // expression for forward horizontal gradient
Expr<1>(image1).reindex([](int y) { return image1.getHeight() - 1 - y; }); // expression for flipping the image vertically.
```

See the unit tests for more examples.

Requirements for Contributors
-----------------------------

If you plan to contribute a patch, please read the Contributor License Agreement at https://opensource.dropbox.com/cla/.

License
-------
Lopper is offered under Apache License 2.0. Please see license.txt for details.

Contributors
------------
Lopper was initially written by Jongmin Baek (jongmin@dropbox.com) with plenty of help and advice from Leonard Fink (leonard@dropbox.com), Lailin Chen (lailin@dropbox.com) and Ying Xiong (yxiong@dropbox.com).
