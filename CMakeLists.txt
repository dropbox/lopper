cmake_minimum_required(VERSION 2.8)
project (lopper CXX)

# Declare dependency on gtest.
enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Check for C++11
include (CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG ("-std=c++11" COMPILER_SUPPORTS_CXX11)
if (NOT COMPILER_SUPPORTS_CXX11)
  message (FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support")
else ()
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif ()

# Header files directory.
include_directories (".")

# Turn on AVX.
OPTION (USE_AVX "Use Intel Streaming SIMD Extensions (AVX)" OFF)
if (USE_AVX)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
endif (USE_AVX)

# Turn on SSE.
OPTION (USE_SSE "Use Intel Streaming SIMD Extensions (SSE)" ON)
if (USE_SSE AND NOT USE_AVX)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2")
endif ()

# Source files.
file (GLOB lopper_test_files "tests/*.cpp")

# Build unit test.
add_executable (lopper_test ${lopper_test_files})

# Link targets.
target_link_libraries (lopper_test ${GTEST_BOTH_LIBRARIES} pthread)

# Add target for running test.
add_custom_target (run_test "./lopper_test" DEPENDS lopper_test)

# Add target for running Valgrind.
find_program (MEMCHECK_COMMAND valgrind)
set (MEMCHECK_OPTIONS --leak-check=full --error-exitcode=1 --quiet)
set (MEMCHECK_TEST ${MEMCHECK_COMMAND} ${MEMCHECK_OPTIONS} ${CMAKE_CURRENT_BINARY_DIR}/lopper_test)
add_custom_target (run_valgrind COMMAND ${MEMCHECK_TEST} DEPENDS lopper_test)
