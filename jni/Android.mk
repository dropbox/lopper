LOCAL_PATH := $(call my-dir)

PLATFORM_DEPENDENT_FLAGS := -DLOPPER_TARGET_NEON -mfloat-abi=softfp -mfpu=neon

# Build the test.
include $(CLEAR_VARS)
LOCAL_MODULE	:= lopper_test
LOCAL_MODULEFILENAME := lopper_test
LOCAL_CPPFLAGS  := -Werror -Wno-unused -fexceptions -std=c++11 -O3 -g -fPIE $(PLATFORM_DEPENDENT_FLAGS)
LOCAL_SRC_FILES := $(shell cd $(LOCAL_PATH) && ls -1 ../tests/*.cpp)
LOCAL_C_INCLUDES := .
LOCAL_STATIC_LIBRARIES := googletest_main
include $(BUILD_EXECUTABLE)

$(call import-module,third_party/googletest)
