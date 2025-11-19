cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

project(wenet VERSION 0.1)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

option(GRPC "whether to build gRPC" OFF)

include_directories(
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/kaldi
)
set(CMAKE_VERBOSE_MAKEFILE on)

include(FetchContent)
include(ExternalProject)
set(FETCHCONTENT_QUIET off)
get_filename_component(fc_base "fc_base" REALPATH BASE_DIR "${CMAKE_SOURCE_DIR}")
set(FETCHCONTENT_BASE_DIR ${fc_base})

if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -fPIC")
endif()

# -------------------------------------------------------
# 1. GFLAGS (Force Static & Disable Install)
# -------------------------------------------------------
FetchContent_Declare(gflags
  URL      https://github.com/gflags/gflags/archive/v2.2.1.zip
  URL_HASH SHA256=4e44b69e709c826734dbbbd5208f61888a2faf63f239d73d8ba0011b2dccc97a
)
# Force Static Build
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GFLAGS_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GFLAGS_BUILD_STATIC_LIBS ON CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "" FORCE)
# FIX: Disable Gflags Install to prevent export noise
set(INSTALL_HEADERS OFF CACHE BOOL "" FORCE)
set(INSTALL_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(INSTALL_STATIC_LIBS OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(gflags)
include_directories(${gflags_BINARY_DIR}/include)


# -------------------------------------------------------
# 2. GLOG (The Manual Bypass Fix)
# -------------------------------------------------------
FetchContent_Declare(glog
  URL      https://github.com/google/glog/archive/v0.4.0.zip
  URL_HASH SHA256=9e1b54eb2782f53cd8af107ecf08d2ab64b8d0dc2b7f5594472f3bd63ca85cdc
)

# CRITICAL FIX: Tell glog NOT to use gflags during configuration.
# This prevents glog from creating the "install(EXPORT)" rule that crashes CMake.
set(WITH_GFLAGS OFF CACHE BOOL "Manual gflags link" FORCE)

FetchContent_MakeAvailable(glog)
include_directories(${glog_SOURCE_DIR}/src ${glog_BINARY_DIR})

# MANUAL WIRING: Now we manually link gflags so the code works.
target_link_libraries(glog PUBLIC gflags_nothreads_static)
target_compile_definitions(glog PUBLIC HAVE_LIB_GFLAGS=1)


# -------------------------------------------------------
# 3. Other Dependencies
# -------------------------------------------------------
# third_party: gtest
FetchContent_Declare(googletest
  URL      https://github.com/google/googletest/archive/release-1.10.0.zip
  URL_HASH SHA256=94c634d499558a76fa649edb13721dce6e98fb1e7018dfaeba3cd7a083945e91
)
FetchContent_MakeAvailable(googletest)

# third_party: boost
FetchContent_Declare(boost
  URL      https://archives.boost.io/release/1.75.0/source/boost_1_75_0.tar.gz
  URL_HASH SHA256=aeb26f80e80945e82ee93e5939baebdca47b9dee80a07d3144be1e1a6a66dd6a
)
FetchContent_MakeAvailable(boost)
include_directories(${boost_SOURCE_DIR})

# third_party: cnpy
FetchContent_Declare(cnpy
  URL      https://github.com/rogersce/cnpy/archive/refs/heads/master.zip
  URL_HASH SHA256=dd58304295ce30e2299fafe8a72cb1acac508139df740c8976bcaab5c17591dc
)
FetchContent_MakeAvailable(cnpy)
include_directories(${cnpy_SOURCE_DIR})

# Redis
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/hiredis)
set(REDIS_PLUS_PLUS_CXX_STANDARD 14)
set(HIREDIS_HEADER ${CMAKE_SOURCE_DIR}/third_party)
set(HIREDIS_LIB hiredis)
set(TEST_HIREDIS_LIB hiredis)
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/redis-plus-plus)

# third_party: libtorch
set(PYTORCH_VERSION "1.13.1")
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${PYTORCH_VERSION}%2Bcpu.zip")
    set(URL_HASH "SHA256=d289185802baa1977701b72fc17c8aa6715481703352ce02d6f710cf007dc4f9")
else()
    message(FATAL_ERROR "Unsupported CMake System Name")
endif()
FetchContent_Declare(libtorch
  URL      ${LIBTORCH_URL}
  URL_HASH ${URL_HASH}
)
set(gtest_force_shared_crt ON CACHE BOOL "Always use msvcrt.dll" FORCE)
FetchContent_MakeAvailable(libtorch)
find_package(Torch REQUIRED PATHS ${libtorch_SOURCE_DIR} NO_DEFAULT_PATH)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -DC10_USE_GLOG")

# utils
add_library(utils STATIC
  utils/string.cc
  utils/utils.cc
)

# OpenFST
if(NOT MSVC)
  set(openfst_SOURCE_DIR ${fc_base}/openfst-src)
  set(openfst_BINARY_DIR ${fc_base}/openfst-build)
  set(openfst_PREFIX_DIR ${fc_base}/openfst-subbuild/openfst-populate-prefix)
  ExternalProject_Add(openfst
    URL                   https://github.com/mjansche/openfst/archive/1.6.5.zip
    URL_HASH              SHA256=b720357a464f42e181d7e33f60867b54044007f50baedc8f4458a3926f4a5a78
    SOURCE_DIR            ${openfst_SOURCE_DIR}
    BINARY_DIR            ${openfst_BINARY_DIR}
    CONFIGURE_COMMAND ${openfst_SOURCE_DIR}/configure --prefix=${openfst_PREFIX_DIR}
                        "CPPFLAGS=-I${gflags_BINARY_DIR}/include -I${glog_SOURCE_DIR}/src -I${glog_BINARY_DIR} -D_GLIBCXX_USE_CXX11_ABI=0"
                        "LDFLAGS=-L${gflags_BINARY_DIR} -L${glog_BINARY_DIR}"
                        "LIBS=-lgflags_nothreads -lglog -lpthread"
    COMMAND               ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/patch/openfst ${openfst_SOURCE_DIR}
    BUILD_COMMAND         make -j 4
  )
  add_dependencies(openfst gflags glog)
  add_dependencies(utils openfst)
  link_directories(${openfst_PREFIX_DIR}/lib)
  target_link_libraries(utils PUBLIC fst dl)
else()
endif()
include_directories(${openfst_SOURCE_DIR}/src/include)

# kaldi
add_subdirectory(kaldi)

# decoder
add_library(decoder STATIC
  decoder/ctc_prefix_beam_search.cc
  decoder/ctc_wfst_beam_search.cc
  decoder/brain_speech_decoder.cc
)
target_link_libraries(decoder PUBLIC ${TORCH_LIBRARIES} kaldi-decoder utils kaldi-fstext)

find_package(ZLIB REQUIRED)

# binary
add_executable(brain_speech_decoder_main bin/brain_speech_decoder_main.cc)
target_link_libraries(brain_speech_decoder_main PUBLIC decoder cnpy-static ${ZLIB_LIBRARIES})

add_subdirectory(pybind11)
pybind11_add_module(lm_decoder python/lm_decoder.cc)
target_link_libraries(lm_decoder PUBLIC decoder)