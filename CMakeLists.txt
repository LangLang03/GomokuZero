cmake_minimum_required(VERSION 3.18)
project(gomoku_cpp LANGUAGES CXX)

# Sanitizer instrumentation. TSan cannot coexist with the CUDA runtime, so the
# LibTorch backend is forced off below when thread sanitizing.
set(GOMOK_SANITIZER "" CACHE STRING
    "Sanitizers to enable: empty, address, thread, undefined, or a comma list")

set(GOMOK_SANITIZER_FLAGS "")
if(GOMOK_SANITIZER)
    string(REPLACE "," ";" _san_list "${GOMOK_SANITIZER}")
    foreach(_san IN LISTS _san_list)
        if(_san STREQUAL "address")
            list(APPEND GOMOK_SANITIZER_FLAGS -fsanitize=address)
        elseif(_san STREQUAL "thread")
            list(APPEND GOMOK_SANITIZER_FLAGS -fsanitize=thread)
        elseif(_san STREQUAL "undefined")
            list(APPEND GOMOK_SANITIZER_FLAGS -fsanitize=undefined)
        else()
            message(FATAL_ERROR "Unknown sanitizer: ${_san}")
        endif()
    endforeach()
    list(APPEND GOMOK_SANITIZER_FLAGS -fno-omit-frame-pointer -g)
    message(STATUS "Gomoku sanitizers: ${GOMOK_SANITIZER}")
endif()

option(GOMOK_ENABLE_CUDA "Enable the optional LibTorch/CUDA backend" ON)
if(GOMOK_SANITIZER MATCHES "thread" AND GOMOK_ENABLE_CUDA)
    message(STATUS
        "Gomoku: TSan selected -> disabling the LibTorch/CUDA backend "
        "(the CUDA runtime is incompatible with ThreadSanitizer)")
    set(GOMOK_ENABLE_CUDA OFF)
endif()
if(GOMOK_ENABLE_CUDA)
    execute_process(
        COMMAND python3 -c "import pathlib, torch; print(pathlib.Path(torch.__file__).parent)"
        OUTPUT_VARIABLE GOMOK_TORCH_ROOT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(EXISTS "${GOMOK_TORCH_ROOT}/lib/libtorch_cuda.so")
        set(Torch_FOUND TRUE)
        set(TORCH_VERSION "Python wheel")
        set(GOMOK_TORCH_INCLUDE "${GOMOK_TORCH_ROOT}/include")
        set(GOMOK_TORCH_LIB "${GOMOK_TORCH_ROOT}/lib")
        set(TORCH_LIBRARIES
            "${GOMOK_TORCH_LIB}/libtorch.so"
            "${GOMOK_TORCH_LIB}/libtorch_cuda.so"
            "${GOMOK_TORCH_LIB}/libtorch_cpu.so"
            "${GOMOK_TORCH_LIB}/libc10_cuda.so"
            "${GOMOK_TORCH_LIB}/libc10.so")
    endif()
endif()

include(CheckIPOSupported)
option(GOMOK_ENABLE_IPO "Enable release link-time optimization" ON)
if(GOMOK_SANITIZER AND GOMOK_ENABLE_IPO)
    # LTO on instrumented translation units is unreliable.
    set(GOMOK_ENABLE_IPO OFF)
endif()
if(GOMOK_ENABLE_IPO)
    check_ipo_supported(RESULT GOMOK_IPO_SUPPORTED OUTPUT GOMOK_IPO_ERROR)
else()
    set(GOMOK_IPO_SUPPORTED FALSE)
endif()

# Applies the configured sanitizer flags to a target.
# Applies the sanitizer-only network overrides to a target.
function(gomok_apply_net_shape target)
    if(GOMOK_SANITIZER_CHANNELS)
        target_compile_definitions(${target} PRIVATE
            GOMOK_NET_CHANNELS=${GOMOK_SANITIZER_CHANNELS})
    endif()
    if(GOMOK_SANITIZER_RES_BLOCKS)
        target_compile_definitions(${target} PRIVATE
            GOMOK_NET_RES_BLOCKS=${GOMOK_SANITIZER_RES_BLOCKS})
    endif()
endfunction()

function(gomok_apply_sanitizer target)
    if(NOT GOMOK_SANITIZER_FLAGS)
        return()
    endif()
    target_compile_options(${target} PRIVATE ${GOMOK_SANITIZER_FLAGS})
    target_link_options(${target} PRIVATE ${GOMOK_SANITIZER_FLAGS})
endfunction()

# Shrinks the trunk for sanitizer runs. TSan instruments every memory access,
# so the production 12x192 trunk turns a self-play run into a very long wait;
# the code under test (MCTS, queues, node arena) does not depend on the shape.
set(GOMOK_SANITIZER_CHANNELS "" CACHE STRING
    "Override NetConfig::kChannels (sanitizer builds only)")
set(GOMOK_SANITIZER_RES_BLOCKS "" CACHE STRING
    "Override NetConfig::kResBlocks (sanitizer builds only)")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Use LibTorch from the Python wheel when CUDA is available.

add_executable(gomoku_train
    src/main.cpp
    src/game.cpp
    src/network.cpp
    src/gradient.cpp
    src/selfplay.cpp
    src/train.cpp
    src/logger.cpp
)

target_include_directories(gomoku_train PRIVATE src)
target_link_libraries(gomoku_train PRIVATE pthread)
if(Torch_FOUND)
    # LTO on this LibTorch translation unit can exhaust memory.
    set_source_files_properties(src/network_cuda.cpp PROPERTIES
        COMPILE_OPTIONS "-O1;-fno-lto")
    target_sources(gomoku_train PRIVATE src/network_cuda.cpp)
    target_compile_definitions(gomoku_train PRIVATE GOMOK_HAS_TORCH=1
                               _GLIBCXX_USE_CXX11_ABI=1)
    target_include_directories(gomoku_train SYSTEM PRIVATE
        "${GOMOK_TORCH_INCLUDE}"
        "${GOMOK_TORCH_INCLUDE}/torch/csrc/api/include")
    target_link_libraries(gomoku_train PRIVATE "${TORCH_LIBRARIES}")
    set_property(TARGET gomoku_train PROPERTY BUILD_RPATH
                 "${GOMOK_TORCH_LIB}")
    message(STATUS "Gomoku CUDA backend: enabled (${TORCH_VERSION})")
else()
    target_sources(gomoku_train PRIVATE src/network_cuda_stub.cpp)
    message(STATUS "Gomoku CUDA backend: unavailable; CPU fallback only")
endif()
if(GOMOK_IPO_SUPPORTED)
    set_property(TARGET gomoku_train PROPERTY
                 INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()
gomok_apply_net_shape(gomoku_train)
gomok_apply_sanitizer(gomoku_train)

include(CTest)
if(BUILD_TESTING)
    add_executable(gomoku_smoke
        tests/smoke.cpp
        src/game.cpp
        src/network.cpp
        src/gradient.cpp
        src/selfplay.cpp
    )
    target_include_directories(gomoku_smoke PRIVATE src)
    target_link_libraries(gomoku_smoke PRIVATE pthread)
    if(Torch_FOUND)
        target_sources(gomoku_smoke PRIVATE src/network_cuda.cpp)
        target_compile_definitions(gomoku_smoke PRIVATE GOMOK_HAS_TORCH=1
                                   _GLIBCXX_USE_CXX11_ABI=1)
        target_include_directories(gomoku_smoke SYSTEM PRIVATE
            "${GOMOK_TORCH_INCLUDE}"
            "${GOMOK_TORCH_INCLUDE}/torch/csrc/api/include")
        target_link_libraries(gomoku_smoke PRIVATE "${TORCH_LIBRARIES}")
        set_property(TARGET gomoku_smoke PROPERTY BUILD_RPATH
                     "${GOMOK_TORCH_LIB}")
    else()
        target_sources(gomoku_smoke PRIVATE src/network_cuda_stub.cpp)
    endif()
    if(GOMOK_IPO_SUPPORTED)
        set_property(TARGET gomoku_smoke PROPERTY
                     INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()
    gomok_apply_net_shape(gomoku_smoke)
    gomok_apply_sanitizer(gomoku_smoke)
    add_test(NAME gomoku_smoke COMMAND gomoku_smoke)

    add_executable(gomoku_game_test
        tests/game_test.cpp
        src/game.cpp
    )
    target_include_directories(gomoku_game_test PRIVATE src)
    gomok_apply_sanitizer(gomoku_game_test)
    add_test(NAME gomoku_game_test COMMAND gomoku_game_test)

    add_executable(gomoku_gradient_test
        tests/gradient_check.cpp
        src/gradient.cpp
    )
    target_include_directories(gomoku_gradient_test PRIVATE src)
    gomok_apply_sanitizer(gomoku_gradient_test)
    add_test(NAME gomoku_gradient_test COMMAND gomoku_gradient_test)

    # CPU forward vs LibTorch forward on identical weights.
    add_executable(gomoku_backend_parity
        tests/backend_parity.cpp
        src/game.cpp
        src/network.cpp
        src/gradient.cpp
        src/logger.cpp
    )
    target_include_directories(gomoku_backend_parity PRIVATE src)
    target_link_libraries(gomoku_backend_parity PRIVATE pthread)
    if(Torch_FOUND)
        target_sources(gomoku_backend_parity PRIVATE src/network_cuda.cpp)
        target_compile_definitions(gomoku_backend_parity PRIVATE GOMOK_HAS_TORCH=1
                                   _GLIBCXX_USE_CXX11_ABI=1)
        target_include_directories(gomoku_backend_parity SYSTEM PRIVATE
            "${GOMOK_TORCH_INCLUDE}"
            "${GOMOK_TORCH_INCLUDE}/torch/csrc/api/include")
        target_link_libraries(gomoku_backend_parity PRIVATE "${TORCH_LIBRARIES}")
        set_property(TARGET gomoku_backend_parity PROPERTY BUILD_RPATH
                     "${GOMOK_TORCH_LIB}")
    else()
        target_sources(gomoku_backend_parity PRIVATE src/network_cuda_stub.cpp)
    endif()
    gomok_apply_net_shape(gomoku_backend_parity)
    gomok_apply_sanitizer(gomoku_backend_parity)
    add_test(NAME gomoku_backend_parity COMMAND gomoku_backend_parity)

    # Pins the correctness fixes reported against the source.
    add_executable(gomoku_regression
        tests/regression.cpp
        src/game.cpp
        src/network.cpp
        src/gradient.cpp
        src/selfplay.cpp
        src/logger.cpp
    )
    target_include_directories(gomoku_regression PRIVATE src)
    target_link_libraries(gomoku_regression PRIVATE pthread)
    if(Torch_FOUND)
        target_sources(gomoku_regression PRIVATE src/network_cuda.cpp)
        target_compile_definitions(gomoku_regression PRIVATE GOMOK_HAS_TORCH=1
                                   _GLIBCXX_USE_CXX11_ABI=1)
        target_include_directories(gomoku_regression SYSTEM PRIVATE
            "${GOMOK_TORCH_INCLUDE}"
            "${GOMOK_TORCH_INCLUDE}/torch/csrc/api/include")
        target_link_libraries(gomoku_regression PRIVATE "${TORCH_LIBRARIES}")
        set_property(TARGET gomoku_regression PROPERTY BUILD_RPATH
                     "${GOMOK_TORCH_LIB}")
    else()
        target_sources(gomoku_regression PRIVATE src/network_cuda_stub.cpp)
    endif()
    gomok_apply_net_shape(gomoku_regression)
    gomok_apply_sanitizer(gomoku_regression)
    add_test(NAME gomoku_regression COMMAND gomoku_regression)
endif()
