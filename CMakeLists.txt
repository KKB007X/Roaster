cmake_minimum_required(VERSION 3.16)
project(roaster C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)
find_package(Vulkan REQUIRED)

find_program(GLSLC_EXECUTABLE glslc)
find_program(GLSLANG_VALIDATOR_EXECUTABLE glslangValidator)
set(ROASTER_SHADER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders")
set(ROASTER_SHADER_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY "${ROASTER_SHADER_BINARY_DIR}")

set(ROASTER_SHADER_OUTPUTS)
foreach(shader q4k_matvec q6k_matvec)
    if(GLSLC_EXECUTABLE)
        add_custom_command(
            OUTPUT "${ROASTER_SHADER_BINARY_DIR}/${shader}.spv"
            COMMAND "${GLSLC_EXECUTABLE}" --target-env=vulkan1.1
                    "${ROASTER_SHADER_SOURCE_DIR}/${shader}.comp"
                    -o "${ROASTER_SHADER_BINARY_DIR}/${shader}.spv"
            DEPENDS "${ROASTER_SHADER_SOURCE_DIR}/${shader}.comp"
            VERBATIM)
    elseif(GLSLANG_VALIDATOR_EXECUTABLE)
        add_custom_command(
            OUTPUT "${ROASTER_SHADER_BINARY_DIR}/${shader}.spv"
            COMMAND "${GLSLANG_VALIDATOR_EXECUTABLE}" -V --target-env vulkan1.1
                    "${ROASTER_SHADER_SOURCE_DIR}/${shader}.comp"
                    -o "${ROASTER_SHADER_BINARY_DIR}/${shader}.spv"
            DEPENDS "${ROASTER_SHADER_SOURCE_DIR}/${shader}.comp"
            VERBATIM)
    else()
        message(FATAL_ERROR "Vulkan shader compiler not found. Install glslc or glslangValidator (Ubuntu: glslc or glslang-tools).")
    endif()
    list(APPEND ROASTER_SHADER_OUTPUTS "${ROASTER_SHADER_BINARY_DIR}/${shader}.spv")
endforeach()
add_custom_target(roaster_shaders ALL DEPENDS ${ROASTER_SHADER_OUTPUTS})

add_executable(roaster
    src/main.c
    src/tokenizer.c
    src/weights.c
    src/model.c
    src/kernels.c
    src/slots.c
    src/server.c
    src/client.c
)
add_dependencies(roaster roaster_shaders)

target_include_directories(roaster PRIVATE src)
target_compile_definitions(roaster PRIVATE ROASTER_SHADER_DIR="${ROASTER_SHADER_BINARY_DIR}")
target_compile_options(roaster PRIVATE -O3 -march=native -Wall -Wextra)
target_link_libraries(roaster PRIVATE m OpenMP::OpenMP_C Threads::Threads Vulkan::Vulkan)
