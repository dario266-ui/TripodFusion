cmake_minimum_required(VERSION 3.22.1)
project("tripodfusion")

add_library(
    tripodfusion
    SHARED
    native_fusion.cpp
)

find_library(log-lib log)
find_library(jnigraphics-lib jnigraphics)

target_link_libraries(
    tripodfusion
    ${log-lib}
    ${jnigraphics-lib}
)
