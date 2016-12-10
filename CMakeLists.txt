cmake_minimum_required(VERSION 3.5)
project(fastText)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
    src/args.cc
    src/dictionary.cc
    src/fasttext.cc
    src/main.cc
    src/matrix.cc
    src/model.cc
    src/utils.cc
    src/vector.cc
    Makefile)

find_package(Threads)
add_executable(fastText ${SOURCE_FILES})
target_link_libraries(fastText ${CMAKE_THREAD_LIBS_INIT})