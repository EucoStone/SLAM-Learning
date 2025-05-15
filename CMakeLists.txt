#<------chapter 1------>
#设置最小要求的cmake版本
cmake_minimum_required(VERSION 2.8)
#声明工程
project(SLAM学习用)
#设置编译模式为debug
#SET(CMAKE_BUILD_TYPE "Debug")
#添加一个库
#语法为 add_library(库名字  库源代码名字)
#可以创建两种库，静态库每次编译都会生成一个，共享库只有一个
add_library(hello_shared SHARED libmain.cpp)
#添加可执行文件
#语法为 add_executable(可执行程序名字  程序源代码名字)
add_executable(main main.cpp)
add_executable(useHello useHello.cpp)
#连接运行的代码与库
target_link_libraries(useHello hello_shared)
target_link_libraries(main hello_shared)
#用于将指定目录拉入编译器搜索目录之中
#include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#<------chapter 2------>
add_executable(useEigen useEigen.cpp)
#由于eigen库只有头文件，所以只需要把目录拉入检索名单即可，无需连接
include_directories("/usr/include/eigen3")