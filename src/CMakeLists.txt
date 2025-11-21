# 包含 interface 和 模块的 include
include_directories(interface)
include_directories(module/include)

# 找到源文件
aux_source_directory(module/source libSRCS)

# 生成动态库
#add_library(${CMAKE_PROJECT_NAME} STATIC ${libSRCS})
add_library(${CMAKE_PROJECT_NAME} SHARED ${libSRCS})

#add_executable(main module/source/main.cpp module/source/rb_cah_detector.cpp module/source/rb_cah_utils.cpp module/source/rb_cah_signature.cpp)

#target_link_libraries(main ${EXTRA_LIBS} -fopenmp -lpthread -lm)

# 包含include 
target_include_directories(${CMAKE_PROJECT_NAME}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/module/include
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/interface>
    $<INSTALL_INTERFACE:include>
)

# 链接库
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${EXTRA_LIBS} -lpthread -lm)

# 安装接口 头文件 和 库
FILE(GLOB INTERFACES interface/*.h)
install(FILES ${INTERFACES} DESTINATION include)
install(TARGETS ${CMAKE_PROJECT_NAME} DESTINATION lib)
message(STATUS "CMAKE_PROJECT_NAME = ${CMAKE_PROJECT_NAME}")
