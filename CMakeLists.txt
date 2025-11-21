cmake_minimum_required (VERSION 3.10)

# 获取git的tag版本号
execute_process(COMMAND git describe --tags 
    TIMEOUT 5
    OUTPUT_VARIABLE GIT_TAG_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# 获得 md5 值
execute_process(COMMAND git describe --always 
    TIMEOUT 5
    OUTPUT_VARIABLE GIT_COMMIT_MD5
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "building from git tag ${GIT_TAG_VERSION}")
add_definitions(-DBUILD_TAG_VERSION=\"${GIT_TAG_VERSION}\")

# 设置项目名称和版本号
project(CAH)

# 获取当前时间
string(TIMESTAMP BUILDDATETIME "%Y-%m-%d %H:%M:%S")


# 设置安装目录
set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/install")
set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# 要求编译器支持C++11
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)


if(NOT WIN32)
    set(CMAKE_C_FLAGS_RELEASE    "${CMAKE_C_FLAGS_RELEASE} -s")
    set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -s")
#     set(CMAKE_C_FLAGS_RELEASE    "${CMAKE_C_FLAGS_RELEASE} -fPIC")
#     set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -fPIC")
endif()

# 送入网络图像大小
set(TGT_IM_SZ  224)

# 目标阈值
set(PROB_THR  0.25)

# NMS 阈值
set(NMS_THR   0.25)

# TODO: 类别数, rknn 需要, 根据情况进行修改 
set(OBJ_CLASS_NUM 4)
# 最多目标数
set(OBJ_NUMB_MAX_SIZE 64)

######################### 设置内部参数部分 ################################
# TODO: 模型前缀 model prefix, 每次训练导出模型时请记得修改  ***************
set(MODEL_PREFIX   cah)

# 参数设置
set(BUFFER_LEN 5)

# 内部收缩像素数
set(INNER_BORDER  5)

# 外部扩展像素数
set(OUTTER_BORDER 5)

# 设置目标与ROI重叠阈值(0~1)
set(OVERLAP_THR  0.1)

# 设置低灵敏度的倍数(灵敏度越低，重叠阈值越大 1~2)
set(LOW_SENS_RATIO  1.3)

# 设置高灵敏度的倍数(灵敏度越高，重叠阈值越小, 0~1)
set(HIGH_SENES_RATIO 0.8)
##########################################################################



# 开启相关选项
option (WITH_OPENMP     "Use OpenMP"                                                             ON )
option (WITH_NCNN       "Use ncnn inference "                                                    OFF )
option (WITH_RKNN       "Use rknn npu inference"                                                 ON )
option (LOAD_MEM_MODEL  "Build lib, with load network from memory, else load from file"          OFF )
option (BUILD_DEBUG     "Build Debug, will print or save some temp message"                      OFF)
option (BUILD_EXAMPLES  "Build examples"                                                         ON)

# 使用openmp
if (WITH_OPENMP)
    add_definitions(-fopenmp)
endif(WITH_OPENMP)

# 编译类型
if (BUILD_DEBUG)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build types: Release Debug" FORCE)
    option (SHOW_DEBUG_IMG  "Show image inner the api"       ON )   # 显示调试图像
    option (SHOW_DEBUG_TIME "Show process time in the api"   ON )   # 显示内部时间
    option (SHOW_DEBUG_INFO "Show debug info in the api"     ON )   # 显示内部打印
else()
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build types: Release Debug" FORCE)
    option (SHOW_DEBUG_IMG  "Show image inner the api"     OFF )   # 不显示调试图像
    option (SHOW_DEBUG_TIME "Show process time in the api" ON )   # 不显示内部时间
    option (SHOW_DEBUG_INFO "Show debug info in the api"   OFF )   # 不显示内部打印
endif()



# OpenCV路径设置
set(OpenCV_DIR "/usr")
set(OpenCV_INCLUDE_DIRS "${OpenCV_DIR}/include/opencv4")
set(OpenCV_LIB_DIRS "${OpenCV_DIR}/lib/aarch64-linux-gnu")

# set(OpenCV_INCLUDE_DIRS "/usr/local/include/opencv4")
# set(OpenCV_LIB_DIRS "/usr/local/lib")
set(OpenCV_3RD_DIRS "${OpenCV_DIR}/3rdparty/lib")
include_directories("${OpenCV_INCLUDE_DIRS}")
link_directories("${OpenCV_LIB_DIRS}")
link_directories("${OpenCV_3RD_DIRS}")
list(APPEND EXTRA_LIBS opencv_imgproc opencv_core opencv_highgui opencv_videoio opencv_imgcodecs opencv_calib3d -fopenmp)


set(depthai_DIR "/home/firefly/depthai-2-29-0/usr/local")
set(depthai_INCLUDE_DIRS "${depthai_DIR}/include" "${depthai_DIR}/include/depthai" "${depthai_DIR}/lib/cmake/depthai/dependencies/include"  "${depthai_DIR}/include/depthai-shared/3rdparty")
set(depthai_LIB_DIRS "${depthai_DIR}/lib" "${depthai_DIR}/lib/cmake/depthai/dependencies/lib")

include_directories("${depthai_INCLUDE_DIRS}")
link_directories("${depthai_LIB_DIRS}")
#link_directories("${depthai_3RD_DIRS}")
#list(APPEND EXTRA_LIBS opencv_imgproc opencv_core opencv_highgui opencv_videoio opencv_imgcodecs -fopenmp)
list(APPEND EXTRA_LIBS depthai-core depthai-opencv)





# 设置使用 NCNN
if (WITH_NCNN)
    option (WITH_RKNN       "Use rknn npu inference"   OFF )
    # ncnn路径设置
    set(NCNN_DIR "/home/ztl/app/down/ncnn/build/install") 
    set(NCNN_INCLUDE_DIR "${NCNN_DIR}/include/ncnn")
    set(NCNN_LIB_DIR "${NCNN_DIR}/lib")
    include_directories("${NCNN_INCLUDE_DIR}")
    link_directories("${NCNN_LIB_DIR}")
    list(APPEND EXTRA_LIBS ncnn)
    
    if (LOAD_MEM_MODEL)
        message(STATUS "USE NCNN, aand Load memory model!")
    else()
        # 当不用内存方式时, 安装模型文件
        set(NCNN_PARAM_PATH "models/cah.ncnn.param")
        set(NCNN_BIN_PATH   "models/cah.ncnn.bin")
        install(FILES
            "assets/models/cah.ncnn.bin"
            "assets/models/cah.ncnn.param"
            DESTINATION assets/models)
    endif()
elseif(WITH_RKNN)
    option (WITH_NCNN       "Use ncnn inference "      OFF )
    include_directories(${CMAKE_SOURCE_DIR}/../3rdparty)

    # rknn api runtime
    set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/../3rdparty/runtime)
    set(RKNN_RT_LIB ${RKNN_API_PATH}/lib/librknnrt.so)
    include_directories(${RKNN_API_PATH}/include)
    list(APPEND EXTRA_LIBS ${RKNN_RT_LIB})

    # rga
    set(RGA_PATH ${CMAKE_SOURCE_DIR}/../3rdparty/rga)
    set(RGA_LIB ${RGA_PATH}/lib/librga.so)
    include_directories(${RGA_PATH}/include)
    list(APPEND EXTRA_LIBS ${RGA_LIB})

    # depthai   libdepthai-opencv.so
    set(DEPTHAI_PATH ${depthai_DIR}/lib)
    set(depthai_LIB ${DEPTHAI_PATH}/libdepthai-core.so ${DEPTHAI_PATH}/libdepthai-opencv.so ${DEPTHAI_PATH}/libpoint_gen.so)

    set(MODEL_NAME "models/cah.rknn")
    install(FILES
        "assets/models/cah.rknn"
        DESTINATION models
    )

    install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
    install(PROGRAMS ${RGA_LIB}  DESTINATION lib)
    install(PROGRAMS ${depthai_LIB}  DESTINATION lib)

endif()


find_package(PCL)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


set(Eigen3_INCLUDE_DIRS "/usr/include/eigen3")
include_directories("${Eigen3_INCLUDE_DIRS}")
set(Vtk_INCLUDE_DIRS "/usr/include/vtk-7.1")
include_directories("${Vtk_INCLUDE_DIRS}")
set(Vtk_LIB_DIRS "/usr/lib/aarch64-linux-gnu")
link_directories("${Vtk_LIB_DIRS}")

# 查找VTK库
find_package(VTK COMPONENTS vtkCommonCore vtkFiltersCore vtkIOImage vtkRenderingCore vtkRenderingOpenGL2 REQUIRED)


# windows
if(MSVC)
    add_definitions(/utf-8)
    add_definitions(/MT)
    # add_definitions(/MDd)
    add_definitions(/MD)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-DSI_API_EXPORTS)
endif()

# 配置文件到实例目录
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/rb_cah_gconfig.h.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/examples/rb_cah_gconfig.h"
)

# 配置文件到 module 内部
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/rb_cah_gconfig.h.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/module/include/rb_cah_gconfig.h"
)

# 生成签名信息串的cpp源文件
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/rb_cah_signature.cpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/rb_cah_signature.cpp"
)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/rb_cah_signature.cpp.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/module/source/rb_cah_signature.cpp"
)

# 包含头文件
include_directories("${CMAKE_BINARY_DIR}")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src/interface")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src/module/include")

# 添加源代码目录进行编译
add_subdirectory(src)

# 添加用例
if (BUILD_EXAMPLES)
    add_subdirectory(examples)
endif(BUILD_EXAMPLES)


# 调试信息
message(STATUS "---------------- message start -------------------------------")
message(STATUS "Tags: ${GIT_TAG_VERSION}, Commits md5: ${GIT_COMMIT_MD5}")
message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
message(STATUS "Building at ${BUILDDATETIME}")
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
message(STATUS "BUILD_SHARED_LIBRARY = ${BUILD_SHARED_LIBRARY}")
message(STATUS "OpenCV Path = ${OpenCV_DIR}")
message(STATUS "NCNN Path = ${NCNN_DIR}")
message(STATUS "RGA Path = ${RGA_PATH}")
message(STATUS "---------------- message end -------------------------------")

# 打包库, 可以用make package
include (InstallRequiredSystemLibraries)
set(CPACK_RESOURCE_FILE_LICENSE
    "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_VERSION  ${GIT_TAG_VERSION})
set(CPACK_GENERATOR "TGZ")

