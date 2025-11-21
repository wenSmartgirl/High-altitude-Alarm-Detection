
# 只有安装了opencv才去生成一些工具
include_directories(../src/interface)

add_executable(cah_example_image rb_cah_example_image.cpp rb_cah_example_utils.cpp)

target_link_libraries(cah_example_image ${CMAKE_PROJECT_NAME} ${PCL_LIBRARIES}  ${EXTRA_LIBS} point_gen -fopenmp -lpthread -lm)

install (TARGETS cah_example_image RUNTIME DESTINATION examples)

FILE(GLOB IMS ../assets/images/*.*)
install(FILES ${IMS} DESTINATION examples)

# 临时测试时使用
install(FILES ${IMS} DESTINATION ${PROJECT_SOURCE_DIR}/temp_example)
install (TARGETS cah_example_image RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/temp_example)
#install (TARGETS main RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/temp_example)

if (WITH_NCNN)
    install(FILES "../assets/models/cah.ncnn.param" DESTINATION ${PROJECT_SOURCE_DIR}/temp_example/models)
    install(FILES "../assets/models/cah.ncnn.bin" DESTINATION ${PROJECT_SOURCE_DIR}/temp_example/models)
elseif (WITH_RKNN)
    install(FILES "../assets/models/cah.rknn" DESTINATION ${PROJECT_SOURCE_DIR}/temp_example/models)
    install(PROGRAMS ${RKNN_RT_LIB} DESTINATION ${PROJECT_SOURCE_DIR}/temp_example/lib)
    install(PROGRAMS ${RGA_LIB} DESTINATION ${PROJECT_SOURCE_DIR}/temp_example/lib)

endif()

