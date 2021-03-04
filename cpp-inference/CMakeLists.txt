cmake_minimum_required(VERSION 3.16)
project(segmentation C CXX)
set(CMAKE_CXX_STANDARD 11)
find_package( OpenCV REQUIRED )

macro(print_all_variables)
    message(STATUS "print_all_variables------------------------------------------{")
    get_cmake_property(_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
    message(STATUS "print_all_variables------------------------------------------}")
endmacro()



set(TENSORFLOW_SOURCE_DIR "" CACHE PATH
  "Directory that contains the TensorFlow project"
)
if(NOT TENSORFLOW_SOURCE_DIR)
  get_filename_component(TENSORFLOW_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/tensorflow_src"
    ABSOLUTE
  )
endif()


add_subdirectory(
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite"
  EXCLUDE_FROM_ALL
)

include_directories( ${OpenCV_INCLUDE_DIRS} )



add_executable(segmentation
  segmentation.cc
)
target_link_libraries(segmentation
  tensorflow-lite
  ${CMAKE_DL_LIBS}
  ${OpenCV_LIBS}
)
print_all_variables()
