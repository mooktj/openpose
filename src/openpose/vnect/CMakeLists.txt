set(SOURCES_OP_VNECT
    vNect.cpp
    draw3DPython.cpp)

include(${CMAKE_SOURCE_DIR}/cmake/Utils.cmake)
prepend(SOURCES_OP_VNECT_WITH_CP ${CMAKE_CURRENT_SOURCE_DIR} ${SOURCES_OP_VNECT})
set(SOURCES_OP_VNECT_WITH_CP ${SOURCES_OP_VNECT_WITH_CP} PARENT_SCOPE)
set(SOURCES_OPENPOSE ${SOURCES_OPENPOSE} ${SOURCES_OP_VNECT_WITH_CP} PARENT_SCOPE)

if (UNIX OR APPLE)
  if (${GPU_MODE} MATCHES "CUDA")
    cuda_add_library(openpose_vnect ${SOURCES_OP_VNECT})
    set(CMAKE_CXX_FLAGS "-I/usr/include/python2.7 -lpython2.7")
  else ()
    add_library(openpose_vnect ${SOURCES_OP_VNECT})
  endif ()

  add_definitions(${CMAKE_CXX_FLAGS})

##############################################
#  find_package(PythonInterp REQUIRED)
#  find_package(PythonLibs REQUIRED)

#  include_directories(${Boost_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS})
#  link_directories(${Boost_LIBRARY_DIR})

#  add_library(pylib SHARED pylib.cpp)
#  target_link_libraries(pylib ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
#  add_dependencies(mylib Boost)

#  set_target_properties(pylib PROPERTIES SUFFIX .so)
#  set_target_properties(pylib PROPERTIES PREFIX "")
##############################################

# include_directories(-I/usr/include/python2.7 -lpython2.7 -std=c++11)
  add_library(caffe SHARED IMPORTED)
  set_property(TARGET caffe PROPERTY IMPORTED_LOCATION ${Caffe_LIBS}) 
  target_link_libraries(openpose_vnect caffe ${MKL_LIBS} openpose_core ${examples_3rdparty_libraries})
#  target_link_libraries(${CMAKE_CXX_FLAGS})


#add_library(openpose_vnect_c SHARED ${SOURCE_FILES})
#include_directories(${PYTHON_INCLUDE_DIRS})
#target_link_libraries(openpose_vnect_c ${PYTHON_LIBRARIES})

  if (BUILD_CAFFE)
    add_dependencies(openpose_vnect openpose)
  endif (BUILD_CAFFE)

  install(TARGETS openpose_vnect
    EXPORT OpenPose
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/openpose)


endif (UNIX OR APPLE)
