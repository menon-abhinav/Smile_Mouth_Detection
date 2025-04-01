cmake_minimum_required(VERSION 3.16..3.21)
set(APP_NAME SmileMouthDetection)
project(${APP_NAME})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Detect the OS and set the TensorFlow Lite library path
if(APPLE)
    message(STATUS "Compiling for macOS")
    set(CMAKE_OSX_SYSROOT /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk)
    set(CMAKE_CXX_FLAGS "-isysroot ${CMAKE_OSX_SYSROOT} -I/usr/include -I${CMAKE_OSX_SYSROOT}/usr/include")
    set(TFLITE_LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/libs/mac/libtensorflowlite.dylib)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/libs/mac)

elseif(UNIX AND NOT APPLE)
    message(STATUS "Compiling for Linux")
    set(TFLite_PATH "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH
        "Directory that contains the TensorFlow-Lite include and lib")
    set(TFLite_INCLUDE_DIRS "${TFLite_PATH}/include")
    set(TFLite_LIBS "${TFLite_PATH}/lib/linux/libtflite.so")
    include_directories(${TFLite_INCLUDE_DIRS})
    link_directories(${TFLite_PATH}/lib/linux)
endif()

add_executable(${APP_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/demo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ModelLoader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ModelLoader.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DetectionPostProcess.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DetectionPostProcess.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/FaceLandmark.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/FaceLandmark.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/FaceDetection.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/FaceDetection.hpp
)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

if(APPLE)
    target_link_libraries(${APP_NAME} PRIVATE
        ${OpenCV_LIBS}
        ${TFLITE_LIB_PATH}
    )
elseif(UNIX AND NOT APPLE)
    target_link_libraries(${APP_NAME} PRIVATE
        ${OpenCV_LIBS}
        ${TFLite_LIBS}  
        pthread         
        dl            
        m              
    )
endif()