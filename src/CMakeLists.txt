cmake_minimum_required(VERSION 3.16..3.21)
set(APP_NAME SmileMouthDetection)
project(${APP_NAME})

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_OSX_SYSROOT /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk)
set(CMAKE_CXX_FLAGS "-isysroot ${CMAKE_OSX_SYSROOT} -I/usr/include -I${CMAKE_OSX_SYSROOT}/usr/include")

# Make executable app.
add_executable(${APP_NAME})

# Find opencv libraries
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Specify where TensorFlow Lite headers are located
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_library(tensorflowlite SHARED IMPORTED)
set_property(TARGET tensorflowlite PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/libs/mac/libtensorflowlite.dylib)

# Add include path
target_include_directories(${APP_NAME} 
    PRIVATE ${OpenCV_INCLUDE_DIRS} 
    PRIVATE ${TFLite_INCLUDE_DIRS})

# Link libraries to app.
target_link_libraries(${APP_NAME} 
    PRIVATE ${OpenCV_LIBS} 
    PRIVATE ${TFLite_LIBS}
    tensorflowlite
)

## Build in multi-process.
target_compile_options(${APP_NAME}
    PRIVATE)

target_sources(${APP_NAME}
        PRIVATE
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