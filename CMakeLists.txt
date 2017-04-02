cmake_minimum_required(VERSION 2.8.4)
set (CMAKE_CXX_STANDARD 11)
PROJECT(facemorph)

# common
find_package(Boost REQUIRED COMPONENTS filesystem)
include_directories(${Boost_INCLUDE_DIRS}) 

# landmarks
find_package( dlib REQUIRED )
include_directories(${dlib_INCLUDE_DIRS})
ADD_EXECUTABLE(face_landmarks_ex face_landmarks_ex.cpp)
TARGET_LINK_LIBRARIES(face_landmarks_ex ${Boost_LIBRARIES} ${dlib_LIBS})

# face-average
find_package( OpenCV 3 REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})
ADD_EXECUTABLE(face_average_ex face_average_ex.cpp)
TARGET_LINK_LIBRARIES(face_average_ex ${Boost_LIBRARIES}  ${OpenCV_LIBS} )



