cmake_minimum_required(VERSION 3.1)

project( CS412_Lab02 )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} ".")

add_executable( harris Harris.cpp Detector.cpp DerivedDetector.cpp HarrisDetector.cpp BlobDetector.cpp DoGDetector.cpp )
add_executable( dog DoG.cpp Detector.cpp DerivedDetector.cpp HarrisDetector.cpp BlobDetector.cpp DoGDetector.cpp )
add_executable( blob Blob.cpp Detector.cpp DerivedDetector.cpp HarrisDetector.cpp BlobDetector.cpp DoGDetector.cpp )
add_executable( test_case test_.cpp)
add_executable( m m.cpp Detector.cpp        Descriptor.cpp
                        DerivedDetector.cpp DerivedDescriptor.cpp
                        HarrisDetector.cpp  BlobDetector.cpp        DoGDetector.cpp
                        SIFTDesExt.cpp      LBPDesExt.cpp
                        Matcher.cpp)

set_property(TARGET harris      PROPERTY CXX_STANDARD 20)
set_property(TARGET dog         PROPERTY CXX_STANDARD 20)
set_property(TARGET blob        PROPERTY CXX_STANDARD 20)
set_property(TARGET m           PROPERTY CXX_STANDARD 20)
set_property(TARGET test_case   PROPERTY CXX_STANDARD 20)

set_property(TARGET harris      PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET dog         PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET blob        PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET m           PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET test_case   PROPERTY CXX_STANDARD_REQUIRED ON)

target_link_libraries( harris       ${OpenCV_LIBS} )
target_link_libraries( dog          ${OpenCV_LIBS} )
target_link_libraries( blob         ${OpenCV_LIBS} )
target_link_libraries( m            ${OpenCV_LIBS} )
target_link_libraries( test_case    ${OpenCV_LIBS} )