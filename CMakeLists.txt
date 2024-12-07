project (current)

set(SOURCES main.cpp stream.cpp)

add_executable(current ${SOURCES})
target_link_libraries(current PRIVATE sample_lib)