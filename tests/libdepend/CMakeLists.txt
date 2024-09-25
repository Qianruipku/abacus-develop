# parameter: independent
add_executable(parameter_libdepend_test main.cpp)
target_link_libraries(parameter_libdepend_test parameter)

# base:
#   - device
#   - parameter 
add_executable(base_libdepend_test main.cpp)
target_link_libraries(base_libdepend_test base device parameter ${math_libs})

# device:
#   - base
#   - parameter
add_executable(device_libdepend_test main.cpp)
target_link_libraries(device_libdepend_test device base parameter ${math_libs})