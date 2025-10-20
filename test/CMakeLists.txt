
file(GLOB_RECURSE TEST_FILES ./*.cpp)

set(TestsName "run_test")

add_executable(${TestsName} ${TEST_FILES})

#target_link_libraries(${TestsName} PRIVATE ${ProjectName} gtest)
target_link_libraries(${TestsName} PRIVATE ${ProjectName} GTest::gtest GTest::gtest_main)

enable_testing()
add_test(NAME ${TestsName} COMMAND ${TestsName}) 