include(GoogleTest)

file(GLOB_RECURSE BGCLIB_TEST_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

list(FILTER BGCLIB_TEST_SOURCES EXCLUDE REGEX "/TSOM/")

add_executable(bgclib_core_tests
    ${BGCLIB_TEST_SOURCES}
)

target_link_libraries(bgclib_core_tests
    PRIVATE
        Anomalous::bgclib
        GTest::gtest_main
)

gtest_discover_tests(bgclib_core_tests)
