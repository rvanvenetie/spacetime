cc_binary(
    name = "triangulation_main",
    srcs = ["triangulation_test.cpp"],
    deps = [
        ":triangulation"
        ]
)

cc_library(
    name = "triangulation",
    hdrs = ["tree.hpp",
            "triangulation.hpp"]
)
cc_test(
    name = "hello_world_test",
    srcs = ['hello_world_test.cpp'],
    deps = [
        "@googletest//:gtest_main",
    ],
)
