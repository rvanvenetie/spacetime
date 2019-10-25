cc_library(
    name = "triangulation",
    srcs = ["triangulation.cpp"],
    hdrs = ["tree.hpp",
            "triangulation.hpp"]
)
cc_test(
    name = "triangulation_test",
    srcs = ['triangulation_test.cpp'],
    deps = [
         ":triangulation",
        "@googletest//:gtest_main",
    ],
)
