cc_library(
    name = "triangulation",
    srcs = ["triangulation.cpp"],
    hdrs = [ "triangulation.hpp"],
    deps = [
            "//datastructures:tree"]
)
cc_test(
    name = "triangulation_test",
    srcs = ['triangulation_test.cpp'],
    deps = [
         ":triangulation",
        "@googletest//:gtest_main",
    ],
)
