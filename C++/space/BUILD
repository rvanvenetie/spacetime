cc_library(
    name = "triangulation",
    srcs = ["triangulation.cpp"],
    hdrs = [ "triangulation.hpp"],
    deps = [
            "//datastructures:datastructures"]
)
cc_test(
    name = "triangulation_test",
    srcs = ['triangulation_test.cpp'],
    deps = [
         ":triangulation",
        "@googletest//:gtest_main",
    ],
)
cc_binary(
    name = "triangulation_view",
    srcs = ['triangulation_view.cpp'],
    deps = [
         ":triangulation",
    ],
)
