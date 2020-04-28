#include "operators.hpp"
#ifndef EIGEN_NO_DEBUG
#define COMPILE_WITH_EIGEN_DEBUG
#define EIGEN_NO_DEBUG
#endif

namespace space {}  // namespace space

#ifdef COMPILE_WITH_EIGEN_DEBUG
#undef EIGEN_NO_DEBUG
#undef COMPILE_WITH_EIGEN_DEBUG
#endif
