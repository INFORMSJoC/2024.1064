#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace pwlopt {

struct Point2 {
    double x;
    double y;
};

struct Triangle2 {
    Point2 a;
    Point2 b;
    Point2 c;
};

std::vector<Point2> poisson_disk_sample(
    const Triangle2& triangle,
    double radius,
    double active_darts,
    int max_level,
    std::uint64_t seed
);

}  // namespace pwlopt