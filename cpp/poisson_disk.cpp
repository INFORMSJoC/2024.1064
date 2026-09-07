#include "poisson_disk.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <unordered_map>
#include <vector>

namespace pwlopt {

namespace {

struct Cell {
    int i;
    int j;
};

inline double squared_distance(
    const Point2& a,
    const Point2& b
) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return dx * dx + dy * dy;
}

inline double cross(
    const Point2& a,
    const Point2& b,
    const Point2& p
) {
    return
        (b.x - a.x) * (p.y - a.y)
        - (b.y - a.y) * (p.x - a.x);
}


// ------------------------------------------------------------
// Point in triangle
// ------------------------------------------------------------

inline bool contains(
    const Triangle2& t,
    const Point2& p
) {
    const double c1 = cross(t.a, t.b, p);
    const double c2 = cross(t.b, t.c, p);
    const double c3 = cross(t.c, t.a, p);

    constexpr double eps = 1e-14;

    const bool has_negative =
        (c1 < -eps) ||
        (c2 < -eps) ||
        (c3 < -eps);

    const bool has_positive =
        (c1 > eps) ||
        (c2 > eps) ||
        (c3 > eps);

    return !(has_negative && has_positive);
}


// ------------------------------------------------------------
// Segment intersection
// ------------------------------------------------------------

inline bool segments_intersect(
    const Point2& a,
    const Point2& b,
    const Point2& c,
    const Point2& d
) {
    const double c1 = cross(a, b, c);
    const double c2 = cross(a, b, d);
    const double c3 = cross(c, d, a);
    const double c4 = cross(c, d, b);

    constexpr double eps = 1e-14;

    if (
        ((c1 > eps && c2 < -eps) ||
         (c1 < -eps && c2 > eps)) &&
        ((c3 > eps && c4 < -eps) ||
         (c3 < -eps && c4 > eps))
    ) {
        return true;
    }

    return false;
}


// ------------------------------------------------------------
// Triangle / axis-aligned rectangle intersection
//
// This replaces triangle._cell_intersects().
// ------------------------------------------------------------

inline bool triangle_cell_intersects(
    const Triangle2& t,
    double xmin,
    double ymin,
    double xmax,
    double ymax
) {
    // 1. Any triangle vertex inside the rectangle?
    auto inside_rect =
        [xmin, ymin, xmax, ymax](const Point2& p) {
            return
                p.x >= xmin &&
                p.x <= xmax &&
                p.y >= ymin &&
                p.y <= ymax;
        };

    if (inside_rect(t.a) ||
        inside_rect(t.b) ||
        inside_rect(t.c)) {
        return true;
    }

    // 2. Any rectangle corner inside triangle?
    const Point2 corners[4] = {
        {xmin, ymin},
        {xmax, ymin},
        {xmin, ymax},
        {xmax, ymax}
    };

    for (const auto& p : corners) {
        if (contains(t, p)) {
            return true;
        }
    }

    // 3. Any triangle edge intersects rectangle edge?
    const Point2 rect_edges[4][2] = {
        {{xmin, ymin}, {xmax, ymin}},
        {{xmax, ymin}, {xmax, ymax}},
        {{xmax, ymax}, {xmin, ymax}},
        {{xmin, ymax}, {xmin, ymin}}
    };

    const Point2 tri_edges[3][2] = {
        {t.a, t.b},
        {t.b, t.c},
        {t.c, t.a}
    };

    for (const auto& te : tri_edges) {
        for (const auto& re : rect_edges) {
            if (segments_intersect(
                    te[0], te[1],
                    re[0], re[1])) {
                return true;
            }
        }
    }

    return false;
}


// ------------------------------------------------------------
// Flattened base-grid key
// ------------------------------------------------------------

inline std::int64_t grid_key(
    int i,
    int j,
    int ny
) {
    return
        static_cast<std::int64_t>(i) *
            static_cast<std::int64_t>(ny)
        + j;
}

}  // namespace


// ============================================================================
// Poisson disk sampler
// ============================================================================

std::vector<Point2> poisson_disk_sample(
    const Triangle2& triangle,
    double radius,
    double active_darts,
    int max_level,
    std::uint64_t seed
) {
    if (radius <= 0.0) {
        throw std::invalid_argument(
            "radius must be positive"
        );
    }

    if (active_darts <= 0.0) {
        throw std::invalid_argument(
            "active_darts must be positive"
        );
    }

    if (max_level < 0) {
        throw std::invalid_argument(
            "max_level must be non-negative"
        );
    }

    // ------------------------------------------------------------
    // Bounding box
    // ------------------------------------------------------------

    const double lo_x =
        std::min({
            triangle.a.x,
            triangle.b.x,
            triangle.c.x
        });

    const double lo_y =
        std::min({
            triangle.a.y,
            triangle.b.y,
            triangle.c.y
        });

    const double hi_x =
        std::max({
            triangle.a.x,
            triangle.b.x,
            triangle.c.x
        });

    const double hi_y =
        std::max({
            triangle.a.y,
            triangle.b.y,
            triangle.c.y
        });

    // Ebeida base-cell diagonal = radius.
    const double s0 =
        radius / std::sqrt(2.0);

    const int nx =
        std::max(
            1,
            static_cast<int>(
                std::ceil((hi_x - lo_x) / s0)
            )
        );

    const int ny =
        std::max(
            1,
            static_cast<int>(
                std::ceil((hi_y - lo_y) / s0)
            )
        );

    const double radius2 =
        radius * radius;

    // ------------------------------------------------------------
    // Spatial grid
    //
    // base cell -> sample index
    // ------------------------------------------------------------

    std::unordered_map<std::int64_t, std::size_t> grid;

    // We use separate vectors rather than vector<Point2> mainly
    // to make the storage simple and contiguous.
    std::vector<Point2> samples;

    // ------------------------------------------------------------
    // Random number generator
    // ------------------------------------------------------------

    std::mt19937_64 rng(seed);

    // ------------------------------------------------------------
    // Initial active cells
    // ------------------------------------------------------------

    std::vector<std::int64_t> active;

    active.reserve(
        static_cast<std::size_t>(nx) *
        static_cast<std::size_t>(ny)
    );

    for (int i = 0; i < nx; ++i) {
        const double x0 =
            lo_x + static_cast<double>(i) * s0;

        const double x1 = x0 + s0;

        for (int j = 0; j < ny; ++j) {
            const double y0 =
                lo_y + static_cast<double>(j) * s0;

            const double y1 = y0 + s0;

            if (triangle_cell_intersects(
                    triangle,
                    x0,
                    y0,
                    x1,
                    y1
                )) {
                active.push_back(
                    grid_key(i, j, ny)
                );
            }
        }
    }

    // ------------------------------------------------------------
    // Sampling
    // ------------------------------------------------------------

    int level = 0;

    while (!active.empty()) {

        const std::size_t n_active =
            active.size();

        const std::size_t n_darts =
            std::max<std::size_t>(
                1,
                static_cast<std::size_t>(
                    std::ceil(
                        active_darts *
                        static_cast<double>(n_active)
                    )
                )
            );

        // --------------------------------------------------------
        // Darts
        // --------------------------------------------------------

        for (std::size_t dart = 0;
             dart < n_darts;
             ++dart) {

            if (active.empty()) {
                break;
            }

            // Uniform random active-cell index.
            std::uniform_int_distribution<std::size_t>
                cell_distribution(
                    0,
                    active.size() - 1
                );

            const std::size_t pos =
                cell_distribution(rng);

            const std::int64_t cell =
                active[pos];

            // At level l, cells per base-grid row:
            const int stride =
                ny * (1 << level);

            const int i =
                static_cast<int>(
                    cell / stride
                );

            const int j =
                static_cast<int>(
                    cell % stride
                );

            // ----------------------------------------------------
            // Check whether base cell already contains a sample.
            // ----------------------------------------------------

            const int factor =
                1 << level;

            const int base_i =
                i / factor;

            const int base_j =
                j / factor;

            const std::int64_t base_key =
                grid_key(
                    base_i,
                    base_j,
                    ny
                );

            if (grid.find(base_key) != grid.end()) {
                active[pos] = active.back();
                active.pop_back();
                continue;
            }

            // ----------------------------------------------------
            // Current cell bounds
            // ----------------------------------------------------

            const double side =
                s0 / static_cast<double>(factor);

            const double x0 =
                lo_x +
                static_cast<double>(i) * side;

            const double y0 =
                lo_y +
                static_cast<double>(j) * side;

            std::uniform_real_distribution<double>
                x_distribution(
                    x0,
                    x0 + side
                );

            std::uniform_real_distribution<double>
                y_distribution(
                    y0,
                    y0 + side
                );

            const Point2 p{
                x_distribution(rng),
                y_distribution(rng)
            };

            // ----------------------------------------------------
            // Triangle containment
            // ----------------------------------------------------

            if (!contains(triangle, p)) {
                continue;
            }

            // ----------------------------------------------------
            // Disk test
            // ----------------------------------------------------

            const int pi =
                static_cast<int>(
                    std::floor(
                        (p.x - lo_x) / s0
                    )
                );

            const int pj =
                static_cast<int>(
                    std::floor(
                        (p.y - lo_y) / s0
                    )
                );

            bool free = true;

            // The original implementation checks [-2, 2].
            for (int di = -2; di <= 2 && free; ++di) {
                const int ii = pi + di;

                if (ii < 0 || ii >= nx) {
                    continue;
                }

                for (int dj = -2; dj <= 2; ++dj) {
                    const int jj = pj + dj;

                    if (jj < 0 || jj >= ny) {
                        continue;
                    }

                    const auto it =
                        grid.find(
                            grid_key(ii, jj, ny)
                        );

                    if (it == grid.end()) {
                        continue;
                    }

                    const Point2& q =
                        samples[it->second];

                    const double dx =
                        p.x - q.x;

                    const double dy =
                        p.y - q.y;

                    if (
                        dx * dx +
                        dy * dy <
                        radius2
                    ) {
                        free = false;
                        break;
                    }
                }
            }

            if (!free) {
                continue;
            }

            // ----------------------------------------------------
            // Accept point
            // ----------------------------------------------------

            const std::size_t sample_index =
                samples.size();

            samples.push_back(p);

            const std::int64_t key =
                grid_key(
                    base_i,
                    base_j,
                    ny
                );

            if (grid.find(key) != grid.end()) {
                throw std::runtime_error(
                    "Base grid contains two samples."
                );
            }

            grid.emplace(
                key,
                sample_index
            );

            // Remove active cell.
            active[pos] = active.back();
            active.pop_back();
        }

        // --------------------------------------------------------
        // Stop
        // --------------------------------------------------------

        if (
            active.empty() ||
            level >= max_level
        ) {
            break;
        }

        // --------------------------------------------------------
        // Refine
        // --------------------------------------------------------

        const int next_level =
            level + 1;

        const int old_stride =
            ny * (1 << level);

        const int new_stride =
            old_stride * 2;

        const int factor =
            1 << next_level;

        const double side =
            s0 / static_cast<double>(factor);

        std::vector<std::int64_t> new_active;

        new_active.reserve(
            active.size() * 2
        );

        for (const auto cell : active) {

            const int i =
                static_cast<int>(
                    cell / old_stride
                );

            const int j =
                static_cast<int>(
                    cell % old_stride
                );

            const int ii =
                2 * i;

            const int jj =
                2 * j;

            const std::int64_t children[4] = {
                static_cast<std::int64_t>(ii)
                    * new_stride + jj,

                static_cast<std::int64_t>(ii + 1)
                    * new_stride + jj,

                static_cast<std::int64_t>(ii)
                    * new_stride + (jj + 1),

                static_cast<std::int64_t>(ii + 1)
                    * new_stride + (jj + 1)
            };

            for (const auto child : children) {

                const int ci =
                    static_cast<int>(
                        child / new_stride
                    );

                const int cj =
                    static_cast<int>(
                        child % new_stride
                    );

                const double x0 =
                    lo_x +
                    static_cast<double>(ci) * side;

                const double y0 =
                    lo_y +
                    static_cast<double>(cj) * side;

                const double x1 =
                    x0 + side;

                const double y1 =
                    y0 + side;

                if (!triangle_cell_intersects(
                        triangle,
                        x0,
                        y0,
                        x1,
                        y1
                    )) {
                    continue;
                }

                // ------------------------------------------------
                // Cell coverage test.
                //
                // For a rectangle and point q, the farthest
                // corner has:
                //
                // max(|qx-x0|, |qx-x1|)
                // max(|qy-y0|, |qy-y1|)
                // ------------------------------------------------

                const double cx =
                    0.5 * (x0 + x1);

                const double cy =
                    0.5 * (y0 + y1);

                const int base_ci =
                    static_cast<int>(
                        std::floor(
                            (cx - lo_x) / s0
                        )
                    );

                const int base_cj =
                    static_cast<int>(
                        std::floor(
                            (cy - lo_y) / s0
                        )
                    );

                bool covered = false;

                for (
                    int di = -2;
                    di <= 2 && !covered;
                    ++di
                ) {
                    const int gi =
                        base_ci + di;

                    if (gi < 0 || gi >= nx) {
                        continue;
                    }

                    for (int dj = -2; dj <= 2; ++dj) {

                        const int gj =
                            base_cj + dj;

                        if (
                            gj < 0 ||
                            gj >= ny
                        ) {
                            continue;
                        }

                        const auto it =
                            grid.find(
                                grid_key(
                                    gi,
                                    gj,
                                    ny
                                )
                            );

                        if (it == grid.end()) {
                            continue;
                        }

                        const Point2& q =
                            samples[it->second];

                        const double dx =
                            std::max(
                                std::abs(q.x - x0),
                                std::abs(q.x - x1)
                            );

                        const double dy =
                            std::max(
                                std::abs(q.y - y0),
                                std::abs(q.y - y1)
                            );

                        if (
                            dx * dx +
                            dy * dy <=
                            radius2
                        ) {
                            covered = true;
                            break;
                        }
                    }
                }

                if (!covered) {
                    new_active.push_back(child);
                }
            }
        }

        active.swap(new_active);
        level = next_level;
    }

    return samples;
}

}  // namespace pwlopt