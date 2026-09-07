from numpy import iinfo, ndarray, uint64
from numpy.random import Generator, default_rng

from ._poisson_disk import sample as cpp_sample


class PoissonDiskSampler:

    def __init__(self, triangle: object, radius: float):
        if radius <= 0:
            raise ValueError("radius must be positive")

        self.radius = radius
        self.triangle = triangle

    def sample(
        self,
        active_darts: float = 0.5,
        max_level: int = 10,
        rng: Generator | None = None,
    ) -> ndarray:

        if rng is None:
            rng = default_rng()

        # Generate a seed from NumPy's Generator.
        seed = int(
            rng.integers(
                0,
                iinfo(uint64).max,
                dtype=uint64,
            )
        )

        return cpp_sample(
            self.triangle.P,
            self.radius,
            active_darts,
            max_level,
            seed,
        )