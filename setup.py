from pathlib import Path

import numpy as np
from setuptools import Extension, setup


ext_module = Extension(
    "fast_bocpd._core",
    sources=[
        "fast_bocpd/_c/bocpd_core.c",
        "fast_bocpd/_c/hazard.c",
        "fast_bocpd/_c/gaussian_nig.c",
        "fast_bocpd/_c/student_t_ng.c",
        "fast_bocpd/_c/student_t_ng_grid.c",
        "fast_bocpd/_c/poisson_gamma.c",
        "fast_bocpd/_c/gamma_gamma_fixed_shape.c",
        "fast_bocpd/_c/bernoulli_beta.c",
        "fast_bocpd/_c/binomial_beta.c",
    ],
    include_dirs=[np.get_include()],
    extra_compile_args=[
        "-std=c99",
        "-O3",
        "-march=native",
        "-fomit-frame-pointer",
        "-Wall",
        "-Wextra",
        "-fPIC",
    ],
    extra_link_args=["-lm"],
)

# Metadata is now in pyproject.toml - setup.py only handles C extension
setup(
    ext_modules=[ext_module],
    include_package_data=True,
    package_data={"fast_bocpd": ["_c/*.c", "_c/*.h"]},
    zip_safe=False,
)
