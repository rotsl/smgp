#!/usr/bin/env python
"""Setuptools script to compile the Cython-accelerated SMGP extension.

Run from the repository root::

    python setup_speed.py build_ext --inplace

If Cython is not installed this script prints a message and exits
gracefully.
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    try:
        from Cython.Build import cythonize  # type: ignore[import-untyped]
    except ImportError:
        print(
            "Cython is not installed.  Skipping the accelerated extension build.\n"
            "Install it with:  pip install cython\n"
            "The pure-Python fallback in smgp.enhanced.speed.accelerated will be used.",
        )
        sys.exit(0)

    from setuptools import Extension, setup

    import numpy as np  # noqa: E402  (needed after setuptools is available)

    ext = Extension(
        name="smgp.enhanced.speed._accelerated",
        sources=[str(Path("src/smgp/enhanced/speed/_accelerated.pyx"))],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )

    setup(
        name="smgp-speed-ext",
        ext_modules=cythonize(
            [ext],
            compiler_directives={
                "boundscheck": False,
                "wraparound": False,
                "cdivision": True,
                "language_level": "3",
            },
        ),
        script_args=sys.argv[1:],
    )


if __name__ == "__main__":
    main()
