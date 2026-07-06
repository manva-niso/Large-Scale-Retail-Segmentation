from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "retail_cpp",
        ["cpp/bindings.cpp", "cpp/rfm_aggregate.cpp", "cpp/kmeans_core.cpp"],
        extra_compile_args=["-O3", "-std=c++17", "-pthread"],
        extra_link_args=["-pthread"],
    ),
]

setup(
    name="retail_cpp",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
