from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
from setuptools.dist import Distribution
import os

class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

setup(
  name="quantizer_cpp",
  ext_modules=[
    CppExtension(
      name="quantizer_cpp",
      sources=["quantizer.cpp", "src/quant.cpp"],
      include_dirs=[
        os.path.join(os.path.dirname(__file__), "src"), 
        os.path.join(os.path.dirname(__file__), "vendor")
      ],
      extra_compile_args=["-O3", "-march=native", "-std=c++20", "-fopenmp"],
      extra_link_args=["-fopenmp"],
    ),
  ],
  cmdclass={
    'build_ext': BuildExtension
  },
  python_requires='>=3.8',
  install_requires=[
    'torch>=2.0.0',
  ],
  setup_requires=[
    'torch>=2.0.0',
    'ninja',
    'numpy',
  ],
  distclass=BinaryDistribution,
)