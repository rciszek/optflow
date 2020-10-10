#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import subprocess
import numpy

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext


def find_path(file_regexp):
    # Search for a path containing the file matching 'file_regexp' under /usr/
    with subprocess.Popen(
        ("find", "/usr/", "-name", file_regexp.split("/")[-1]),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    ) as ps:
        try:
            grep_out = subprocess.check_output(
                ["grep", "-P", "-o", "-m", "1", "^.*(?={})".format(file_regexp)],
                stdin=ps.stdout,
            )
            ps.wait()
            return grep_out.decode().strip()
        except:
            raise EnvironmentError(
                "Could not locate {} under /usr/".format(file_regexp)
            )


def list_opencv_libs(lib_path):

    # Construct a valid list of opencv libs
    with subprocess.Popen(
        ("find", lib_path), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    ) as ps:
        try:
            grep_out = subprocess.check_output(
                ["grep", "-P", "-o", "(?<=/lib)(opencv.*)(?=.so)"], stdin=ps.stdout
            )
            ps.wait()
            return ["-l" + l for l in grep_out.decode().splitlines()]
        except:
            raise EnvironmentError(
                "Could not locate opencv libs under {}".format(lib_path)
            )


def locate_cuda():
    # first check if the CUDAHOME env variable is in use
    if "CUDAHOME" in os.environ:
        home = os.environ["CUDAHOME"]
        nvcc = os.path.join(home, "bin", "nvcc")
        cuda_incs = os.path.join(home, "include")
        cuda_libs = os.path.join(home, "lib64")

        if not os.path.exists(nvcc):
            raise EnvironmentError(
                "The CUDA path could not be located. CUDAHOME not properly set."
            )
    else:
        # otherwise, search for NVCC
        try:
            nvcc = subprocess.check_output("which nvcc".split()).decode()
        except subprocess.CalledProcessError:
            raise EnvironmentError(
                "The CUDA path could not be located. Try to set CUDAHOME."
            )

        home = os.path.abspath(os.path.dirname(nvcc) + "/..")

        cuda_incs = os.path.abspath(find_path("cuda.h"))
        cuda_libs = os.path.abspath(find_path("libcudart.so"))

    cudaconfig = {"home": home, "nvcc": nvcc, "include": cuda_incs, "lib": cuda_libs}

    return cudaconfig


def check_gcc():
    """User may define a specific gcc version if e.g system 
    gcc version exceeds the maximum supported by CUDA"""
    if "CC" in os.environ:
        return os.environ["CC"]
    else:
        try:
            # If CC is not set, use system gcc.
            return subprocess.check_output("which gcc".split()).decode().strip()
        except subprocess.CalledProcessError:
            raise EnvironmentError("The gcc path could not be located.")


cuda = locate_cuda()
cuda_gcc = check_gcc()


def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.
    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""

    # tell the compiler it can processes .cu
    self.src_extensions.append(".cu")

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):

        if os.path.splitext(src)[1] == ".cu":
            # use the cuda for .cu files
            self.set_executable("compiler_so", str(cuda["nvcc"]))
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs["nvcc"]

        else:
            postargs = extra_postargs["gcc"]

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


# Find opencv include dir, opencv lib path, and list opencv libs
opencv_incs = [find_path("opencv2/core/core.hpp")]
if "OPENCV_LIBS" in os.environ:
    opencv_libs_path = os.environ["OPENCV_LIBS"]
else:
    opencv_libs_path = find_path("libopencv_core.so")

opencv_libs = list_opencv_libs(opencv_libs_path)

if "GPU_ARCH" in os.environ:
    gpu_arch = os.environ["GPU_ARCH"]
else:
    gpu_arch = 52

eppm_src = [
    "EPPM/bao_pmflow_census_kernel.cu",
    "EPPM/bao_pmflow_refine_kernel.cu",
    "EPPM/bao_flow_patchmatch_multiscale_cuda.cpp",
    "EPPM/bao_flow_patchmatch_multiscale_kernel.cu",
    "EPPM/bao_pmflow_kernel.cu",
    "EPPM/basic/bao_basic_cuda.cpp",
]

extensions = [
    Extension(
        "optflow",
        sources=["optflow.pyx"] + eppm_src,
        include_dirs=[numpy.get_include(), cuda["include"], "EPPM", "EPPM/basic"]
        + opencv_incs,
        language="c++",
        extra_link_args=["-L", opencv_libs_path]
        + opencv_libs
        + ["-L", cuda["lib"], "-lcudart", "-g"],
        extra_compile_args={
            "gcc": ["-g"],
            "nvcc": [
                "-arch=sm_{}".format(gpu_arch),
                "-gencode=arch=compute_{},code=sm_{}".format(gpu_arch, gpu_arch),
                "--ptxas-options=-v",
                "-c",
                "--compiler-options",
                "'-fPIC'",
                "--compiler-bindir",
                cuda_gcc,
            ],
        },
    )
]

setup(
    name="optflow",
    version="1.0",
    ext_modules=cythonize(extensions),
    cmdclass={"build_ext": custom_build_ext},
)
