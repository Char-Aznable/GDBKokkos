#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright 2020 Char Aznable <aznable.char.0083@gmail.com>
# Description: Utilities for pretty-printing Kokkos::View
#
# Distributed under terms of the 3-clause BSD license.
import numpy as np
import re
import subprocess
import pytest
import textwrap

@pytest.fixture(scope="module", params=["Kokkos::LayoutRight",
                                        "Kokkos::LayoutLeft",
                                        "Kokkos::LayoutStride"])
def foreachViewLayout(request):
    cpp = getattr(request.module, "cpp", "")
    layout = request.param
    shape = getattr(request.module, "shape", (3, 4, 5))
    # For testing purpose, we fix the strides in the case of
    # Kokkos::LayoutStride. For LayoutLeft and LayoutRight, we set the
    # strides to be the expected ones
    strides = (1, 3, 15)
    if layout == "Kokkos::LayoutRight":
        strides = np.ones_like(shape, dtype=int)
        strides[1:] = shape[::-1][:-1]
        strides = tuple(np.cumprod(strides)[::-1])
    elif layout == "Kokkos::LayoutLeft":
        strides = np.ones_like(shape, dtype=int)
        strides[1:] = shape[:-1]
        strides = tuple(np.cumprod(strides))
    ctorStride = ",".join(
        map(str, [ i for t in zip(shape, strides) for i in t ]))
    ctorShape = ",".join(map(str, shape))
    # Set the C++ template parameter for layout
    cpp = re.sub(r"/\*TestViewLayoutTparam\*/", layout, cpp)
    # Set the ctor for the view defining the shape and strides
    ctor = {"Kokkos::LayoutRight" : ctorShape,
        "Kokkos::LayoutLeft" : ctorShape,
        "Kokkos::LayoutStride" : f"{layout}({ctorStride})",
        }
    cpp = re.sub(r"/\*TestViewLayoutCtor\*/", ctor[layout], cpp)

    yield layout, shape, strides, cpp


@pytest.fixture(scope="module")
def writeCPP(tmpdir_factory, foreachViewLayout):
    """Write a c++ source file for gdb to examine the Kokkos::View

    Args:
        tmpdir_factory (TempdirFactory): pytest fixture for generating a
        temporary directory and file

    Returns: TODO

    """
    layout, shape, strides, cpp = foreachViewLayout
    fnShape = "-".join(map(str, shape))
    fnLayout = re.sub(r"::", "", layout)
    p = tmpdir_factory.mktemp(f"TestView{fnLayout}{fnShape}")
    fCXX = p.join("test.cpp")
    fCXX.write(textwrap.dedent(cpp))
    yield (p, fCXX, layout, shape, strides)


@pytest.fixture(scope="module")
def generateCMakeProject(writeCPP):
    p, _, layout, shape, strides = writeCPP
    fCMakeLists = p.join("CMakeLists.txt")
    fCMakeLists.write(textwrap.dedent(
        """
        cmake_minimum_required(VERSION 3.14)
        project(testLayoutRight CXX)
        include(FetchContent)

        FetchContent_Declare(
          kokkos
          GIT_REPOSITORY https://github.com/kokkos/kokkos.git
          GIT_TAG        origin/develop
          GIT_SHALLOW 1
          GIT_PROGRESS ON
        )

        FetchContent_MakeAvailable(kokkos)

        aux_source_directory(${CMAKE_CURRENT_LIST_DIR} testSources)

        foreach(testSource ${testSources})
          get_filename_component(testBinary ${testSource} NAME_WE)
          add_executable(${testBinary} ${testSource})
          target_include_directories(${testBinary} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
          target_link_libraries(${testBinary} kokkos)
          add_test(NAME ${testBinary} COMMAND ${testBinary})
        endforeach()
        """
        ))
    fBuildDir = p.mkdir("build")
    with fBuildDir.as_cwd():
        cmdCmake = [f"cmake {fCMakeLists.dirpath().strpath} "
                    f"-DCMAKE_CXX_COMPILER=x86_64-conda-linux-gnu-g++ "
                    f"-DCMAKE_CXX_STANDARD=14 "
                    f"-DCMAKE_CXX_FLAGS='-DDEBUG -O0 -g' "
                    f"-DCMAKE_VERBOSE_MAKEFILE=ON "
                    ]
        rCmake = subprocess.run(cmdCmake, shell=True, capture_output=True,
                                encoding='utf-8')
        assert rCmake.returncode == 0, f"Error with running cmake: {rCmake.stderr}"
        cmdMake = [f"make -j"]
        rMake = subprocess.run(cmdMake, shell=True, capture_output=True,
                               encoding='utf-8')
        assert rMake.returncode == 0, f"Error with running make: {rMake.stderr}"
    fTest = fBuildDir.join("test")
    yield (p, fBuildDir, fTest, layout, shape, strides)


@pytest.fixture(scope="function")
def runGDB(request):
    content = getattr(request.module, "gdbinit", "")
    def _runGDB(gdbComms : str, fPath, executable : str):
        nonlocal content
        content = re.sub(r"/\*TestGDBComms\*/", gdbComms, content)
        fPath.write(textwrap.dedent(content))
        cmd = [f"gdb -batch "
               f"{executable} "
               f"-x {fPath.realpath().strpath}"
               ]
        r = subprocess.run(cmd,
                           shell=True,
                           capture_output=True,
                           encoding='utf-8'
                           )
        assert r.returncode == 0,f"GDB error: {r.stderr}"
        # Get the output from GDB since the last breakpoint mark
        return (r,
                re.compile(r"\s*\d+\s*breakpoint\(\)\s*;\s*\n").split(r.stdout)[1].strip())
    return _runGDB

