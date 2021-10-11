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
import pandas as pd

@pytest.fixture(scope="session")
def initCMakeProject(tmpdir_factory):
    p = tmpdir_factory.mktemp(f"testGDBKokkos")
    fCMakeLists = p.join("CMakeLists.txt")
    fCMakeLists.write(textwrap.dedent(
        """
        cmake_minimum_required(VERSION 3.14)
        project(testGDBKokkos CXX)
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
    yield p


@pytest.fixture(scope="module")
def generateViewTypes(request):
    cppTemplate = getattr(request.module, "cpp", "")
    nameStruct = getattr(request.module, "nameStruct", "")
    cppStruct = getattr(request.module, "cppStruct", "")
    # Replace the nested struct definition in the template
    cppTemplate = re.sub(r"/\*TestViewStruct\*/", cppStruct, cppTemplate)
    layoutsInput = ["Kokkos::LayoutRight", "Kokkos::LayoutLeft",
               "Kokkos::LayoutStride"]
    valueTypesInput = ["int", "long", "unsigned int", "unsigned long", "float",
                       "double", nameStruct]
    nRanksTotal = 3
    nDynamicRanksInput = np.arange(nRanksTotal + 1)
    typeCombo = (layoutsInput, valueTypesInput, nDynamicRanksInput)
    types = pd.DataFrame(
        np.array(np.meshgrid(*typeCombo)).T.reshape(-1, len(typeCombo)),
        columns=["layout", "valueType", "nDynamicRanks"])
    types['nDynamicRanks'] = types['nDynamicRanks'].astype(int)
    shape = np.array(getattr(request.module, "shape", [3, 4, 5]))

    assert shape.size == nRanksTotal,\
        f"Input shape {shape} is not rank {nRanksTotal}"

    # Generate the array type given valueTypes and nDynamicRanksInput
    # Columns of arrayRanks are indexed by the respective number of ranks
    # in dynamic or static dimension
    staticRanks = [""] + [ f"[{i}]" for i in map(str, shape[::-1]) ]
    dynamicRanks = [""] + ["*"] * shape.size
    arrayStaticRanks = []
    arrayDynamicRanks = []
    for i in range(len(staticRanks)):
        arrayStaticRanks.append(staticRanks[i])
        arrayDynamicRanks.append(dynamicRanks[i])
        if i > 0:
            arrayStaticRanks[-1] += arrayStaticRanks[i - 1]
            arrayDynamicRanks[-1] += arrayDynamicRanks[i - 1]

    arrayRanks = pd.DataFrame({"rankStatic" : arrayStaticRanks,
        "rankDynamic": arrayDynamicRanks})

    layouts = []
    valueTypes = []
    arrayTypes = []
    cpps = []
    strides = []
    shapes = []
    for _, row in types.iterrows():
        layout = row["layout"]
        valueType = row["valueType"]
        nDynamicRanks = row["nDynamicRanks"]
        nStaticRanks = nRanksTotal - nDynamicRanks
        arrayExtents = str(arrayRanks['rankDynamic'][nDynamicRanks]) + \
            str(arrayRanks['rankStatic'][nStaticRanks])
        arrayType = valueType + arrayExtents
        ctor = ""
        stride = np.ones_like(shape, dtype=int)
        shapeDynamicRanks = shape[:nDynamicRanks]
        # Compute the expected stride and ctor arguments for the view
        if layout == "Kokkos::LayoutRight":
            stride[1:] = shape[::-1][:-1]
            stride = np.cumprod(stride)[::-1]
            ctor = ",".join(map(str, shapeDynamicRanks))
        elif layout == "Kokkos::LayoutLeft":
            stride[1:] = shape[:-1]
            stride = np.cumprod(stride)
            ctor = ",".join(map(str, shapeDynamicRanks))
        elif layout == "Kokkos::LayoutStride":
            # For testing purpose, we fix the stride in the case of
            # Kokkos::LayoutStride. As far as I can tell, the layout ctor needs
            # input for all ranks even if the static ranks have implicit extents
            # already specified in the type
            stride = np.array([1, 3, 15])
            ctorStride = ",".join(
                map(str, [ i for t in zip(shape, stride) for i in t ]))
            ctor = f"{layout}({ctorStride})"
        # Set the C++ template parameter for layout
        cpp = re.sub(r"/\*TestViewLayoutTparam\*/", str(layout), cppTemplate)
        # Set the view's value type
        cpp = re.sub(r"/\*TestViewValueType\*/", str(valueType), cpp)
        # Set the ctor for the view defining the shape and stride
        # Set view array type
        cpp = re.sub(r"/\*TestViewArrExtents\*/", arrayExtents, cpp)
        cpp = re.sub(r"/\*TestViewLayoutCtor\*/",
                     f", {ctor}" if ctor != "" else "", cpp)
        layouts.append(layout)
        valueTypes.append(valueType)
        arrayTypes.append(arrayType)
        cpps.append(cpp)
        strides.append(stride)
        shapes.append(shape)
    ans = pd.DataFrame({"cpp" : cpps, "layout" : layouts,
                        "valueType" : valueTypes,
                        "nDynamicRanks" : types["nDynamicRanks"],
                        "arrayType" : arrayTypes,
                        "stride" : strides, "shape" : shapes })
    yield ans


@pytest.fixture(scope="module")
def writeCPP(initCMakeProject, generateViewTypes):
    p = initCMakeProject
    cppSources = generateViewTypes
    fcpps = []
    for _, row in cppSources.iterrows():
        cpp = row["cpp"]
        layout = row["layout"]
        arrayType = row["arrayType"]
        shape = row["shape"]
        fnShape = "-".join(map(str, shape))
        fnLayout = re.sub(r"::", "", layout)
        fnArrayType = re.sub(r"\s+", "", arrayType)
        fnArrayType = re.sub(r"[*]", "x", fnArrayType)
        fnArrayType = re.sub(r"\[", "", fnArrayType)
        fnArrayType = re.sub(r"\]", "-", fnArrayType)
        fcpp = p.join(f"testView{fnArrayType}{fnLayout}{fnShape}.cpp")
        fcpp.write(textwrap.dedent(cpp))
        fcpps.append(fcpp)
    cppSources.insert(1, "fcpp", fcpps)
    yield p, cppSources


@pytest.fixture(scope="module")
def compileCPP(writeCPP):
    p, cppSources = writeCPP
    fBuildDir = p.mkdir("build")
    with fBuildDir.as_cwd():
        cmdCmake = [f"cmake {str(p)} "
                    f"-DCMAKE_CXX_COMPILER=x86_64-conda-linux-gnu-g++ "
                    f"-DCMAKE_CXX_STANDARD=14 "
                    f"-DCMAKE_CXX_FLAGS='-DDEBUG -O0 -g' "
                    f"-DCMAKE_VERBOSE_MAKEFILE=ON "
                    ]
        rCmake = subprocess.run(cmdCmake, shell=True, capture_output=True,
                                encoding="utf-8")
        assert rCmake.returncode == 0, f"Error with running cmake: {rCmake.stderr}"
        cmdMake = [f"make -j"]
        rMake = subprocess.run(cmdMake, shell=True, capture_output=True,
                               encoding="utf-8")
        assert rMake.returncode == 0, f"Error with running make: {rMake.stderr}"
    # Get the executables
    fTests = []
    for _, row in cppSources.iterrows():
        fcpp = row["fcpp"]
        print(fcpp)
        fTest = fcpp.purebasename
        fTests.append(fBuildDir.join(str(fTest)))
    cppSources.insert(2, "fTest", fTests)
    yield p, cppSources


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
                           encoding="utf-8"
                           )
        assert r.returncode == 0,f"GDB error: {r.stderr}"
        # Get the output from GDB since the last breakpoint mark
        return (r,
                re.compile(r"\s*\d+\s*breakpoint\(\)\s*;\s*\n").split(r.stdout)[1].strip())
    return _runGDB

