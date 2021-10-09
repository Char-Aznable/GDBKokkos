#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright 2020 Char Aznable <aznable.char.0083@gmail.com>
# Description: Test basic view properties retrieve from the GDB debug session
#
# Distributed under terms of the 3-clause BSD license.
import re
import numpy as np

shape = (3, 4, 5)
cpp = f"""
    #include <Kokkos_Core.hpp>

    void breakpoint() {{ return; }}

    int main(int argc, char* argv[]) {{
      Kokkos::initialize(argc,argv); {{

      Kokkos::View<float***, /*TestViewLayoutTparam*/> v("v",
        /*TestViewLayoutCtor*/);

      Kokkos::parallel_for(v.span(), KOKKOS_LAMBDA (const int i)
        {{ *(v.data()+i) = i; }});

      breakpoint();

      }} Kokkos::finalize();

    }}
"""

gdbinit = f"""
    set confirm off

    b breakpoint()
      commands
        silent
        frame 1
      end

    run

    /*TestGDBComms*/
    quit
    """

def testLayout(generateCMakeProject, runGDB):
    p, _, fTest, layout, _, _ = generateCMakeProject
    fGDB = p.join("testLayout.gdbinit")
    r, resultStr = runGDB(
        """
        py from GDBKokkos.printView import getKokkosViewLayout
        py v = gdb.parse_and_eval('v')
        py print(getKokkosViewLayout(v))

        """,
        fGDB, f"{fTest.realpath().strpath}"
        )
    assert resultStr == layout, f"Wrong output layout: {r.stdout}"


def testExtent(generateCMakeProject, runGDB):
    p, _, fTest, _, shape, _ = generateCMakeProject
    fGDB = p.join("testExtent.gdbinit")
    r, resultStr = runGDB(
        """
        py from GDBKokkos.printView import getKokkosViewExtent
        py v = gdb.parse_and_eval('v')
        py print(getKokkosViewExtent(v))

        """,
        fGDB, f"{fTest.realpath().strpath}"
        )
    result = np.array(re.sub(r"\D+", " ", resultStr).split(), dtype=int)
    assert (result == np.array(shape)).all(), f"Wrong output extent: {r.stdout}"


def testSpan(generateCMakeProject, runGDB):
    p, _, fTest, layout, shape, strides = generateCMakeProject
    fGDB = p.join("testSpan.gdbinit")
    r, resultStr = runGDB(
        """
        py from GDBKokkos.printView import getKokkosViewSpan
        py v = gdb.parse_and_eval('v')
        py print(getKokkosViewSpan(v))

        """,
        fGDB, f"{fTest.realpath().strpath}"
        )
    iMax = np.array(strides).argmax()
    expected = shape[iMax] * strides[iMax]
    assert int(resultStr) == expected, "Wrong output Span: {r.stdout}"


def testStrides(generateCMakeProject, runGDB):
    p, _, fTest, layout, shape, strides = generateCMakeProject
    fGDB = p.join("testStrides.gdbinit")
    r, resultStr = runGDB(
        """
        py from GDBKokkos.printView import getKokkosViewStrides
        py v = gdb.parse_and_eval('v')
        py print(getKokkosViewStrides(v))

        """,
        fGDB, f"{fTest.realpath().strpath}"
        )
    result = np.array(re.sub(r"\D+", " ", resultStr).split(), dtype=int)
    expected = np.array(strides)
    assert (result == expected).all(), f"Wrong output strides: {r.stdout}"


def testTraits(generateCMakeProject, runGDB):
    p, _, fTest, layout, _, _ = generateCMakeProject
    fGDB = p.join("testTraits.gdbinit")
    r, resultStr = runGDB(
        """
        py from GDBKokkos.printView import getKokkosViewTraits
        py v = gdb.parse_and_eval('v')
        py print(getKokkosViewTraits(v))

        """,
        fGDB, f"{fTest.realpath().strpath}"
        )
    assert resultStr == f"Kokkos::ViewTraits<float***, {layout}>",\
        "Wrong output traits: {r.stdout}"


def testValueType(generateCMakeProject, runGDB):
    p, _, fTest, _, _, _ = generateCMakeProject
    fGDB = p.join("testValueType.gdbinit")
    r, resultStr = runGDB(
        """
        py from GDBKokkos.printView import getKokkosViewValueType
        py v = gdb.parse_and_eval('v')
        py print(getKokkosViewValueType(v))

        """,
        fGDB, f"{fTest.realpath().strpath}"
        )
    assert resultStr == f"float",\
        "Wrong output ValueType: {r.stdout}"


def testPrintView(generateCMakeProject, runGDB):
    p, _, fTest, layout, shape, strides = generateCMakeProject
    fGDB = p.join("testPrintView.gdbinit")
    r, resultStr = runGDB(
        """
        py import GDBKokkos
        printView v --noIndex

        """,
        fGDB, f"{fTest.realpath().strpath}"
        )

    result = np.array(resultStr.split()).astype(float).reshape(shape)
    iMax = np.array(strides).argmax()
    span = shape[iMax] * strides[iMax]
    expected = np.arange(span).astype(float)
    if layout == "Kokkos::LayoutRight":
        expected = expected.reshape(shape, order='C')
    elif layout == "Kokkos::LayoutLeft":
        expected = expected.reshape(shape, order='F')
    elif layout == "Kokkos::LayoutStride":
        expected = np.lib.stride_tricks.as_strided(
            expected, shape=shape,
            strides=np.array(strides) * expected.dtype.itemsize,
            writeable=False)
    assert (result == expected).all(),\
        f"printView gives wrong view of shape {tuple}: {r.stdout}. Expected:\n"\
        f"{expected}"
