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

def testLayout(compileCPP, runGDB):
    p, cppSources = compileCPP
    for _, row in cppSources.iterrows():
        fTest = row["fTest"]
        fGDB = p.join(fTest.purebasename + "_TestLayout.gdbinit")
        r, resultStr = runGDB(
            """
            py from GDBKokkos.printView import getKokkosViewLayout
            py v = gdb.parse_and_eval("v")
            py print(getKokkosViewLayout(v))

            """,
            fGDB, f"{fTest.realpath().strpath}"
            )
        assert resultStr == row["layout"], f"Wrong output layout: {r.stdout}"


def testExtent(compileCPP, runGDB):
    p, cppSources = compileCPP
    for _, row in cppSources.iterrows():
        fTest = row["fTest"]
        fGDB = p.join(fTest.purebasename + "_TestExtent.gdbinit")
        r, resultStr = runGDB(
            """
            py from GDBKokkos.printView import getKokkosViewExtent
            py v = gdb.parse_and_eval("v")
            py print(getKokkosViewExtent(v))

            """,
            fGDB, f"{fTest.realpath().strpath}"
            )
        result = np.array(re.sub(r"\D+", " ", resultStr).split(), dtype=int)
        assert (result == np.array(shape)).all(), f"Wrong output extent: {r.stdout}"


def testSpan(compileCPP, runGDB):
    p, cppSources = compileCPP
    for _, row in cppSources.iterrows():
        fTest = row["fTest"]
        stride = row["stride"]
        shape = row["shape"]
        fGDB = p.join(fTest.purebasename + "_TestSpan.gdbinit")
        r, resultStr = runGDB(
            """
            py from GDBKokkos.printView import getKokkosViewSpan
            py v = gdb.parse_and_eval("v")
            py print(getKokkosViewSpan(v))

            """,
            fGDB, f"{fTest.realpath().strpath}"
            )
        iMax = stride.argmax()
        expected = shape[iMax] * stride[iMax]
        assert int(resultStr) == expected, "Wrong output Span: {r.stdout}"


def testStrides(compileCPP, runGDB):
    p, cppSources = compileCPP
    for _, row in cppSources.iterrows():
        fTest = row["fTest"]
        stride = row["stride"]
        fGDB = p.join(fTest.purebasename + "_TestStrides.gdbinit")
        r, resultStr = runGDB(
            """
            py from GDBKokkos.printView import getKokkosViewStrides
            py v = gdb.parse_and_eval("v")
            py print(getKokkosViewStrides(v))

            """,
            fGDB, f"{fTest.realpath().strpath}"
            )
        result = np.array(re.sub(r"\D+", " ", resultStr).split(), dtype=int)
        expected = stride
        assert (result == expected).all(), f"Wrong output strides: {r.stdout}"


def testTraits(compileCPP, runGDB):
    p, cppSources = compileCPP
    for _, row in cppSources.iterrows():
        fTest = row["fTest"]
        layout = row["layout"]
        fGDB = p.join(fTest.purebasename + "_TestTraits.gdbinit")
        r, resultStr = runGDB(
            """
            py from GDBKokkos.printView import getKokkosViewTraits
            py v = gdb.parse_and_eval("v")
            py print(getKokkosViewTraits(v))

            """,
            fGDB, f"{fTest.realpath().strpath}"
            )
        assert resultStr == f"Kokkos::ViewTraits<float***, {layout}>",\
            "Wrong output traits: {r.stdout}"


def testValueType(compileCPP, runGDB):
    p, cppSources = compileCPP
    for _, row in cppSources.iterrows():
        fTest = row["fTest"]
        fGDB = p.join(fTest.purebasename + "_TestValueType.gdbinit")
        r, resultStr = runGDB(
            """
            py from GDBKokkos.printView import getKokkosViewValueType
            py v = gdb.parse_and_eval("v")
            py print(getKokkosViewValueType(v))

            """,
            fGDB, f"{fTest.realpath().strpath}"
            )
        assert resultStr == f"float",\
            "Wrong output ValueType: {r.stdout}"


def testPrintView(compileCPP, runGDB):
    p, cppSources = compileCPP
    for _, row in cppSources.iterrows():
        fTest = row["fTest"]
        fGDB = p.join(fTest.purebasename + "_TestPrintView.gdbinit")
        stride = row["stride"]
        layout = row["layout"]
        r, resultStr = runGDB(
            """
            py import GDBKokkos
            printView v --noIndex

            """,
            fGDB, f"{fTest.realpath().strpath}"
            )

        result = np.array(resultStr.split()).astype(float).reshape(shape)
        iMax = stride.argmax()
        span = shape[iMax] * stride[iMax]
        expected = np.arange(span).astype(float)
        if layout == "Kokkos::LayoutRight":
            expected = expected.reshape(shape, order="C")
        elif layout == "Kokkos::LayoutLeft":
            expected = expected.reshape(shape, order="F")
        elif layout == "Kokkos::LayoutStride":
            expected = np.lib.stride_tricks.as_strided(
                expected, shape=shape,
                strides=stride * expected.dtype.itemsize,
                writeable=False)
        assert (result == expected).all(),\
            f"printView gives wrong view of shape {tuple}: {r.stdout}. Expected:\n"\
            f"{expected}"
