#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright 2020 Char Aznable <aznable.char.0083@gmail.com>
# Description: Utilities for pretty-printing Kokkos::View
#
# Distributed under terms of the 3-clause BSD license.
import re
import numpy as np

shape = (3, 4, 5)
strShape = ",".join(map(str, shape))
layout = "Kokkos::LayoutRight"
span = np.array(shape).prod()
cpp = f"""
    #include <Kokkos_Core.hpp>

    void breakpoint() {{ return; }}

    int main(int argc, char* argv[]) {{
      Kokkos::initialize(argc,argv); {{

      Kokkos::View<float***,{layout}> v("v",{strShape});

      Kokkos::parallel_for(v.span(), KOKKOS_LAMBDA (const int i)
        {{ *(v.data()+i) = i; }});

      breakpoint();

      }} Kokkos::finalize();

    }}
    """

def testLayout(generateCMakeProject, runGDB):
    p, _, fTest = generateCMakeProject
    fGDB = p.join("testLayout.gdbinit")
    r, resultStr = runGDB(
        """
        py from GDBKokkos.printView import getKokkosViewLayout
        set confirm off

        b breakpoint()
          commands
            silent
            frame 1
          end

        run

        py v = gdb.parse_and_eval('v')
        py print(getKokkosViewLayout(v))
        quit

        """,
        fGDB, f"{fTest.realpath().strpath}"
        )
    assert resultStr == layout, f"Wrong output layout: {r.stdout}"


def testExtent(generateCMakeProject, runGDB):
    p, _, fTest = generateCMakeProject
    fGDB = p.join("testExtent.gdbinit")
    r, resultStr = runGDB(
        """
        py from GDBKokkos.printView import getKokkosViewExtent
        set confirm off

        b breakpoint()
          commands
            silent
            frame 1
          end

        run

        py v = gdb.parse_and_eval('v')
        py print(getKokkosViewExtent(v))
        quit

        """,
        fGDB, f"{fTest.realpath().strpath}"
        )
    result = np.array(re.sub(r"\D+", " ", resultStr).split(), dtype=int)
    assert (result == np.array(shape)).all(), f"Wrong output extent: {r.stdout}"


def testStrides(generateCMakeProject, runGDB):
    p, _, fTest = generateCMakeProject
    fGDB = p.join("testExtent.gdbinit")
    r, resultStr = runGDB(
        """
        py from GDBKokkos.printView import getKokkosViewStrides
        set confirm off

        b breakpoint()
          commands
            silent
            frame 1
          end

        run

        py v = gdb.parse_and_eval('v')
        py print(getKokkosViewStrides(v))
        quit

        """,
        fGDB, f"{fTest.realpath().strpath}"
        )
    result = np.array(re.sub(r"\D+", " ", resultStr).split(), dtype=int)
    expected = np.ones_like(shape, dtype=int)
    expected[1:] = shape[::-1][:-1]
    expected = np.cumprod(expected)[::-1]
    assert (result == expected).all(), f"Wrong output strides: {r.stdout}"


def testTraits(generateCMakeProject, runGDB):
    p, _, fTest = generateCMakeProject
    fGDB = p.join("testTraits.gdbinit")
    r, resultStr = runGDB(
        """
        py from GDBKokkos.printView import getKokkosViewTraits
        set confirm off

        b breakpoint()
          commands
            silent
            frame 1
          end

        run

        py v = gdb.parse_and_eval('v')
        py print(getKokkosViewTraits(v))
        quit

        """,
        fGDB, f"{fTest.realpath().strpath}"
        )
    assert resultStr == f"Kokkos::ViewTraits<float***, {layout}>",\
        "Wrong output traits: {r.stdout}"


def testValueType(generateCMakeProject, runGDB):
    p, _, fTest = generateCMakeProject
    fGDB = p.join("testValueType.gdbinit")
    r, resultStr = runGDB(
        """
        py from GDBKokkos.printView import getKokkosViewValueType
        set confirm off

        b breakpoint()
          commands
            silent
            frame 1
          end

        run

        py v = gdb.parse_and_eval('v')
        py print(getKokkosViewValueType(v))
        quit

        """,
        fGDB, f"{fTest.realpath().strpath}"
        )
    assert resultStr == f"float",\
        "Wrong output ValueType: {r.stdout}"


def testSpan(generateCMakeProject, runGDB):
    p, _, fTest = generateCMakeProject
    fGDB = p.join("testSpan.gdbinit")
    r, resultStr = runGDB(
        """
        py from GDBKokkos.printView import getKokkosViewSpan
        set confirm off

        b breakpoint()
          commands
            silent
            frame 1
          end

        run

        py v = gdb.parse_and_eval('v')
        py print(getKokkosViewSpan(v))
        quit

        """,
        fGDB, f"{fTest.realpath().strpath}"
        )
    assert int(resultStr) == np.array(shape).prod(),\
        "Wrong output Span: {r.stdout}"


def testPrintView(generateCMakeProject, runGDB):
    p, _, fTest = generateCMakeProject
    fGDB = p.join("testPrintView.gdbinit")
    r, resultStr = runGDB(
        """
        py import GDBKokkos
        set confirm off

        b breakpoint()
          commands
            silent
            frame 1
          end

        run

        printView v --noIndex
        quit

        """,
        fGDB, f"{fTest.realpath().strpath}"
        )

    result = np.array(resultStr.split()).astype(float).reshape(shape)
    expected = np.arange(span).astype(float).reshape(shape)
    assert (result == expected).all(),\
        f"printView gives wrong view of shape {tuple}: {r.stdout}. Expected:\n"\
        f"{expected}"
