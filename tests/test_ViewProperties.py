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
import ast

shape = [3, 4, 5]
nameStruct = "Nested"
cppStruct = f"""
    struct Inner {{
      long i_[2][3];
      bool b_[2];
      float f_[2];
    }};

    struct {nameStruct} {{
      KOKKOS_INLINE_FUNCTION {nameStruct}() = default;
      KOKKOS_INLINE_FUNCTION {nameStruct}(const int i)
        :a_{{{{i,i,i,i,i,i}}, {{i > 0, i > 0}}, {{i / 1.f, i / 1.f}}}}
        ,i_{{i}}
        ,d_{{i / 1.}}
        {{}}
      Inner a_;
      int i_;
      double d_;
    }};
    """
dtypeStruct = [('a_', [('i_', '|i8', (2, 3)), ('b_', '|b1', 2), ('f_', '|f4', 2)]),
               ('i_', '|i4'), ('d_', '|f8')]
cpp = f"""
    #include <Kokkos_Core.hpp>

    void breakpoint() {{ return; }}

    /*TestViewStruct*/

    using T = /*TestViewValueType*/;
    using Tarr = T/*TestViewArrExtents*/;

    int main(int argc, char* argv[]) {{
      Kokkos::initialize(argc,argv); {{

      Kokkos::View<Tarr, /*TestViewLayoutTparam*/> v("v"
        /*TestViewLayoutCtor*/);

      Kokkos::parallel_for(v.span(), KOKKOS_LAMBDA (const int i)
        {{ *(v.data()+i) = static_cast<T>(i); }});

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

def testLayout(compileTestView, runGDB):
    p, cppSources = compileTestView
    for _, row in cppSources.iterrows():
        fTest = row["fTest"]
        fGDB = p.join(fTest.purebasename + "_TestLayout.gdbinit")
        print(f"fTest = {fTest} fGDB = {fGDB}\n")
        r, resultStr = runGDB(
            """
            py from GDBKokkos.printView import getKokkosViewLayout
            py v = gdb.parse_and_eval("v")
            py print(getKokkosViewLayout(v))

            """,
            fGDB, f"{fTest.realpath().strpath}"
            )
        assert resultStr == row["layout"], f"Wrong output layout: {r.stdout}"


def testExtent(compileTestView, runGDB):
    p, cppSources = compileTestView
    for _, row in cppSources.iterrows():
        fTest = row["fTest"]
        fGDB = p.join(fTest.purebasename + "_TestExtent.gdbinit")
        print(f"fTest = {fTest} fGDB = {fGDB}\n")
        r, resultStr = runGDB(
            """
            py from GDBKokkos.printView import getKokkosViewExtent
            py v = gdb.parse_and_eval("v")
            py print(getKokkosViewExtent(v)[0])

            """,
            fGDB, f"{fTest.realpath().strpath}"
            )
        result = np.array(re.sub(r"\D+", " ", resultStr).split(), dtype=int)
        assert (result == row["shape"]).all(), f"Wrong output extent: {r.stdout}"


def testSpan(compileTestView, runGDB):
    p, cppSources = compileTestView
    for _, row in cppSources.iterrows():
        fTest = row["fTest"]
        stride = row["stride"]
        shape = row["shape"]
        fGDB = p.join(fTest.purebasename + "_TestSpan.gdbinit")
        print(f"fTest = {fTest} fGDB = {fGDB}\n")
        r, resultStr = runGDB(
            """
            py from GDBKokkos.printView import getKokkosViewSpan
            py v = gdb.parse_and_eval("v")
            py print(getKokkosViewSpan(v))

            """,
            fGDB, f"{fTest.realpath().strpath}"
            )
        if row["nDynamicRanks"] == -1:
            # This is a scalar view
            expected = 1
        else:
            iMax = stride.argmax()
            expected = shape[iMax] * stride[iMax]
        assert int(resultStr) == expected, "Wrong output Span: {r.stdout}"


def testStrides(compileTestView, runGDB):
    p, cppSources = compileTestView
    for _, row in cppSources.iterrows():
        fTest = row["fTest"]
        stride = row["stride"]
        fGDB = p.join(fTest.purebasename + "_TestStrides.gdbinit")
        print(f"fTest = {fTest} fGDB = {fGDB}\n")
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


def testTraits(compileTestView, runGDB):
    p, cppSources = compileTestView
    for _, row in cppSources.iterrows():
        fTest = row["fTest"]
        layout = row["layout"]
        arrayType = row["arrayType"]
        fGDB = p.join(fTest.purebasename + "_TestTraits.gdbinit")
        print(f"fTest = {fTest} fGDB = {fGDB}\n")
        r, resultStr = runGDB(
            """
            py from GDBKokkos.printView import getKokkosViewTraits
            py v = gdb.parse_and_eval("v")
            py print(getKokkosViewTraits(v))

            """,
            fGDB, f"{fTest.realpath().strpath}"
            )
        # Remove space to prevent choking on type alias
        # but add back the space after 'unsigned' keyword
        resultStr = re.sub(r"\s+", "", resultStr)
        resultStr = re.sub(r"unsigned", "unsigned ", resultStr)
        assert resultStr == f"Kokkos::ViewTraits<{arrayType},{layout}>",\
            "Wrong output traits: {r.stdout}"


def testValueType(compileTestView, runGDB):
    p, cppSources = compileTestView
    for _, row in cppSources.iterrows():
        fTest = row["fTest"]
        valueType = row["valueType"]
        fGDB = p.join(fTest.purebasename + "_TestValueType.gdbinit")
        print(f"fTest = {fTest} fGDB = {fGDB}\n")
        r, resultStr = runGDB(
            """
            py from GDBKokkos.printView import getKokkosViewValueType
            py v = gdb.parse_and_eval("v")
            py print(getKokkosViewValueType(v))

            """,
            fGDB, f"{fTest.realpath().strpath}"
            )
        assert resultStr == valueType,\
            "Wrong output ValueType: {r.stdout}"


def viewValueType2NumpyDtype(valueType):
    ans = {"int" : "|i4", "long" : "|i8", "unsigned int" : "|u4",
           "unsigned long"  : "|u8", "float" : "|f4", "double" : "|f8",
           nameStruct : dtypeStruct}
    return ans[valueType]


def testPrintView(compileTestView, runGDB):
    p, cppSources = compileTestView
    for _, row in cppSources.iterrows():
        fTest = row["fTest"]
        fGDB = p.join(fTest.purebasename + "_TestPrintView.gdbinit")
        stride = row["stride"]
        layout = row["layout"]
        valueType = row["valueType"]
        shape = row["shape"]
        print(f"fTest = {fTest} fGDB = {fGDB}\n")
        r, resultStr = runGDB(
            """
            py import GDBKokkos
            printView v --noIndex

            """,
            fGDB, f"{fTest.realpath().strpath}"
            )
        dtype = np.dtype(viewValueType2NumpyDtype(valueType))
        # Process the result into numpy array. The '--noIndex' option render the
        # multidimensional array as layout-right multi-line string and we parse
        # that string into shaped array
        if dtype.names is None:
            # just split the string into array for simple numeric type
            result = np.array(resultStr.split(), dtype=dtype).reshape(shape)
        else:
            # need to trim off whitespaces and convert to list of tuples for
            # nested structured type
            resultStr = re.sub(r"\n", " ", resultStr)
            resultStr = re.sub(r"^\s+", "", resultStr)
            resultStr = re.sub(r"\)\s+\(", "), (", resultStr)
            # Add "," between elements of nested list
            resultStr = re.sub(r"((?<=\d))\s+((?=\d+))", r"\g<1>, \g<2>", resultStr)
            l = list(ast.literal_eval(resultStr))
            if shape.size == 0:
                # this is a scalar view so l is a list but we need a tuple
                l = tuple(l)
            result = np.array(l, dtype=dtype).reshape(shape)
        # Get the expected array by reshaping np.arange(span) taking into
        # account stride
        if stride.size > 0:
            iMax = stride.argmax()
            span = shape[iMax] * stride[iMax]
        else:
            # This is a scalar view
            span = 1
        expected = np.arange(span).astype(dtype)
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
            f"printView gives wrong view of shape {shape}: {r.stdout}. Expected:\n"\
            f"{expected}\nBut got:\n"\
            f"{result}"
