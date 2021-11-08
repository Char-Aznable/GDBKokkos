#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright 2021 Char Aznable <aznable.char.0083@gmail.com>
# Description: Test utils in GDBKokkos.printCrsMatrix
#
# Distributed under terms of the MIT license.
import re
import numpy as np
import io

cpp = f"""
    #include <Kokkos_Core.hpp>
    #include <KokkosSparse_CrsMatrix.hpp>

    using HostExecutionSpace = Kokkos::DefaultHostExecutionSpace;
    using HostMemorySpace = HostExecutionSpace::memory_space;
    using HostDevice = Kokkos::Device<HostExecutionSpace, HostMemorySpace>;

    using M = typename KokkosSparse::CrsMatrix<float, int, HostDevice>;
    using RowMap = typename M::row_map_type::non_const_type;
    using ColIdx = typename M::index_type;
    using Values  = typename M::values_type;
    using O = typename RowMap::value_type;
    using I = typename ColIdx::value_type;
    using V = typename Values::value_type;

    void breakpoint() {{ return; }}

    int main(int argc, char* argv[]) {{
      Kokkos::initialize(argc,argv); {{

      const int nCols = 6;
      const int nRows = 4;
      const O nnz = 8;
      Kokkos::View<V*> vs("vs", nnz), vs1("vs1", nnz);;
      Kokkos::View<O*> rs("rs", nRows + 1);
      Kokkos::View<I*> cs("cs", nnz);

      for(O i = 0; i < nnz; ++i) {{
        vs(i) = i + 1;
        vs1(i) = vs(i) + 10;
      }}
      rs(0) = 0;
      rs(1) = 2;
      rs(2) = 4;
      rs(3) = 7;
      rs(4) = 8;
      cs(0) = 0;
      cs(1) = 1;
      cs(2) = 1;
      cs(3) = 3;
      cs(4) = 2;
      cs(5) = 3;
      cs(6) = 4;
      cs(7) = 5;

      /* matrix([[1, 2, 0, 0, 0, 0], */
      /*         [0, 3, 0, 4, 0, 0], */
      /*         [0, 0, 5, 6, 7, 0], */
      /*         [0, 0, 0, 0, 0, 8]]) */
      M m("m", nRows, nCols, nnz, vs, rs, cs);

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


def testCrs2ScipySparse(compileTestCrsMatrix, runGDB):
    p, cppSources = compileTestCrsMatrix
    for _, row in cppSources.iterrows():
        fTest = row["fTest"]
        fGDB = p.join(fTest.purebasename + "_TestCrs2ScipySparse.gdbinit")
        print(f"fTest = {fTest} fGDB = {fGDB}\n")
        r, resultStr = runGDB(
            """
            py from GDBKokkos.printCrsMatrix import crs2ScipySparse
            py m = gdb.parse_and_eval("m")
            py print(crs2ScipySparse(m).toarray())

            """,
            fGDB, f"{fTest.realpath().strpath}"
            )
        resultStr = re.sub(r"[\[\]]", "", resultStr)
        result = np.array(resultStr.split(), dtype=np.float32).reshape(4, 6)
        expected = np.array(
            [[1, 2, 0, 0, 0, 0],
             [0, 3, 0, 4, 0, 0],
             [0, 0, 5, 6, 7, 0],
             [0, 0, 0, 0, 0, 8]], dtype=np.float32)
        assert (result == expected).all(),\
            f"""crs2ScipySparse gives wrong view {result}. Expected:
            {expected}
            """


def testCrs2ScipySparseValues(compileTestCrsMatrix, runGDB):
    p, cppSources = compileTestCrsMatrix
    for _, row in cppSources.iterrows():
        fTest = row["fTest"]
        fGDB = p.join(fTest.purebasename + "_TestCrs2ScipySparseValues.gdbinit")
        print(f"fTest = {fTest} fGDB = {fGDB}\n")
        r, resultStr = runGDB(
            """
            py from GDBKokkos.printCrsMatrix import crs2ScipySparse
            py from GDBKokkos.printView import view2NumpyArray
            py m = gdb.parse_and_eval("m")
            py vs1 = view2NumpyArray(gdb.parse_and_eval("vs1"))
            py print(crs2ScipySparse(m, vs1).toarray())

            """,
            fGDB, f"{fTest.realpath().strpath}"
            )
        resultStr = re.sub(r"[\[\]]", "", resultStr)
        result = np.array(resultStr.split(), dtype=np.float32).reshape(4, 6)
        expected = np.array(
            [[11, 12, 0, 0,  0,  0],
             [0,  13, 0, 14, 0,  0],
             [0,  0, 15, 16, 17, 0],
             [0,  0,  0,  0, 0, 18]], dtype=np.float32)
        assert (result == expected).all(),\
            f"""crs2ScipySparse gives wrong view {result}. Expected:
            {expected}
            """


def testPrintCrsMatrix(compileTestCrsMatrix, runGDB):
    p, cppSources = compileTestCrsMatrix
    for _, row in cppSources.iterrows():
        fTest = row["fTest"]
        fGDB = p.join(fTest.purebasename + "_TestPrintCrsMatrix.gdbinit")
        print(f"fTest = {fTest} fGDB = {fGDB}\n")
        r, resultStr = runGDB(
            """
            py import GDBKokkos
            printCrsMatrix m

            """,
            fGDB, f"{fTest.realpath().strpath}"
            )
        resultStr = re.sub(r"[\[\]]", "", resultStr)
        fh = io.StringIO(resultStr)
        result = np.loadtxt(fh)
        expected = np.array(
            [[1, 2, 0, 0, 0, 0],
             [0, 3, 0, 4, 0, 0],
             [0, 0, 5, 6, 7, 0],
             [0, 0, 0, 0, 0, 8]], dtype=np.float32)
        assert (result == expected).all(),\
            f"""printCrsMatrix gives wrong view {result}. Expected:
            {expected}
            """


def testPrintCrsMatrixValues(compileTestCrsMatrix, runGDB):
    p, cppSources = compileTestCrsMatrix
    for _, row in cppSources.iterrows():
        fTest = row["fTest"]
        fGDB = p.join(fTest.purebasename + "_TestPrintCrsMatrixValues.gdbinit")
        print(f"fTest = {fTest} fGDB = {fGDB}\n")
        r, resultStr = runGDB(
            """
            py import GDBKokkos
            printCrsMatrix m --values vs1

            """,
            fGDB, f"{fTest.realpath().strpath}"
            )
        resultStr = re.sub(r"[\[\]]", "", resultStr)
        fh = io.StringIO(resultStr)
        result = np.loadtxt(fh)
        expected = np.array(
            [[11, 12, 0, 0,  0,  0],
             [0,  13, 0, 14, 0,  0],
             [0,  0, 15, 16, 17, 0],
             [0,  0,  0,  0, 0, 18]], dtype=np.float32)
        assert (result == expected).all(),\
            f"""printCrsMatrix --values gives wrong view {result}. Expected:
            {expected}
            """


def testPrintCrsMatrixCoo(compileTestCrsMatrix, runGDB):
    p, cppSources = compileTestCrsMatrix
    for _, row in cppSources.iterrows():
        fTest = row["fTest"]
        fGDB = p.join(fTest.purebasename + "_TestPrintCrsMatrixCoo.gdbinit")
        print(f"fTest = {fTest} fGDB = {fGDB}\n")
        r, resultStr = runGDB(
            """
            py import GDBKokkos
            printCrsMatrix m --printCoo

            """,
            fGDB, f"{fTest.realpath().strpath}"
            )
        resultStr = re.sub(r"[(),]", "", resultStr)
        fh = io.StringIO(resultStr)
        data = np.loadtxt(fh)
        result = np.zeros((4, 6), dtype=np.float32)
        result[data[:, 0].astype(int), data[:, 1].astype(int)] = data[:, 2]
        expected = np.array(
            [[1, 2, 0, 0, 0, 0],
             [0, 3, 0, 4, 0, 0],
             [0, 0, 5, 6, 7, 0],
             [0, 0, 0, 0, 0, 8]], dtype=np.float32)
        assert (result == expected).all(),\
            f"""printCrsMatrix --printCoo gives wrong view {result}. Expected:
            {expected}
            """
