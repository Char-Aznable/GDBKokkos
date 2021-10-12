#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright 2021 Char Aznable <aznable.char.0083@gmail.com>
# Description: Test utils in GDBKokkos.printUnorderedMap
#
# Distributed under terms of the MIT license.
import re
import numpy as np
import ast

cpp = f"""
    #include <Kokkos_Core.hpp>
    #include <Kokkos_UnorderedMap.hpp>

    void breakpoint() {{ return; }}

    struct Key {{ int k_[4]; }};
    struct Value {{ int v_[2]; }};

    using Map = Kokkos::UnorderedMap<Key, Value>;

    int main(int argc, char* argv[]) {{
      Kokkos::initialize(argc,argv); {{

      Map m(100);

      Kokkos::parallel_for(10, KOKKOS_LAMBDA (const int i)
        {{ m.insert(Key{{i, i, i, i}}, Value{{i + 1, i + 1}}); }});

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

def testGetMapValidIndices(compileTestUnorderedMap, runGDB):
    p, cppSources = compileTestUnorderedMap
    for _, row in cppSources.iterrows():
        fTest = row["fTest"]
        fGDB = p.join(fTest.purebasename + "_TestGetMapValidIndices.gdbinit")
        print(f"fTest = {fTest} fGDB = {fGDB}\n")
        r, resultStr = runGDB(
            """
            py from GDBKokkos.printUnorderedMap import getMapValidIndices
            py m = gdb.parse_and_eval("m")
            py print(getMapValidIndices(m)[0])

            """,
            fGDB, f"{fTest.realpath().strpath}"
            )
        resultStr = re.sub(r"[\[\]]", "", resultStr)
        result = np.array(resultStr.split(), dtype=np.int32)
        assert result.size == 10,\
            f"Wrong number of valid indices {result.size}: {r.stdout}"
        assert (result < 128).all(),\
            f"Number of valid indices exceeds map capacity: {r.stdout}"


def parsePrintUnorderedMapOutput(s : str):
    s = re.sub(r"\n", " ", s)
    s = re.sub(r"^\s+", "", s)
    s = re.sub(r"\)\s+\(", "), (", s)
    l = list(ast.literal_eval(s))
    ans = []
    for i in range(len(l)):
        if i % 2:
            # odd index is value
            iKey = i - 1
            iVal = i
        else:
            # odd index is value
            iKey = i
            iVal = i + 1
        val = np.array(l[iVal][0])
        key = np.array(l[iKey][0])
        ans.append((key, val))
    return ans



def testGetMapKeysVals(compileTestUnorderedMap, runGDB):
    p, cppSources = compileTestUnorderedMap
    for _, row in cppSources.iterrows():
        fTest = row["fTest"]
        fGDB = p.join(fTest.purebasename + "_TestGetMapKeysVals.gdbinit")
        print(f"fTest = {fTest} fGDB = {fGDB}\n")
        r, resultStr = runGDB(
            """
            py from GDBKokkos.printUnorderedMap import getMapKeysVals
            py import pandas as pd
            py m = gdb.parse_and_eval("m")
            py keys, vals, _ = getMapKeysVals(m)
            py ans = pd.DataFrame({"Key" : keys, "Value" : vals})
            py print(ans.to_string(index=False,header=False))

            """,
            fGDB, f"{fTest.realpath().strpath}"
            )
        d = parsePrintUnorderedMapOutput(resultStr)
        for key, val in d:
            print(f"{key} -> {val}\n")
            assert (val == val[0]).all(), f"Wrong values: {r.stdout}"
            assert (key == key[0]).all(), f"Wrong keys: {r.stdout}"
            assert val[0] == key[0] + 1, f"Wrong keys - vals: {r.stdout}"


def testPrintUnorderedMap(compileTestUnorderedMap, runGDB):
    p, cppSources = compileTestUnorderedMap
    for _, row in cppSources.iterrows():
        fTest = row["fTest"]
        fGDB = p.join(fTest.purebasename + "_TestUnorderedMap.gdbinit")
        print(f"fTest = {fTest} fGDB = {fGDB}\n")
        r, resultStr = runGDB(
            """
            py import GDBKokkos
            py m = gdb.parse_and_eval("m")
            printUnorderedMap m --noIndex

            """,
            fGDB, f"{fTest.realpath().strpath}"
            )
        d = parsePrintUnorderedMapOutput(resultStr)
        for key, val in d:
            print(f"{key} -> {val}\n")
            assert (val == val[0]).all(), f"Wrong values: {r.stdout}"
            assert (key == key[0]).all(), f"Wrong keys: {r.stdout}"
            assert val[0] == key[0] + 1, f"Wrong keys - vals: {r.stdout}"
