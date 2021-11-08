#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright 2021 Char Aznable <aznable.char.0083@gmail.com>
# Description: Utilities for pretty-printing KokkosSparse::CrsMatrix
#
# Distributed under terms of the 3-clause BSD license.
import gdb
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sps
from collections.abc import Iterable
from GDBKokkos.printView import (
    view2NumpyArray, getKokkosViewValueType)


def crs2ScipySparse(m : gdb.Value, values : np.ndarray = None):
    """Convert a KokkosSparse::CrsMatrix object to scipy.sparse.csr_matrix

    Args:
        m (gdb.Value): Input KokkosSparse::CrsMatrix
        values (np.ndarray): Use these as the values of the output
        scipy.sparse.crs_matrix instead of m.values

    Returns: scipy.sparse.crs_matrix

    """
    rowMap = view2NumpyArray(m['graph']['row_map'])
    colIdx = view2NumpyArray(m['graph']['entries'])
    if values is None:
        values = view2NumpyArray(m['values'])
    else:
        assert values.ndim == 1, f"Input values {values} is not 1-d array"
        assert values.size == colIdx.size,\
            f"Input values {values} has different size than m.graph.entries"
    nCols = m['numCols_']
    nRows = rowMap.size - 1
    return sps.csr_matrix((values, colIdx, rowMap), shape=(nRows, nCols))


class printCrsMatrix(gdb.Command):
    """gdb command that prints a KokkosSparse::CrsMatrix
    """
    def __init__(self):
        # This registers the class to be used as a gdb command
        super(printCrsMatrix, self).__init__("printCrsMatrix", gdb.COMMAND_DATA)

    def parseArguments(self, inputArgs):
        parser = argparse.ArgumentParser()
        parser.add_argument("matrix", type=str,
                            help="Name of the CrsMatrix object. Only basic\
                            numeric value types are supported in the matrix's\
                            values view array")
        parser.add_argument("--printCoo", action='store_true', default=False,
                            help="Print the CrsMatrix in COO format. If not\
                            set, will print as np.ndarray.")
        parser.add_argument("--values", type=str, default=None,
                            help="Print the matrix as if its values are\
                            replaced by those in this input view. This is\
                            useful for printing the result of an sparse\
                            matrix algorithm which updates the content\
                            of the matrix to this values")
        return parser.parse_args(inputArgs.split())

    def invoke(self, inputArgs, from_tty):
        args = self.parseArguments(inputArgs)
        m = gdb.parse_and_eval(args.matrix)
        values = view2NumpyArray(m['values']) if args.values is None else\
            view2NumpyArray(gdb.parse_and_eval(args.values))
        rowMap = view2NumpyArray(m['graph']['row_map'])
        colIdx = view2NumpyArray(m['graph']['entries'])
        nCols = m['numCols_']
        nRows = rowMap.size - 1
        mSparse = sps.csr_matrix((values, colIdx, rowMap), shape=(nRows, nCols))
        print(f"# shape: {mSparse.shape}\n")
        if(args.printCoo):
            print(mSparse)
        else:
            # Note: the user can adjust the numpy array print settings with:
            # py np.set_printoptions(edgeitems=30, linewidth=100000)
            # before calling this command
            print(mSparse.toarray())
