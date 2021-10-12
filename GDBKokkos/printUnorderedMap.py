#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright 2021 Char Aznable <aznable.char.0083@gmail.com>
# Description: Utilities for pretty-printing Kokkos::UnorderedMap
#
# Distributed under terms of the 3-clause BSD license.
import re
import gdb
import argparse
import numpy as np
import pandas as pd
import pickle
from GDBKokkos.printView import (
    view2NumpyArray, getKokkosViewValueType)


def getMapValidIndices(m : gdb.Value):
    """Get the index of valid entries in a Kokkos::UnorderedMap


    Args:
        m (gdb.Value): Input Kokkos::UnorderedMap object
    Returns: tuple of np.ndarray of the valid indices, capacity of the map
    """
    bitset = view2NumpyArray(m["m_available_indexes"]["m_blocks"])
    capacity = int(m["m_available_indexes"]["m_size"])
    ans = np.where(
        np.unpackbits(bitset.view(np.uint8), bitorder="little"))[0]
    return ans, capacity


def getMapKeysVals(m : gdb.Value):
    """Get Kokkos::UnorderedMap's valid keys and values as numpy array


    Args:
        m (gdb.Value): Input Kokkos::UnorderedMap object
    Returns: tuple of: the keys and values as numpy array and the capacity
    of the map
    """
    mKeys = m["m_keys"]
    mVals = m["m_values"]
    keys = view2NumpyArray(mKeys)
    vals = view2NumpyArray(mVals)

    ids, capacity = getMapValidIndices(m)
    keysValid = keys[ids]
    valsValid = vals[ids]
    return keysValid, valsValid, capacity


class printUnorderedMap(gdb.Command):
    """gdb command that prints a Kokkos::UnorderedMap
    """
    def __init__(self):
        # This registers the class to be used as a gdb command
        super(printUnorderedMap, self).__init__("printUnorderedMap", gdb.COMMAND_DATA)

    def parseArguments(self, inputArgs):
        parser = argparse.ArgumentParser()
        parser.add_argument("map", type=str,
                            help="Name of the UnorderedMap object")
        parser.add_argument("--noIndex", action='store_true', default=False,
                            help="Do not show the pandas.DataFrame header")
        return parser.parse_args(inputArgs.split())

    def invoke(self, inputArgs, from_tty):
        args = self.parseArguments(inputArgs)
        m = gdb.parse_and_eval(args.map)
        # Get the keys and vals
        keys, vals, _ = getMapKeysVals(m)
        # Convert to dataframe
        # NOTE: as of Pandas version 1.2.4, Python 3.8.10 and GDB 9.2, passing
        # pd.DataFrame from python session to a gdb.Command results in garbage
        # pd.DataFrame, e.g., as if some columns are garbage collected. So we
        # have to pass the keys and vals and create the pd.DataFrame here
        mKeys = m["m_keys"]
        mVals = m["m_values"]
        TKey = getKokkosViewValueType(mKeys)
        TVal = getKokkosViewValueType(mVals)
        df = pd.DataFrame({str(TKey) : keys, str(TVal) : vals})
        with pd.option_context('display.max_seq_items', None,
                               'display.max_rows', 99999,
                               'display.max_colwidth', 2000,
                               ):
            print(df.to_string(index=not args.noIndex,
                               header=not args.noIndex))
