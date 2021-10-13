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
from collections.abc import Iterable
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


def getMapKeysVals(m : gdb.Value, depthMax : int = 3):
    """Get Kokkos::UnorderedMap's valid keys and values as numpy array


    Args:
        m (gdb.Value): Input Kokkos::UnorderedMap object
        depthMax (int): Maximal allowed recursion depth into the nested struct.
        Any struct or class below this level will be treated as byte string of
        the same size as the struct
    Returns: tuple of: the keys and values as numpy array and the capacity
    of the map
    """
    mKeys = m["m_keys"]
    mVals = m["m_values"]
    keys = view2NumpyArray(mKeys, depthMax)
    vals = view2NumpyArray(mVals, depthMax)

    ids, capacity = getMapValidIndices(m)
    keysValid = keys[ids]
    valsValid = vals[ids]
    return keysValid, valsValid, capacity

def removeByTypesFromNested(l, typesRemove : tuple):
    """Remove items in the input iterable l which are of one of the types in
    typesRemove


    Args:
        l : Input iterable
        typesRemove (tuple): Input types to be removed
    Returns: Iteratable with all nested items of the types removed
    """
    # remove all the bytes
    l = filter(lambda i : not isinstance(i, typesRemove), l)
    # recurse into the sublist
    l = [ removeByTypesFromNested(i, typesRemove) if isinstance(i, (list, tuple))
         else i for i in l  ]
    # clean up the empty sublists
    l = [ i for i in l if not isinstance(i, (list, tuple)) or len(i) > 0 ]
    return l


def flattenNested(l):
    """Flatten a nested iterable to a list

    Element of the type np.void is also considered iterable due to how GDBKokkos
    convert view value type to numpy structured array so np.void object will be
    converted to tuple and flattened


    Args:
        l : Input iterable
    Returns: list of elements with no nesting iterable
    """
    ans = []
    for i in l:
        if isinstance(i, Iterable):
            ans.extend(flattenNested(i))
        elif isinstance(i, np.void):
            ans.extend(flattenNested(tuple(i)))
        else:
            ans.append(i)
    return ans


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
        parser.add_argument("--hideTypes", type=str, nargs='*', default=[],
                            help="When keys or values of the map have deep\
                            nested structure, GDBKokkos won't go all the way\
                            to pick up all the underlying data type to render\
                            but rather just cast anything below a certain\
                            nesting depth level, which is currently 3, to byte\
                            string. In those cases, the byte string makes\
                            rendering difficult so one can remove\
                            the byte string before printing keys or values by\
                            passing to 'bytes' to this option")
        parser.add_argument("--flatten", action='store_true', default=False,
                            help="Flatten the nested struct of keys and vals")
        parser.add_argument("--sortKeys", action='store_true', default=False,
                            help="Sort the keys lexicographically. This only\
                            works if '--hideTypes bytes' and '--flatten' are\
                            given")
        parser.add_argument("--depthMax", type=int, default=3,
                            help="Maximal allowed recursion depth into the\
                            nested struct of view value type. Any struct or\
                            class below this level will be treated as byte \
                            string of the same size as the struct")
        return parser.parse_args(inputArgs.split())

    def invoke(self, inputArgs, from_tty):
        args = self.parseArguments(inputArgs)
        m = gdb.parse_and_eval(args.map)
        # Get the keys and vals
        keys, vals, _ = getMapKeysVals(m, args.depthMax)

        # Remove unwanted types
        typesRemove = tuple([ eval(t) for t in args.hideTypes ])
        if len(typesRemove):
            keys = removeByTypesFromNested(keys.tolist(), typesRemove)
            vals = removeByTypesFromNested(vals.tolist(), typesRemove)

        if args.flatten:
            n = len(keys)
            keys = np.array(flattenNested(keys)).reshape(n, -1).tolist()
            vals = np.array(flattenNested(vals)).reshape(n, -1).tolist()

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

        if "bytes" in args.hideTypes and args.flatten and args.sortKeys:
            df = df.sort_values(by=[str(TKey)])

        with pd.option_context('display.max_seq_items', None,
                               'display.max_rows', 99999,
                               'display.max_colwidth', 2000,
                               ):
            print(df.to_string(index=not args.noIndex,
                               header=not args.noIndex))
