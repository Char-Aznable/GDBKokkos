#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright 2020 Char Aznable <aznable.char.0083@gmail.com>
# Description: Utilities for pretty-printing Kokkos::View
#
# Distributed under terms of the 3-clause BSD license.
import re
import gdb
import argparse
import numpy as np
import pandas as pd
from GDBKokkos.utils import (
    name2type, foreachMemberOfClass,foreachBaseType,
    pointer2numpy)

class printViewMetadata(gdb.Command):
    """gdb command that prints a Kokkos::View
    """
    def __init__(self):
        # This registers the class to be used as a gdb command
        super(printViewMetadata, self).__init__("printViewMetadata", gdb.COMMAND_DATA)

    def parseArguments(self, inputArgs):
        parser = argparse.ArgumentParser()
        parser.add_argument("view", type=str,
                            help="Name of the view object")
        #parser.add_argument("ranges", type=int, nargs='*', default=[],
        #                    help="Ranges for each of the dimension. Default to\
        #                    print the entire view")
        #parser.add_argument("--noIndex", action='store_true', default=False,
        #                    help="Do not show the rank indices when printing the view")
        return parser.parse_args(inputArgs.split())
    def printKnowledge(self, view: gdb.Value):
      data = view['m_map']['m_impl_handle']
      #print(data)
    def parseRanges(self, ranges : list):
        """Convert input ranges into tuple of slices of numpy array


        Args:
            ranges (list): Input list of int as slicing index
        Returns: tuple of slice objects
        """
        arr = np.array(ranges).astype(int)
        assert (not arr.size % 2) or (not arr.size % 3),\
            "Input ranges must be a series of doublets:\
            begin0, end0, begin1,end1... or triplets:\
            begin0, end0, stride0, begin1, end1, stride1..."
        if not arr.size % 2:
            arr = arr.reshape(-1, 2)
        elif not arr.size % 3:
            arr = arr.reshape(-1, 3)
        return tuple( slice( *r ) for r in arr )

    def invoke(self, inputArgs, from_tty):
        args = self.parseArguments(inputArgs)
        v = args.view
        frame = gdb.selected_frame()
        block = frame.block()
        names = set()
        while block:
            if(block.is_global):
                #print()
                #print('global vars')
                pass
            for symbol in block:
                if (symbol.is_argument or symbol.is_variable):
                    name = symbol.name
                    if not name in names:
                        #print('{} = {}'.format(name, symbol.value(frame)))
                        names.add(name)
            block = block.superblock
        handle = gdb.parse_and_eval(args.view)['m_map']['m_impl_handle'] 
        gdb.parse_and_eval("(void)printData((void*)"+str(handle)+")")
        #view = gdb.parse_and_eval(args.view)
        #r = self.parseRanges(args.ranges)
        #arr = view2NumpyArray(view)[r]
        #if np.asarray(arr.shape).size <= 1:
        #    print(arr)
        #else:
        #    idx = pd.MultiIndex.from_product([ np.arange(dim) for dim in arr.shape[:-1] ],
        #                                     names=[f"Dim{iDim}" for iDim in range(len(arr.shape)-1)])
        #    df = pd.DataFrame(arr.reshape(-1, arr.shape[-1])).set_index(idx)
        #    with pd.option_context('display.max_seq_items', None,
        #                           'display.max_rows', 99999,
        #                           'display.max_colwidth', 2000,
        #                           ):
        #        print(df.to_string(index=not args.noIndex,
        #                           header=not args.noIndex))
