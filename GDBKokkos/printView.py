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


def getKokkosViewExtent(view : gdb.Value):
    """Get the size for all dimensions of a Kokkos::View as an numpy array


    Args:
        view (gdb.Value): Input Kokkos::View
    Returns: np.ndarray of the view's extents and np.ndarray of whether a
    dimension is dynamic (1) or static (0). In case of a scalar view, the
    extents will be an empty array
    """
    # This is the ViewDimension class as template argumnet to ViewMapping
    # whose template parameters in turn indicate the size of the static view dimension
    # or zero if it's a dynamic dimension
    viewDimName = view['m_map']['m_impl_offset'].type.template_argument(0).name
    # for the dynamic dimension, get the size from ViewDimension::D0, ::D1, ...
    extents = np.array(re.findall(r'\d+', viewDimName)).astype(int)
    isDynamic = np.zeros_like(extents).astype(int)
    # There is the special case of extents being of size 0, which is a
    # specialization for ViewDimension<>, which indicates the view is a scalar
    # view (rank 0 view). In this case, we return an array of zero size
    for iDim in range(extents.size):
        if extents[iDim] > 0:
            continue
        extents[iDim] = view['m_map']['m_impl_offset']['m_dim'][f"N{iDim}"]
        isDynamic[iDim] = 1
    return extents, isDynamic


def getKokkosViewLayout(view : gdb.Value):
    """Get a Kokkos::View's memory layout as gdb.Type


    Args:
        view (gdb.Value): Input Kokkos::View
    Returns: gdb.Type of the view's layout
    """
    View = gdb.types.get_basic_type(view.type)
    return name2type(f"{View.name}::array_layout")


def getKokkosViewStrides(view : gdb.Value):
    """Get a Kokkos::View's strides as np.ndarray

    Note that Kokkos::View's m_stride member has different meaning depending on
    the View's memory layout. This function will try to return the strides in
    the same dimensionality as the View's extents unless the View defaults to no
    stride. See
    https://github.com/kokkos/kokkos/blob/master/core/src/impl/Kokkos_ViewMapping.hpp
    for details


    Args:
        view (gdb.Value): TODO

    Returns: a np.ndarray of strides, which can be zero-sized when the view is
    a scalar view

    """

    # first get the m_stride from Kokkos::View
    mStride = None
    offset = view['m_map']['m_impl_offset']
    if 'm_stride' in { m.name for m in foreachMemberOfClass(offset.type) }:
        strides = offset['m_stride']
        Tstrides = gdb.types.get_basic_type(strides.type)
        if Tstrides.code == gdb.TYPE_CODE_INT:
            mStride = np.array([strides], dtype=int)
        elif re.match(r"^Kokkos::Impl::ViewStride<\d+>$",
                      Tstrides.name) is not None:
            mStride = np.array([ strides[f"{f.name}"]
                                for f in foreachMemberOfClass(strides.type) ],
                               dtype=int)
        else:
            raise TypeError(f"Can't get strides from type {Tstrides.name}")

    extents, _ = getKokkosViewExtent(view)
    layout = getKokkosViewLayout(view)

    # if the view doesn't have stride
    if mStride is None:

        # There is nothing we can use to figure out the strides for LayoutStride
        # when we can't access m_stride
        assert layout != "Kokkos::LayoutStride",\
            f"View {str(view)} has layout stride it has no m_stride member"

        # Here, view must be LayoutLeft or LayoutRight
        # so we can safely compute the strides for all-static rank view
        aStrides = np.ones(extents.size, dtype=int)
        if layout.name == "Kokkos::LayoutRight":
            aStrides[1:] = extents[::-1][:-1]
            aStrides = np.cumprod(aStrides)[::-1]
        elif layout.name == "Kokkos::LayoutLeft":
            aStrides[1:] = extents[:-1]
            aStrides = np.cumprod(aStrides)
        else:
            raise TypeError(f"Can't handle Kokkos::View layout {layout.name}")

        return aStrides

    # then parse the strides for each dimension
    if layout.name == "Kokkos::LayoutRight":
        assert mStride.size == 1,\
            f"Can't handle other than 1 stride in layout {layout.name}"
        # mStride is the stride of rank0, which varies the slowest
        aStrides = np.ones(extents.size, dtype=int)
        aStrides[0] = mStride[0]
        aStrides[1:-1] = np.cumprod(extents[::-1][:-2])
    elif layout.name == "Kokkos::LayoutLeft":
        assert mStride.size == 1,\
            f"Can't handle other than 1 stride in layout {layout.name}"
        # mStride is the stride of rank1
        aStrides = np.ones(extents.size, dtype=int)
        aStrides[1] = mStride[0]
        aStrides[2:] = mStride[0] * np.cumprod(extents[1:-1])
    elif layout.name == "Kokkos::LayoutStride":
        assert mStride.size == extents.size,\
            f"Strides size differs from extents size in layout {layout.name}"
        aStrides = mStride
    else:
        raise TypeError(f"Can't handle Kokkos::View layout {layout.name}")
    return aStrides

def getKokkosViewTraits(view : gdb.Value):
    """Get the base Kokkos::ViewTraits of a Kokkos::View


    Args:
        view (gdb.Value): Input Kokkos::View
    Returns: gdb.Type of the view's trait type
    """
    View = gdb.types.get_basic_type(view.type)
    Trait = next(foreachBaseType(View))
    assert Trait is not None, f"Can't find traits for view of type {View.name}"
    return Trait

def getKokkosViewValueType(view : gdb.Value):
    """Get a Kokkos::View's value type


    Args:
        view (gdb.Value): Input Kokkos::View
    Returns: gdb.Type of the view's value_type
    """
    View = gdb.types.get_basic_type(view.type)
    return name2type(f"{View.name}::value_type")

def getKokkosViewSpan(view : gdb.Value):
    """Get a Kokkos::View's span

    Args:
        view (gdb.Value): Input Kokkos::View

    Returns: span of the view as int

    """
    strides = getKokkosViewStrides(view)
    extents, _ = getKokkosViewExtent(view)
    if strides.size == 0:
        # If extents.size == 0, i.e., view is a scalar view, this will return 1,
        # according to
        # https://numpy.org/doc/stable/reference/generated/numpy.prod.html
        return np.prod(extents)
    else:
        return (extents * strides).max()


def view2NumpyArray(view : gdb.Value):
    """Create numpy array from a Kokkos::View


    Args:
        view (gdb.Value): Input Kokkos::View
    Returns: np.ndarray representing the view
    """
    View = gdb.types.get_basic_type(view.type)
    if re.match("Kokkos::DualView", View.name):
        # Always use device-side of view for debugging purpose
        view = view['d_view']
    elif re.match("Kokkos::View", View.name):
        pass
    else:
        raise RuntimeError(f"Don't know how to handle view of type {View.name}")

    data = view['m_map']['m_impl_handle']
    extents, _ = getKokkosViewExtent(view)
    strides = getKokkosViewStrides(view)
    dType = getKokkosViewValueType(view)
    span = getKokkosViewSpan(view)

    ans = pointer2numpy(data, dType, span, tuple(extents),
                        None if strides is None else tuple(strides))
    return ans


class printView(gdb.Command):
    """gdb command that prints a Kokkos::View
    """
    def __init__(self):
        # This registers the class to be used as a gdb command
        super(printView, self).__init__("printView", gdb.COMMAND_DATA)

    def parseArguments(self, inputArgs):
        parser = argparse.ArgumentParser()
        parser.add_argument("view", type=str,
                            help="Name of the view object")
        parser.add_argument("ranges", type=int, nargs='*', default=[],
                            help="Ranges for each of the dimension. Default to\
                            print the entire view")
        parser.add_argument("--noIndex", action='store_true', default=False,
                            help="Do not show the rank indices when printing\
                            the view. This will render a multidimensional array\
                            into a multi-line string in a right-layout fashion")
        return parser.parse_args(inputArgs.split())

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
        view = gdb.parse_and_eval(args.view)
        r = self.parseRanges(args.ranges)
        arr = view2NumpyArray(view)[r]
        if not args.noIndex:
            # Print the view type traits
            extents, _ = getKokkosViewExtent(view)
            strides = getKokkosViewStrides(view)
            dType = getKokkosViewValueType(view)
            span = getKokkosViewSpan(view)
            layout = getKokkosViewLayout(view)
            print(f"# span: {span}; extents: {extents}; strides: {strides}; "
                  f"layout: {layout.name}; type: {dType}\n")
        if np.asarray(arr.shape).size <= 1:
            print(arr)
        else:
            idx = pd.MultiIndex.from_product(
                [ np.arange(dim) for dim in arr.shape[:-1] ],
                names=[f"Dim{iDim}" for iDim in range(len(arr.shape)-1)])
            if arr.dtype.names is None:
                # This is n numeric array
                df = pd.DataFrame(arr.reshape(-1, arr.shape[-1])).set_index(idx)
            else:
                # This is a structured array
                df = pd.DataFrame.from_records(
                    arr.reshape(-1, arr.shape[-1]).tolist()).set_index(idx)
            with pd.option_context('display.max_seq_items', None,
                                   'display.max_rows', 99999,
                                   'display.max_colwidth', 2000,
                                   ):
                print(df.to_string(index=not args.noIndex,
                                   header=not args.noIndex))

