#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright 2020 Char Aznable <aznable.char.0083@gmail.com>
# Description: Utilities for pretty-printing Kokkos::View
#
# Distributed under terms of the 3-clause BSD license.
import numpy as np
import gdb

def name2type(name : str):
    """Get gdb.Type from a string of the type name


    Args:
        name (str): Input name of the C++ type
    Returns:  gdb.Type for the requested type
    """
    symbolTuple = gdb.lookup_symbol(name)
    assert symbolTuple[0] is not None and not symbolTuple[1], f"{name} is not a type"
    t = gdb.types.get_basic_type(symbolTuple[0].type)
    return t


def foreachMemberOfClass(T : gdb.Type):
    """Generator to iterate through member of a class

    Args:
        T (gdb.Type): Input type

    Yields: field from T that is not is_base_class

    """
    for field in T.fields():
        if not field.is_base_class:
            yield field


def foreachBaseType(T : gdb.Type):
    """Generator to iterate through the base types of a type

    Args:
        T (gdb.Type): Input type

    Yields: field from T that is is_base_class

    """
    for field in T.fields():
        if field.is_base_class:
            yield field.type


def pointer2numpy(addr : int, t : gdb.Type, span : int, shape : tuple,
                  strides : tuple = None):
    """Create a numpy array from given memory address, type and span

    Args:
        addr (int): The address in the C program's memory space
        t (gdb.Type): The C-type of the array element
        span (int): The total span of the array in the unit of number of
        elements.
        shape (tuple): The shape of the output array
        strides (tuple) : strides of each dimension of the array in the unit of
        number of elements

    Returns: np.ndarray representing the given memory chunk

    """

    # Get the underlying type, removing typdefs
    T = gdb.types.get_basic_type(t)

    sizeT = T.sizeof
    isSigned = re.search(r"unsigned", T.name) is None if t.code == gdb.TYPE_CODE_INT else None

    typeCode2TypeStr = {
        gdb.TYPE_CODE_INT : f"|{'i' if isSigned else 'u'}{sizeT}",
        gdb.TYPE_CODE_FLT : f"|f{sizeT}",
        gdb.TYPE_CODE_BOOL : f"|b{sizeT}",
        }

    typestr = typeCode2TypeStr[T.code]


    # create a gdb.Value representing the array, i.e., arr.type == T[span]
    # TODO: to support CUDA global memory space array, use something like:
    # arr = gdb.parse_and_eval(f"*((@global {T.name} *){addr})@{span}")
    arr = gdb.parse_and_eval(f"*(({T.name} *){addr})@{span}")
    # copy the array over
    # This is equivlant to 
    # aTmp = np.frombuffer(
    #     bytearray(gdb.selected_inferior().read_memory(addr, sizeTotal)),
    #     dtype=typestr)
    aTmp = np.array([arr[i] for i in range(span)], dtype=typestr)

    arrayAPIDict = aTmp.__array_interface__.copy()
    arrayAPIDict['shape'] = shape
    if strides is not None:
        aStrides = np.asarray(strides) * sizeT
        arrayAPIDict['strides'] = tuple(aStrides)

    class ArrayHolder():
        def __init__(self):
            self.__array_interface__ = arrayAPIDict

    holder = ArrayHolder()
    return np.array(holder, copy=True)
