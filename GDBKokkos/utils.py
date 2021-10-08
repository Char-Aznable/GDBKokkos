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
import re

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

def parseGDBArrayType(T : gdb.Type):
    """Parse an input GDB array type into the basic element type and span


    Args:
        T (gdb.Type): Input type
    Returns: tuple (gdb.Type for the element, span of the array)
    """
    assert T.code == gdb.TYPE_CODE_ARRAY, f"Type {str(T)} is not array"
    m = re.match(r'(\S+)\s*\[(\d+)\]', str(T))
    assert m is not None, f"Can't parse type {str(T)} to array"
    Element = name2type(m.group(1))
    span = int(m.group(2))
    return (Element, span)


def arithmeticType2NumpyDtype(t : gdb.Type):
    """parse basic arithmetic type into np.dtype string


    Args:
        t (gdb.Type): Input type
    Returns: a tuple (element type, span) where span can be the extent of the
    input array type or None if input is scalar type
    """

    T = gdb.types.get_basic_type(t)

    assert T.code != gdb.TYPE_CODE_STRUCT, f"Input {str(t)} is not arithmetic type"

    # parse array
    span = None
    if T.code == gdb.TYPE_CODE_ARRAY:
        T, span = parseGDBArrayType(T)

    sizeT = T.sizeof
    isSigned = re.search(r"unsigned", T.name) is None if T.code == gdb.TYPE_CODE_INT else None

    typeCode2TypeStr = {
        gdb.TYPE_CODE_INT : f"|{'i' if isSigned else 'u'}{sizeT}",
        gdb.TYPE_CODE_FLT : f"|f{sizeT}",
        gdb.TYPE_CODE_BOOL : f"|b{sizeT}",
        }

    return (typeCode2TypeStr[T.code], span)


def struct2dtype(T : gdb.Type):
    """Convert a gdb.Type to a list of np.dtype with potentially nested struct

    Nested struct are parsed as what would appear in a numpy structured array
    with the output name prefixed with outter class scope name

    Args:
        T (gdb.Type): Input type to be parsed
    Returns: list [name, np.dtype]
    """
    assert T.code == gdb.TYPE_CODE_STRUCT, f"Input type {str(T)} is not struct"

    ans = []
    for f in foreachMemberOfClass(T):
        name = f.name
        t = f.type
        if t.code == gdb.TYPE_CODE_STRUCT:
            ans.append((name, struct2dtype(t)))
        else:
            dtype, span = arithmeticType2NumpyDtype(t)
            ans.append((name, dtype) if span is None else (name, dtype, span))
    return ans


def type2dtype(T : gdb.Type):
    """Convert input gdb.Type to np.dtype

    For simple scalar type, this will return a single string of np.dtype

    For C array of struct (potentially nested), it will output a list of
    np.dtype fields with the name of each field being the struct member variable
    (potentially prefixed with outter class name if nesting). Array type will be
    suffixed with the span of the array

    Args:
        T (gdb.Type): Input type
    Returns: str or list of np.dtype
    """

    #TODO: convert structed array to pandas dataframe:
    # pd.DataFrame.from_records(x.tolist(), columns=x.dtype.names)

    dtypes = None
    if T.code == gdb.TYPE_CODE_STRUCT:
        # this is a struct, potentially nested. we first flatten out the nested
        # structs and create a numpy structured array using the fields in the
        # flatten struct
        dtypes = struct2dtype(T)
    else:
        dtype, span = arithmeticType2NumpyDtype(T)
        if span is None:
            # this is a scalar type so we use dtypes to create a normal numpy
            # array
            dtypes = dtype
        else:
            # this is an array type so we use dtypes to create a numpy
            # structured array
            dtypes = [(T.name, dtype, span)]
    return dtypes


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
    sizeTotal = span * sizeT

    typestr = type2dtype(T)
    dtype = np.dtype(typestr, align=True)

    # in case the alignment assumption failed, e.g., due to the compiler, check
    # if the size of the type is consistent
    assert dtype.itemsize == sizeT,\
        f"np.dtype reports type {str(T)} has size {dtype.itemsize} but its "\
        "actual size is {sizeT}. It's likely that the alignment assumption "\
        "of np.dtype failed."

    # create a gdb.Value representing the array, i.e., arr.type == T[span]
    # arr = gdb.parse_and_eval(f"*(({T.name} *){addr})@{span}")
    # TODO: to support CUDA global memory space array, use something like:
    # arr = gdb.parse_and_eval(f"*((@global {T.name} *){addr})@{span}")
    # or create the buffer with
    # buffer = gdb.selected_inferior().read_memory(f"(@global {T.name} *){addr}", sizeTotal)
    buffer = gdb.selected_inferior().read_memory(addr, sizeTotal)
    # copy the array over
    # This is equivlant to 
    # aTmp = np.array([arr[i] for i in range(span)], dtype=dtype)
    # for simple dtype. For more complex type with nested structure, the
    # element-wise copying would involve nested loops and is less favored to
    # direct byte copying here
    # TODO: render the memory directly into a ndarray view so that we don't have
    # to copy the content over
    aTmp = np.frombuffer(buffer, dtype=dtype)

    # Convert the span into np.ndarray taking into account strides
    arrayAPIDict = aTmp.__array_interface__.copy()
    arrayAPIDict['shape'] = shape
    # when there exist padding bytes resulting from alignment, e.g., in the case
    # of T is some struct, arrayAPIDict.descr can have fields corresponding to
    # the padded bytes shown as something like ('', '|V3'), which will mess up
    # the final array rendering. The simple solution to remove these padded
    # bytes is to recycle the typestr here
    arrayAPIDict['descr'] = dtype
    if strides is not None:
        aStrides = np.asarray(strides) * sizeT
        arrayAPIDict['strides'] = tuple(aStrides)

    class ArrayHolder():
        def __init__(self):
            self.__array_interface__ = arrayAPIDict

    holder = ArrayHolder()
    # The buffer will be garbage collected once we leave the scope so we need to
    # copy the content over
    return np.array(holder, copy=True)
