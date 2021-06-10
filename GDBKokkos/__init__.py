#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright 2020 Char Aznable <aznable.char.0083@gmail.com>
# Description: Utilities for pretty-printing Kokkos::View
#
# Distributed under terms of the 3-clause BSD license.
import gdb
from GDBKokkos.printView import(
    printView
    )
from GDBKokkos.printViewMetadata import(
    printViewMetadata
    )

# This registers our class to the gdb runtime at "source" time.
printView()
printViewMetadata()
