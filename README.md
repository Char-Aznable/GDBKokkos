# GDBKokkos

[![Build Status](https://github.com/Char-Aznable/GDBKokkos/actions/workflows/test_and_release.yml/badge.svg)](https://github.com/Char-Aznable/GDBKokkos/actions)
[![License: 3-clause BSD](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![PyPI version](https://badge.fury.io/py/GDBKokkos.svg)](https://badge.fury.io/py/GDBKokkos)
[![Python version](https://img.shields.io/badge/python-3.8.10+-blue.svg)](https://www.python.org/downloads/release/python-3810/)

Pretty printer for debugging Kokkos with the GNU Debugger (GDB)

This is a python module to be used in GDB to examine various data structure of
[`Kokkos`](https://github.com/kokkos/kokkos) and
[`Kokkos Kernels`](https://github.com/kokkos/kokkos-kernels).
The following data structure can be pretty-printed in GDB via commands (see
below) :

- [Kokkos::View](https://github.com/kokkos/kokkos/wiki/View)
    - Can pretty-print view content and traits including memory layout, stride and extents
    - User-defined value types with nested structure are supported
- [Kokkos::UnorderedMap](https://github.com/kokkos/kokkos/wiki/Unordered-Map)
    - Can pretty-print key-value pairs
- [KokkosSparse::CrsMatrix](https://github.com/kokkos/kokkos-kernels/wiki/CrsMatrix)
    - Can pretty-print the sparse matrix in dense or COO format

They can also be converted to the respective python data structure objects for
interoperation, which can be useful for quickly verifying numerical results.

## Usage

### Install the package 

GDB is usually built by linking to a specific version of python that it uses
during its debugging session. One needs to make sure that GDBKokkos is installed
using the same python version as the one GDB is linked to. There are a couple of
ways to do this. The easiest way I found is using conda to set up a conda
environment with GDB installed. This repo has a
[environment.yml](https://github.com/Char-Aznable/GDBKokkos/blob/master/environment.yml)
file that can be used to set up such conda environement. Basically, download the
environement.yml and run
```
conda env create -f environment.yml
```
which will set up a conda environment called GDBKokkos.  Then you can install
GDBKokkos by first activating that environement:
```
conda activate GDBKokkos
```
and then pip install it:
```
pip install GDBKokkos
```

It's also possible to directly `pip install GDBKokkos` if the user knows for
sure that it will install with the python that GDB will use in a debug session.

### Import module in GDB

To use this module in GDB, consider this simple Kokkos example code:

```c++
#include <Kokkos_Core.hpp>

void breakpoint() { return; }

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc,argv); {

  Kokkos::View<float***, Kokkos::LayoutRight> v("v", 3, 4, 5);

  Kokkos::parallel_for(v.span(), KOKKOS_LAMBDA (const int i) { *(v.data()+i) = i; });

  auto vSub = Kokkos::subview(v, Kokkos::ALL(), Kokkos::ALL(), 3);

  breakpoint();

  } Kokkos::finalize();

}
```
Build this as `test` (see https://github.com/kokkos/kokkos/wiki/Compiling for instruction) and load this into GDB using `gdb ./test`. Then run

```gdb
(gdb) py import GDBKokkos
```

to import the module. 

### Use the pretty printer of Kokkos::View

#### Print the entire view

Then you can use the `printView` GDB command to print a view. For example:

```gdb
# set break point
(gdb) b test.cpp:14
# run to stop at the break point
(gdb) r
# print the entire views 
(gdb) printView v 
```
This will shows the content of `v`:
```
              0     1     2     3     4
Dim0 Dim1                              
0    0      0.0   1.0   2.0   3.0   4.0
     1      5.0   6.0   7.0   8.0   9.0
     2     10.0  11.0  12.0  13.0  14.0
     3     15.0  16.0  17.0  18.0  19.0
1    0     20.0  21.0  22.0  23.0  24.0
     1     25.0  26.0  27.0  28.0  29.0
     2     30.0  31.0  32.0  33.0  34.0
     3     35.0  36.0  37.0  38.0  39.0
2    0     40.0  41.0  42.0  43.0  44.0
     1     45.0  46.0  47.0  48.0  49.0
     2     50.0  51.0  52.0  53.0  54.0
     3     55.0  56.0  57.0  58.0  59.0

```
where the first row is the index of the last dimension of `v` and the first two
columns show the indices of the first two dimension. You can also ask
`printView` to print the type traits of the view such as its span using:
```gdb
(gdb) printView v --viewTraits
# span: 60; extents: [3 4 5]; strides: [20  5  1]; layout: Kokkos::LayoutRight; type: float

              0     1     2     3     4
Dim0 Dim1                              
0    0      0.0   1.0   2.0   3.0   4.0
     1      5.0   6.0   7.0   8.0   9.0
     2     10.0  11.0  12.0  13.0  14.0
     3     15.0  16.0  17.0  18.0  19.0
1    0     20.0  21.0  22.0  23.0  24.0
     1     25.0  26.0  27.0  28.0  29.0
     2     30.0  31.0  32.0  33.0  34.0
     3     35.0  36.0  37.0  38.0  39.0
2    0     40.0  41.0  42.0  43.0  44.0
     1     45.0  46.0  47.0  48.0  49.0
     2     50.0  51.0  52.0  53.0  54.0
     3     55.0  56.0  57.0  58.0  59.0

```

You can disable showing the indices using

```
(gdb) printView v --noIndex
```
to see just the view data:
```
0.0  1.0  2.0  3.0  4.0
5.0  6.0  7.0  8.0  9.0
10.0 11.0 12.0 13.0 14.0
15.0 16.0 17.0 18.0 19.0
20.0 21.0 22.0 23.0 24.0
25.0 26.0 27.0 28.0 29.0
30.0 31.0 32.0 33.0 34.0
35.0 36.0 37.0 38.0 39.0
40.0 41.0 42.0 43.0 44.0
45.0 46.0 47.0 48.0 49.0
50.0 51.0 52.0 53.0 54.0
55.0 56.0 57.0 58.0 59.0

```

#### Print the sliced view

To see the sliced view `v[0:2, 0:4, 3:4]` (in Numpy notation), supply a set of doublets of indices for each dimension:

```gdb
(gdb) printView v 0 2 0 4 3 4
```
to get:
```
              0
Dim0 Dim1      
0    0      3.0
     1      8.0
     2     13.0
     3     18.0
1    0     23.0
     1     28.0
     2     33.0
     3     38.0

```
which is the same data content as the subview `vSub` (except for difference in numer of dimensions):
```
(gdb) printView vSub
         0     1     2     3
Dim0                        
0      3.0   8.0  13.0  18.0
1     23.0  28.0  33.0  38.0
2     43.0  48.0  53.0  58.0

```

You can also use triplet of indices to provide stride for each dimension. For example, to see `v[0:2, 0:4:2, 0:5:2]` (in Numpy notation):

```gdb
(gdb) printView v 0 2 1 0 4 2 0 5 2
```

to get:
```
              0     1     2
Dim0 Dim1                  
0    0      0.0   2.0   4.0
     1     10.0  12.0  14.0
1    0     20.0  22.0  24.0
     1     30.0  32.0  34.0
```

Note that if you need the stride for any dimension, you have to supply strides for all the other dimensions even though they would be `1`.

You can see the options description for `printView` by

```gdb
(gdb) printView -h
usage: [-h] [--viewTraits] [--noIndex] view [ranges [ranges ...]]

positional arguments:
  view          Name of the view object
  ranges        Ranges for each of the dimension. Default to print the entire view

optional arguments:
  -h, --help    show this help message and exit
  --viewTraits  Print the span, extents, shape and value type of the view
  --noIndex     Do not show the rank indices when printing the view. This will render a multidimensional array into a multi-line string in a right-layout fashion

```

### <a name="CustomType"></a> Support for user-defined view value type

The pretty printer supports printing view of user-defined class or struct. For
example, 

```c++
#include <Kokkos_Core.hpp>

struct A {
  long i_[2];
  bool b_[2];
  float f_[2];
};

struct T {
  KOKKOS_INLINE_FUNCTION T() = default;
  KOKKOS_INLINE_FUNCTION T(const int i)
    :a_{{i,i}, {i > 0, i > 0}, {i / 1.f, i / 1.f}}
    ,i_{i}
    ,d_{i / 1.}
    {}

  A a_;
  int i_;
  double d_;
};

int main(int argc, char* argv[]) {
Kokkos::initialize(argc,argv); {

  Kokkos::View<T**,Kokkos::LayoutLeft, Kokkos::HostSpace> v("v", 2, 3);
  for(int i = 0; i < v.extent_int(0); ++i) {
    for(int j = 0; j < v.extent_int(1); ++j) {
      v(i, j) = T{i+j};
    }
  }

} Kokkos::finalize(); }
```

```gdb
(gdb) printView v

                                                   0                                             1                                             2
Dim0
0     (([0, 0], [False, False], [0.0, 0.0]), 0, 0.0)  (([1, 1], [True, True], [1.0, 1.0]), 1, 1.0)  (([2, 2], [True, True], [2.0, 2.0]), 2, 2.0)
1       (([1, 1], [True, True], [1.0, 1.0]), 1, 1.0)  (([2, 2], [True, True], [2.0, 2.0]), 2, 2.0)  (([3, 3], [True, True], [3.0, 3.0]), 3, 3.0)
```

where the view is rendered as a [numpy structured
array](https://numpy.org/doc/stable/user/basics.rec.html#structured-arrays). For
nested struct or class as in the example, each level of nesting class will be
represented as a level of tuple parenthesis. Array member of a class will be
rendered as python list with brackets. However, this is a default limit on how
deep the recursion will go to pick up the nested struct in a view value type,
which can be control by the `--depthMax` option of `printView`, which defaults
to 3. Any struct nested below `depthMax` level will be rendered as byte string
of the same size as the struct.

## Interoperation with python

Aside from pretty printing, one can also copy a view into a numpy array in
python. This comes in handy when the user wants to quickly verify certain aspect
of a view such as its max or sum with the help of numpy functions. For the
example in the [previous section](#-support-for-user-defined-view-value-type),
one can copy the view to a numpy array using:

```gdb
(gdb) py from GDBKokkos.printView import view2NumpyArray
(gdb) py v = gdb.parse_and_eval('v')
(gdb) py varr = view2NumpyArray(v)
(gdb) py print(varr)
[[(([0, 0], [False, False], [0., 0.]), 0, 0.)
  (([1, 1], [ True,  True], [1., 1.]), 1, 1.)
  (([2, 2], [ True,  True], [2., 2.]), 2, 2.)]
 [(([1, 1], [ True,  True], [1., 1.]), 1, 1.)
  (([2, 2], [ True,  True], [2., 2.]), 2, 2.)
  (([3, 3], [ True,  True], [3., 3.]), 3, 3.)]]
```

To get the max over the class member `i_`, one can do:

```gdb
(gdb) py print(varr['i_'].max())
3
```

### Get properties of Kokkos::View

To use the utility python functions to get various properties of Kokkos::View, load the functions with:

```gdb
(gdb) py from GDBKokkos.printView import *
```

To get the layout, extent, strides and span of the Kokkos::View:

```gdb
(gdb) py v = gdb.parse_and_eval('v')
(gdb) py print( getKokkosViewLayout(v) )
Kokkos::LayoutRight
(gdb) py print( getKokkosViewExtent(v) )
[3 4 5]
(gdb) py print( getKokkosViewStrides(v) )
[20  5  1]
(gdb) py print( getKokkosViewSpan(v) )
60
```

## Support for Kokkos::UnorderedMap

It's possible to pretty print the content of a Kokkos::UnorderedMap using the
gdb command `printUnorderedMap`. For example:

```c++
#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;

static constexpr std::size_t kokkosAlignmentBytes = 8;

struct alignas(kokkosAlignmentBytes) Key {
  int k_[2];
};

struct alignas(kokkosAlignmentBytes) Value {
  int v_[2];
};

int main(int argc, char* argv[]) {
Kokkos::initialize(argc,argv); {

  using Map = Kokkos::UnorderedMap<Key,Value,ExecutionSpace>;

  Map m(10);
  Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace>(0, 10),
    KOKKOS_LAMBDA(const int i) {
      const Key k{i / 2, i / 2};
      Value v{-1, -1};
      if(i % 2) {
        v.v_[1] = i;
      } else {
        v.v_[0] = i;
      }
      const auto r = m.insert(k, v);
      if(r.existing()) {
        const auto iValue = r.index();
        if(i % 2) {
          m.value_at(iValue).v_[1] = i;
        } else {
          m.value_at(iValue).v_[0] = i;
        }
      }
    });
  Kokkos::fence();

  std::cout << "\n";

} Kokkos::finalize(); }
```

```gdb
(gdb) py import GDBKokkos
(gdb) printUnorderedMap m
         Key      Value
0  ([3, 3],)  ([6, 7],)
1  ([2, 2],)  ([4, 5],)
2  ([0, 0],)  ([0, 1],)
3  ([4, 4],)  ([8, 9],)
4  ([1, 1],)  ([2, 3],)
```

For complicated key and value type with nested structure, the option
`--flatten` can flatten out the nested structure:

```gdb
(gdb) py import GDBKokkos
(gdb) printUnorderedMap m --flatten
      Key   Value
0  [3, 3]  [6, 7]
1  [2, 2]  [4, 5]
2  [0, 0]  [0, 1]
3  [4, 4]  [8, 9]
4  [1, 1]  [2, 3]

```
Another option `--hideTypes` can take a list of python type string so that
element of the key and value that is instance of these types in the list will
be removed. For example `printUnorderedMap --hideTypes bytes bool` will remove
the byte string and boolean members from any nested struct in the key and value
type. This helps the user to focus on certain aspect of the keys and values by
ignoring auxiliary information. For a complete list of command line options, use
`printUnorderedMap -h`.

The option `--sortKeys` can sort the output by the keys in the map, provided
that the options `--hideTypes bytes` and `--flatten` are given:
```gdb
(gdb) py import GDBKokkos
(gdb) printUnorderedMap m --flatten --hideTypes bytes --sortKeys 
      Key   Value
2  [0, 0]  [0, 1]
4  [1, 1]  [2, 3]
1  [2, 2]  [4, 5]
0  [3, 3]  [6, 7]
3  [4, 4]  [8, 9]
```

One can also get the keys and values from the UnorderedMap as numpy array using:

```gdb
(gdb) py from GDBKokkos.printUnorderedMap import getMapKeysVals
(gdb) py m = gdb.parse_and_eval("m")
(gdb) py keys, vals, capacity = getMapKeysVals(m)
(gdb) py print(keys)
[([3, 3],) ([2, 2],) ([0, 0],) ([4, 4],) ([1, 1],)]
(gdb) py print(vals)
[([6, 7],) ([4, 5],) ([0, 1],) ([8, 9],) ([2, 3],)]
(gdb) py print(capacity)
128
```

## Support for KokkosSparse::CrsMatrix

The `printCrsMatrix` command can print
[KokkosSparse::CrsMatrix](https://github.com/kokkos/kokkos-kernels/wiki/CrsMatrix)
in either dense matrix format or in COO sparse format. Consider this code:

```c++
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

void breakpoint() { return; }

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc,argv); {

  const int nCols = 6;
  const int nRows = 4;
  const O nnz = 8;
  Kokkos::View<V*> vs("vs", nnz), vs1("vs1", nnz);;
  Kokkos::View<O*> rs("rs", nRows + 1);
  Kokkos::View<I*> cs("cs", nnz);

  for(O i = 0; i < nnz; ++i) {
    vs(i) = i + 1;
    vs1(i) = vs(i) + 10;
  }
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

  } Kokkos::finalize(); }
```

To print the content of the sparse matrix `m`, do:

```gdb
(gdb) py import GDBKokkos
(gdb) printCrsMatrix m 
# shape: (4, 6)

[[1. 2. 0. 0. 0. 0.]
 [0. 3. 0. 4. 0. 0.]
 [0. 0. 5. 6. 7. 0.]
 [0. 0. 0. 0. 0. 8.]]

```
To print `m` in COO format:
```gdb
(gdb) py import GDBKokkos
(gdb) printCrsMatrix m --printCoo
# shape: (4, 6)

  (0, 0)	1.0
  (0, 1)	2.0
  (1, 1)	3.0
  (1, 3)	4.0
  (2, 2)	5.0
  (2, 3)	6.0
  (2, 4)	7.0
  (3, 5)	8.0

```
Sometimes it would be helpful to examine the results of a sparse matrix
algorithm where the sparsity pattern of `m` is used to hold another set of
values such as `vs1` in the above example. To print `m`'s sparsity pattern with
`vs1` as content:
```gdb
(gdb) py import GDBKokkos
(gdb) printCrsMatrix m --values vs1
# shape: (4, 6)

[[11 12  0  0  0  0]
 [ 0 13  0 14  0  0]
 [ 0  0 15 16 17  0]
 [ 0  0  0  0  0 18]]

```

The `CrsMatrix` can be converted to a `scipy.sparse.csr_matrix` object for numpy
or scipy interoperation:

```gdb
(gdb) py from GDBKokkos.printCrsMatrix import crs2ScipySparse
(gdb) py from GDBKokkos.printView import view2NumpyArray
(gdb) py m = crs2ScipySparse(gdb.parse_and_eval("m"))
(gdb) py print(m)
  (0, 0)	1.0
  (0, 1)	2.0
  (1, 1)	3.0
  (1, 3)	4.0
  (2, 2)	5.0
  (2, 3)	6.0
  (2, 4)	7.0
  (3, 5)	8.0
(gdb) py print(m.toarray())
[[1. 2. 0. 0. 0. 0.]
 [0. 3. 0. 4. 0. 0.]
 [0. 0. 5. 6. 7. 0.]
 [0. 0. 0. 0. 0. 8.]]
(gdb) py vs1 = view2NumpyArray(gdb.parse_and_eval("vs1"))
(gdb) py m1 = crs2ScipySparse(gdb.parse_and_eval("m"), vs1)
(gdb) py print(m1)
  (0, 0)	11
  (0, 1)	12
  (1, 1)	13
  (1, 3)	14
  (2, 2)	15
  (2, 3)	16
  (2, 4)	17
  (3, 5)	18

```


## Known limitations

- Currently you can't use this in `cuda-gdb` to print out View in `Kokkos::CudaSpace` because `cuda-gdb` doesn't support python3 until very recently. But this feature can be added easily.
- The printView with slicing can't reduce the dimensionality of the view. It would take a different slice indices notations to implement this.
