# GDBKokkos Runtime Tool

This adds the ability to track a View's history (deep copies, where it was allocated) to the data GDBKokkos provides.

## Usage

First, do everything mentioned in the root README, this relies on the GDB python support above.

Then, build the Kokkos tool library in this directory (in this directory, type "make")

Finally, when you run your app in GDB, just load the Kokkos Tool:

```bash
KOKKOS_PROFILE_LIBRARY=/path/to/kp_gdb_extension.so gdb --args ./test
```

### Use the metadata printer

As in the root README, remember to 

```gdb
py import GDBKokkos
```

Then, if you want to print out the metadata of a given View, just

```gdb
printViewMetadata myView
```
