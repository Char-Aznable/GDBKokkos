default: all

all: kp_gdb_extension.so

kp_gdb_extension.so: kp_gdb_extension.cpp
	g++ -Wno-pointer-arith -O3 -fPIC -std=c++14 -shared -o kp_gdb_extension.so kp_gdb_extension.cpp
