#!/bin/bash

CTF2PRV=`which ctf2prv`

module purge
module load gcc/7.2.0 mkl python/3.7.4 swig/3.0.12 babeltrace2/2.0.3

$CTF2PRV $@
