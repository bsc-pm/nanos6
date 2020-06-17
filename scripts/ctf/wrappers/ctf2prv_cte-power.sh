#!/bin/bash

CTF2PRV=`which ctf2prv`

module load swig python/3.6.5 babeltrace2

$CTF2PRV $@
