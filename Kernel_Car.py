#!/usr/bin/env python3

"""
Program written by Mattias Kockum
On 20/12/2020
The aim of this program is to train a car to drive itself on a GPU
"""

from C_to_python import *

kernel_code = (C_to_string("Kernel_Car.c"))
