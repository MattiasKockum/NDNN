#!/usr/bin/env python3

"""
Program written by Mattias Kockum
On 20/12/2020
The aim of this program is to train and test my networks on the well known
mnist
It is meant to turn on GPU
"""

from Kernel_AI import *
from GPU_code_maker import *

def Kernel_Code_MNIST(function_name, nb_add_neurons, period):
    """
    Function, nb_neurons and period are here because
    they are the same for every Network
    Note : Image from the MNIST is squished in 1D
    """
    AI_code = Kernel_Code_Network(28**2, 10, nb_add_neurons, period,
                                  function_name)
    code = (AI_code + C_to_string("Kernel_MNIST.c"))
    return(code)
