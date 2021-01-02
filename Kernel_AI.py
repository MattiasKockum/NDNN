#!/usr/bin/env python3

"""
Program written by Mattias Kockum
On the 20/12/2020
The aim of this program is to create an AI that runs on GPU for
parallel training
"""

from GPU_code_maker import *

def Kernel_Code_Network(nb_sensors, nb_actors, nb_add_neurons,
                        period, function_name):
    if function_name == "ramp":
            function_code = C_to_string("ramp.c")
    elif function_name == "sigmoid":
            function_code = C_to_string("sigmoid.c")
    elif function_name == "segments":
            function_code = C_to_string("segments.c")
    code = (defines(nb_sensors, nb_actors, nb_add_neurons, period,
                    function_name)
            + function_code
            + C_to_string("Kernel_AI.c")
            )
    return(code)
