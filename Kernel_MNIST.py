#!/usr/bin/env python3

"""
Program written by Mattias Kockum
On 20/12/2020
The aim of this program is to train and test my networks on the well known
mnist
It is meant to turn on GPU
"""

from Kernel_AI import *

def Kernel_Code_MNIST(function_name, nb_add_neurons, period):
    """
    Function, nb_neurons and period are here because
    they are the same for every Network
    Note : Image from the MNIST is squished in 1D
    """
    AI_code = Kernel_Code_Network(28**2, 10, nb_add_neurons, period,
                                  function_name)
    code = (AI_code
    + """__kernel void experience(int numbers,\n"""
    + """       double *image, double *biasAndWeights, float *score)\n"""
    + """{\n"""
    + """   Network N;\n"""
    + """   for (int i = 0; i < NB_TOTAL_NEURONS; i++)\n"""
    + """   {\n"""
    + """       N.values[i] = 0;\n"""
    + """       N.bias[i] = biasAndWeights[i];\n"""
    + """       for (int j = 0; j < NB_TOTAL_NEURONS; i++)\n"""
    + """       {\n"""
    + """           N.weights[j][i] = biasAndWeights[NB_TOTAL_NEURONS +\n"""
    + """               i*NB_TOTAL_Neurons + j]\n"""
    + """       }\n"""
    + """   }\n"""
    + """   float score = 0;\n"""
    + """   for (int j = 0; j < NB_SENSORS; j++)\n"""
    + """   {\n"""
    + """       N.values[j] = images[i][j];\n"""
    + """   }\n"""
    + """   for (int j = 0; j < PERIOD; j++)\n"""
    + """   {\n"""
    + """       iteration(&N);\n"""
    + """   }\n"""
    + """   double max = -2;\n"""
    + """   int indice = 0;\n"""
    + """   for (int j = NB_TOTAL_NEURONS - NB_ACTORS; j<NB_TOTAL_NEURONS;\n"""
    + """          j++)\n"""
    + """   {\n"""
    + """       if (N.values[j] > max)\n"""
    + """       {\n"""
    + """         max = N.values[j];\n"""
    + """         indice = j;\n"""
    + """       }\n"""
    + """   }\n"""
    + """   if (numbers[i] == indice - NB_TOTAL_NEURONS + NB_ACTORS)\n"""
    + """   {\n"""
    + """       score += 1;\n"""
    + """   }\n"""
    + """}\n"""
    )
    return(code)
