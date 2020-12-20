#!/usr/bin/env python3

"""
Program written by Mattias Kockum
On 20/12/2020
The aim of this program is to train and test my networks on the well known
mnist
It is meant to turn on GPU
"""

from Kernel_AI import *

def Kernel_Code_MNIST(function_name, nb_add_neurons, period, number_of_numbers):
    """
    Function, number_of_numbers, nb_neurons and period are here because
    they are the same for every Network
    Note : Images from the MNIST are squished in 1D
    """
    AI_code = Kernel_Code_Network(28**2, 10, nb_add_neurons, period,
                                  function_name)
    code = (AI_code
    + """
    __kernel void experience(int numberOfTests, int *numbers,
            double **images, double *biasAndWeights, float *score)
    {
        Network N;
        for (int i = 0; i < NB_TOTAL_NEURONS; i++)
        {
            N.values[i] = 0;
            N.bias[i] = biasAndWeights[i];
            for (int j = 0; j < NB_TOTAL_NEURONS; i++)
            {
                N.weights[j][i] = biasAndWeights[NB_TOTAL_NEURONS +
                    i*NB_TOTAL_Neurons + j]
            }
        }
        float score = 0;
        for (int i = 0; i < nbOfTests; i++)
        {
            for (int j = 0; j < NB_SENSORS; j++)
            {
                N.values[j] = images[i][j];
            }
            for (int j = 0; j < PERIOD; j++)
            {
                iteration(&N);
            }
            double max = -2;
            int indice = 0;
            for (int j = NB_TOTAL_NEURONS - NB_ACTORS; j < NB_TOTAL_NEURONS;
                    j++)
            {
                if (N.values[j] > max)
                {
                    max = N.values[j];
                    indice = j;
                }
            }
            if (numbers[i] == indice - NB_TOTAL_NEURONS + NB_ACTORS)
            {
                score += 1;
            }
        }
        score = score/nbOfTests;
    }
    """)
