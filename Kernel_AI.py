#!/usr/bin/env python3

"""
Program written by Mattias Kockum
On the 20/12/2020
The aim of this program is to create an AI that runs on GPU for
parallel training
"""

def Kernel_Code_Network(nb_sensors, nb_actors, nb_add_neurons,
                        period, function_name):
    if function_name == "ramp":
            function_code = (
            """double ramp(double x)\n"""
            + """{\n"""
            + """    if (x>0)\n"""
            + """    {\n"""
            + """        return x;\n"""
            + """    }\n"""
            + """    else\n"""
            + """    {\n"""
            + """        return 0;\n"""
            + """    }\n"""
            + """}\n"""
            + """\n"""
            )
    elif function_name == "sigmoid":
            function_code = (
            """double sigmoid(double x)\n"""
            + """{\n"""
            + """    double r;\n"""
            + """    r = -1 + (2/(1+exp(-x)));\n"""
            + """    return r;\n"""
            + """}\n"""
            + """\n"""
            )
    elif function_name == "segments":
            function_code = (
            """double segments(double x)\n"""
            + """{\n"""
            + """   if (x>1)\n"""
            + """    {\n"""
            + """        return 1;\n"""
            + """    }\n"""
            + """   if (x<-1)\n"""
            + """    {\n"""
            + """        return -1;\n"""
            + """    }\n"""
            + """ return x;\n"""
            + """}\n"""
            + """\n"""
            )
    code = (
             """#define NB_SENSORS {}\n""".format(nb_sensors)
            + """#define NB_ACTORS {}\n""".format(nb_actors)
            + """#define NB_ADD_NEURONS {}\n""".format(nb_add_neurons)
            + """#define PERIOD {}\n""".format(period)
            + """#define NB_TOTAL_NEURONS NB_SENSORS + NB_ADD_NEURONS"""
                + """ + NB_ACTORS\n"""
            + """\n"""
            + """typedef struct Network\n"""
            + """{\n"""
            + """    double values[NB_TOTAL_NEURONS];\n"""
            + """    double bias[NB_TOTAL_NEURONS];\n"""
            + """    double weights[NB_TOTAL_NEURONS][NB_TOTAL_NEURONS];\n"""
            + """}\n"""
            + """Network;\n"""
            + """\n"""
            + function_code
            + """void iteration(Network *N)\n"""
            + """{\n"""
            + """    double values2[NB_TOTAL_NEURONS];\n"""
            + """    int i;\n"""
            + """    int j;\n"""
            + """    for (i=0; i<NB_TOTAL_NEURONS; i++)\n"""
            + """    {\n"""
            + """        values2[i] = 0;\n"""
            + """    }\n"""
            + """    for (i=0; i<NB_TOTAL_NEURONS; i++)\n"""
            + """    {\n"""
            + """        for (j=0; j<NB_TOTAL_NEURONS; j++)\n"""
            + """        {\n"""
            + """            values2[j]+=N->weights[j][i]*(N->bias[i]+"""
                + """N->values[i]);\n"""
            + """        }\n"""
            + """    }\n"""
            + """    for (i=0; i<NB_TOTAL_NEURONS; i++)\n"""
            + """    {\n"""
            + """        N->values[i] = {}(values2[i]);\n""".format(
                function_name)
            + """    }\n"""
            + """}\n"""
            + """\n"""
            )
    return(code)
