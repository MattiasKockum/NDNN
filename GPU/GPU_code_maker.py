#!/usr/bin/env python3

"""
Program written by Mattias Kockum
On the 21/12/2020
The aim of this program is to transform C code to python string
"""

def C_to_string(file_name):
    f = open(file_name, "r")
    string = ""
    for line in f.readlines():
        string += line[:-1] + "\n"
    return(string)

def defines(nb_sensors, nb_actors, nb_add_neurons, period, function_name):
    return("""#define NB_SENSORS {}\n""".format(nb_sensors)
            + """#define NB_ACTORS {}\n""".format(nb_actors)
            + """#define NB_ADD_NEURONS {}\n""".format(nb_add_neurons)
            + """#define PERIOD {}\n""".format(period)
            + """#define FUNCTION {}\n""".format(function_name)
            + """#define NB_TOTAL_NEURONS NB_SENSORS + NB_ADD_NEURONS"""
                + """ + NB_ACTORS\n"""
            + """\n""")

def main(file_name):
    string = C_to_string(file_name)
    print(string)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name', metavar='file_name', type=str)
    args = parser.parse_args()
    main(args.file_name)
