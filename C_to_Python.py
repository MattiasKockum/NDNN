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
        string += "+ \"\"\"" + line[:-1] + "\\n\"\"\"\n"
    return(string)

def main(file_name):
    string = C_to_string(file_name)
    print(string)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name', metavar='file_name', type=str)
    args = parser.parse_args()
    main(args.file_name)
