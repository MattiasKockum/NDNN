import sys
sys.path.append("./ModelAI")

from Classic import *

from AI import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


def main(parameters):
    layers = [int(i) for i in parameters[1:]]
    N = ClassicalNetwork(layers)
    displayNetworkGrid(N)


if __name__ == "__main__":
    main(sys.argv)
