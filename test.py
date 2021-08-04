import sys

from AI import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

def gradient(X, Y):
    return(
        + 1*np.exp(-(((X-0)/0.2)**2+((Y+0)/0.2)**2))
        + 1.2*np.exp(-(((X-0.2)/0.1)**2+((Y+0.2)/0.1)**2))
        + 1.4*np.exp(-(((X+0.2)/0.05)**2+((Y+0.2)/0.05)**2))
    )


def displayNetwork(net):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Make data.
        X = np.arange(-0.5, 0.5, 0.01)
        Y = np.arange(-0.5, 0.5, 0.01)
        X, Y = np.meshgrid(X, Y)
        Z = np.zeros(X.shape)
        for x in range(X.shape[0]):
            for y in range(X.shape[1]):
                Z[x][y] = net.process([x, y])

        # Plot the surface.
        surf = ax.plot_wireframe(X, Y, Z, color="#0F0F0F0F")

        """
        zs = [net.process([x, y][0]) for x, y in zip(xs, ys)]
        for gen in range(0, len(zs), step):
            ax.scatter(
                xs[gen:gen+step],
                ys[gen:gen+step],
                zs[gen:gen+step],
                color=(
                    "#"
                    + hex(0)[2:]
                    + hex(0)[2:]
                    + hex(int(256*gen/len(zs)))[2:]
                )
            )
        """

        plt.show()

def main(parameters):
    layers = [int(i) for i in parameters[1:]]
    N = ClassicNetwork(layers)
    displayNetwork(N)


if __name__ == "__main__":
    main(sys.argv)
