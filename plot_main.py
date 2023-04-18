import numpy as np
from matplotlib import pyplot as plt


def A(x, y):
    return (y-1) / (x*y-1) * np.log((x*y-1)/(x*(y-1)))

def B(x, y):
    return (x-1) / (x*y-1) * np.log((x*y-1)/(y*(x-1)))

def C(x, y):
    return (x*y-x-y+1) / (x*y-1) * np.log((x*y-1)/(x*y))

def I(x, y):
    return A(x, y) + B(x, y) + C(x, y)

def plot_single_entry_I():
    alpha = np.arange(2,20)
    beta = np.arange(2,20)

    X, Y = np.meshgrid(alpha, beta)
    Z = I(X, Y)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_title('surface')
    ax.set_xlabel(r'$\alpha$', fontsize=15)
    ax.set_ylabel(r'$\beta$', fontsize=15)
    ax.set_zlabel(r'$I(X;Y)$', fontsize=15)
    ax.set_title(r'$I(X;Y)$ vs. $\alpha,\beta$ - Single Entry Removed', fontsize=15)


if __name__ == '__main__':
    plot_single_entry_I()