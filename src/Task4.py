# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 16:51:09 2019

@author: Florian
"""
import numpy as np
from mayavi import mlab


def get_mt_st_helens(Lx, Ba, B0):
    # https://agilescientific.com/blog/2014/5/6/how-much-rock-was-erupted-from-mt-st-helens.html
    Z = np.loadtxt('../data/st-helens_after.txt')
    Z[Z < 0] = np.nan
    Z = Z-np.nanmin(Z)
    Z = Z/np.nanmax(Z)
    Z[np.isnan(Z)] = 0.0
    Z = np.c_[Z*0, Z, Z*0]
    Z = np.r_[Z*0, Z, Z*0]
    Z = Z[::10, ::10]  # resampling

    Z = Z*Ba
    Z = Z+B0
    Nx, Ny = Z.shape
    Ly = Lx/Nx*Ny
    x = np.linspace(0, Lx, Nx)  # mesh points in x dir
    y = np.linspace(0, Ly, Ny)  # mesh points in y dir
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    X, Y = np.meshgrid(x, y)
    Z = Z.T

    print(X.shape)
    print(Y.shape)
    print(Z.shape)
    return Lx, Ly, dx, dy, X, Y, Z


if __name__ == '__main__':
    Lx = 100
    B0 = -15
    Ba = 15
    Bmx = 50
    Bmy = 50  # symmetrie Linie
    Bs = 10  # ?
    b = 1
    Lx, Ly, dx, dy, X, Y, Z = get_mt_st_helens(Lx, Ba, B0)
    print(dx, dy)
    print(Lx, Ly)
    test_extent = (0, Lx, 0, Ly, B0, Ba)
    bottom = mlab.surf(X.T, Y.T, Z.T, colormap='gist_earth')
    mlab.outline(bottom, color=(0.7, .7, .7), extent=test_extent)
    ax = mlab.axes(bottom, color=(.7, .7, .7), extent=test_extent,
                   ranges=test_extent, xlabel='X', ylabel='Y',
                   zlabel='H')
    
    mlab.view(30, -72, np.sqrt(Lx**2+Ly**2)*2)
    mlab.show()