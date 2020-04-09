# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 12:15:34 2019

@author: Florian
"""
import numpy as np
from numpy import linspace, sqrt, zeros, exp
import os
import time
from glob import glob
from mayavi import mlab
delay = 10  # in ms
from Task4 import get_mt_st_helens

# Output path for you animation images
out_path = './'
out_path = os.path.abspath(out_path)+"\\..\\ani.tmp\\"
prefix = 'ani'
ext = '.png'


def scheme_ijn(u_n, u_nm1, q, b, dx, dy, dt,
               f, t_1, dtdx2, dtdy2, dt2,
               i, j, im1, ip1, jm1, jp1, V, ue_const=False):
    """
    Right-hand side of finite difference at point [i,j].
    im1, ip1 denote i-1, i+1, resp. Similar for jm1, jp1.
    t_1 corresponds to u_n (previous time level relative to u).

    """
#    u_n = u[i, j, n]
#    u_nm1 = u[i, j, n-1]
    t1 = ((b*dt-2)*u_nm1[i, j] +
          2*dt2*f(dx*i, dy*j, t_1) +
          4*u_n[i, j])
    t2 = dtdx2*(- q(dx*(i-.5), dy*j)*u_n[i, j] +
                q(dx*(i-.5), dy*j)*u_n[im1, j] -
                q(dx*(i+.5), dy*j)*u_n[i, j] +
                q(dx*(i+.5), dy*j)*u_n[ip1, j]) if u_n.shape[0] > 1 else 0
    t3 = dtdy2*(- q(dx*i, dy*(j-.5))*u_n[i, j] +
                q(dx*i, dy*(j-.5))*u_n[i, jm1] -
                q(dx*i, dy*(j+.5))*u_n[i, j] +
                q(dx*i, dy*(j+.5))*u_n[i, jp1]) if u_n.shape[1] > 1 else 0
    res = 1/(b*dt + 2)*(t1 + 2*t2 + 2*t3)
    if ue_const:
        if isinstance(res, np.ndarray):
            assert (t2 == 0).all(), ""
            assert (t3 == 0).all(), ""
            assert (res == ue_const).all(), ""
        else:
            assert t2 == 0, ""
            assert t3 == 0, ""
            assert res == ue_const, ""
    return res


def scheme_ij1(u_n, u_nm1, q, b, dx, dy, dt,
               f, t_1, dtdx2, dtdy2, dt2,
               i, j, im1, ip1, jm1, jp1, V, ue_const=False):
    """
    Right-hand side of finite difference at point [i,j] for n=1.
    im1, ip1 denote i-1, i+1, resp. Similar for jm1, jp1.
    t_1 corresponds to u_n (previous time level relative to u).

    """
#    u_n = u[i, j, n]
    t1 = (2*dt - b*dt2)*V(i, j) + \
        dt2*f(dx*i, dy*j, 0) + \
        2*u_n[i, j]
    t2 = dtdx2*(- q(dx*(i-.5), dy*j)*u_n[i, j] +
                q(dx*(i-.5), dy*j)*u_n[im1, j] -
                q(dx*(i+.5), dy*j)*u_n[i, j] +
                q(dx*(i+.5), dy*j)*u_n[ip1, j]) if u_n.shape[0] > 1 else 0
    t3 = dtdy2*(- q(dx*i, dy*(j-.5))*u_n[i, j] +
                q(dx*i, dy*(j-.5))*u_n[i, jm1] -
                q(dx*i, dy*(j+.5))*u_n[i, j] +
                q(dx*i, dy*(j+.5))*u_n[i, jp1]) if u_n.shape[1] > 1 else 0
    res = 0.5 * (t1 + t2 + t3)
    if ue_const:
        if isinstance(res, np.ndarray):
            assert (t2 == 0).all(), ""
            assert (t3 == 0).all(), ""
            assert (res == ue_const).all(), ""
        else:
            assert t2 == 0, ""
            assert t3 == 0, ""
            assert res == ue_const, ""
    return res


def scheme_scalar_mesh(u, u_n, u_nm1, q, b, dx, dy, dt,
                       f, t_1, Nx, Ny, n, V,
                       dtdx2, dtdy2, dt2, bc, ue_const=False):
    if n+1 == 1:
        print("first point")
        scheme_ij = scheme_ij1
    elif n+1 > 1:
        scheme_ij = scheme_ijn
    else:
        raise ValueError("n must be larger than 0!")

    Ix = range(0, u.shape[0])
    Iy = range(0, u.shape[1])
    # Interior points
    for i in Ix[1:-1]:
        for j in Iy[1:-1]:
            im1 = i-1
            ip1 = i+1
            jm1 = j-1
            jp1 = j+1
            u[i, j] = scheme_ij(u_n, u_nm1, q, b, dx, dy, dt,
                                f, t_1, dtdx2, dtdy2, dt2,
                                i, j, im1, ip1, jm1, jp1, V, ue_const)
    # Boundary points
    i = Ix[0]
    ip1 = i+1
    im1 = ip1
    if bc['W'] is None:
        for j in Iy[1:-1]:
            jm1 = j-1
            jp1 = j+1
            u[i, j] = scheme_ij(u_n, u_nm1, q, b, dx, dy, dt,
                                f, t_1, dtdx2, dtdy2, dt2,
                                i, j, im1, ip1, jm1, jp1, V, ue_const)
    else:
        for j in Iy[1:-1]:
            u[i, j] = bc['W'](dx*i, dy*j)
    i = Ix[-1]
    im1 = i-1
    ip1 = im1
    if bc['E'] is None:
        for j in Iy[1:-1]:
            jm1 = j-1
            jp1 = j+1
            u[i, j] = scheme_ij(u_n, u_nm1, q, b, dx, dy, dt,
                                f, t_1, dtdx2, dtdy2, dt2,
                                i, j, im1, ip1, jm1, jp1, V, ue_const)
    else:
        for j in Iy[1:-1]:
            u[i, j] = bc['E'](dx*i, dy*j)
    j = Iy[0]
    jp1 = j+1
    jm1 = jp1
    if bc['S'] is None:
        for i in Ix[1:-1]:
            im1 = i-1
            ip1 = i+1
            u[i, j] = scheme_ij(u_n, u_nm1, q, b, dx, dy, dt,
                                f, t_1, dtdx2, dtdy2, dt2,
                                i, j, im1, ip1, jm1, jp1, V, ue_const)
    else:
        for i in Ix[1:-1]:
            u[i, j] = bc['S'](dx*i, dy*j)
    j = Iy[-1]
    jm1 = j-1
    jp1 = jm1
    if bc['N'] is None:
        for i in Ix[1:-1]:
            im1 = i-1
            ip1 = i+1
            u[i, j] = scheme_ij(u_n, u_nm1, q, b, dx, dy, dt,
                                f, t_1, dtdx2, dtdy2, dt2,
                                i, j, im1, ip1, jm1, jp1, V, ue_const)
    else:
        for i in Ix[1:-1]:
            u[i, j] = bc['N'](dx*i, dy*j)
    # lower left.
    i, j = Ix[0], Iy[0]
    im1 = i+1
    ip1 = i+1
    jm1 = j+1
    jp1 = j+1
    if bc['S'] is None:
        u[i, j] = scheme_ij(u_n, u_nm1, q, b, dx, dy, dt,
                            f, t_1, dtdx2, dtdy2, dt2,
                            i, j, im1, ip1, jm1, jp1, V, ue_const)
    else:
        u[i, j] = bc['S'](dx*i, dy*j)
    # lower right.
    i, j = Ix[-1], Iy[0]
    im1 = i-1
    ip1 = i-1
    jm1 = j+1
    jp1 = j+1
    if bc['S'] is None:
        u[i, j] = scheme_ij(u_n, u_nm1, q, b, dx, dy, dt,
                            f, t_1, dtdx2, dtdy2, dt2,
                            i, j, im1, ip1, jm1, jp1, V, ue_const)
    else:
        u[i, j] = bc['S'](dx*i, dy*j)
    # upper left
    i, j = Ix[-1], Iy[-1]
    im1 = i-1
    ip1 = i-1
    jm1 = j-1
    jp1 = j-1
    if bc['N'] is None:
        u[i, j] = scheme_ij(u_n, u_nm1, q, b, dx, dy, dt,
                            f, t_1, dtdx2, dtdy2, dt2,
                            i, j, im1, ip1, jm1, jp1, V, ue_const)
    else:
        u[i, j] = bc['N'](dx*i, dy*j)
    # upper right.
    i, j = Ix[0], Iy[-1]
    im1 = i+1
    ip1 = i+1
    jm1 = j-1
    jp1 = j-1
    if bc['N'] is None:
        u[i, j] = scheme_ij(u_n, u_nm1, q, b, dx, dy, dt,
                            f, t_1, dtdx2, dtdy2, dt2,
                            i, j, im1, ip1, jm1, jp1, V, ue_const)
    else:
        u[i, j] = bc['N'](dx*i, dy*j)
    return u


def scheme_vectorized_mesh(u, u_n, u_nm1, q, b, dx, dy, dt,
                           f, t_1, Nx, Ny, n, V,
                           dtdx2, dtdy2, dt2, bc, ue_const):
    if n+1 == 1:
        scheme_ij = scheme_ij1
    elif n+1 > 1:
        scheme_ij = scheme_ijn
    else:
        raise ValueError("n must be larger than 0!")
    # # # # # # # # # # # INTERIOR POINTS # # # # # # # # # # # # # # # # # # #
    i = np.arange(0, Nx+1)  # the whole row
    j = np.arange(0, Ny+1)  # the whole column
    ii, jj = np.meshgrid(i, j)  # all rows and all columns
    im1 = ii-1
    ip1 = ii+1
    jm1 = jj-1
    jp1 = jj+1
    # Neumann conditions
    im1[:, 0] = 1
    ip1[:, Nx] = Nx - 1
    jm1[0, :] = 1
    jp1[Ny, :] = Ny - 1
    # flatten for "advanced indexing", see:
    # https://docs.scipy.org/doc/numpy-1.17.0/reference/arrays.indexing.html#advanced-indexing
    ii, jj = ii.ravel(), jj.ravel()
    im1, jm1 = im1.ravel(), jm1.ravel()
    ip1, jp1 = ip1.ravel(), jp1.ravel()
    u[ii, jj] = scheme_ij(u_n, u_nm1, q, b, dx, dy, dt,
                          f, t_1, dtdx2, dtdy2, dt2,
                          ii, jj, im1, ip1, jm1, jp1, V, ue_const)

    # in case the boundary conditions bc are given,
    #  we have to deal with them seperately:
    # # # # # # # # # # # BOUNDARY POINTS # # # # # # # # # # # # # # # # # # #
    Ix = range(0, u.shape[0])
    Iy = range(0, u.shape[1])
    if bc['W'] is not None:
        i_ = Ix[0]
        j_ = j
        u[i_, j_] = bc['W'](dx*i_, dy*j_)
    if bc['E'] is not None:
        i_ = Ix[-1]
        j_ = j
        u[i_, j_] = bc['E'](dx*i_, dy*j_)
    if bc['S'] is not None:
        i_ = i
        j_ = Iy[0]
        u[i_, j_] = bc['S'](dx*i_, dy*j_)
    if bc['N'] is not None:
        i_ = i
        j_ = Iy[-1]
        u[i_, j_] = bc['N'](dx*i_, dy*j_)
    # # # # # # # # # # # # CORNER POINTS # # # # # # # # # # # # # # # # # # #
    if bc['S'] is not None:
        # lower left.
        i, j = Ix[0], Iy[0]
        u[i, j] = bc['S'](dx*i, dy*j)
    if bc['S'] is not None:
        # lower right.
        i, j = Ix[-1], Iy[0]
        u[i, j] = bc['S'](dx*i, dy*j)
    if bc['N'] is not None:
        # upper right.
        i, j = Ix[-1], Iy[-1]
        u[i, j] = bc['N'](dx*i, dy*j)
    if bc['N'] is not None:
        # upper left.
        i, j = Ix[0], Iy[-1]
        i, j = Ix[0], Iy[-1]
        u[i, j] = bc['N'](dx*i, dy*j)
    return u


def get_dt(Lx, Ly, Nx, Ny, dt, q):
    """computes the grid"""
    # Find max(c) using a fake mesh and adapt dx to C and dt
    if isinstance(q, (float, int)):
        c_max = q**.5

        def q(x, y):
            return c_max**2

    elif callable(q):
        x_fake = np.linspace(0, Lx, 101)
        y_fake = np.linspace(0, Ly, 101)
        q_fake = q(*np.meshgrid(x_fake, y_fake))
        c_max = np.max(q_fake)**.5

    x = np.linspace(0, Lx, Nx+1)  # mesh points in x dir
    y = np.linspace(0, Ly, Ny+1)  # mesh points in y dir
    dx = 0 if len(x) == 1 else x[1] - x[0]
    dy = 0 if len(y) == 1 else y[1] - y[0]
    if dt <= 0:                # max time step?
        if dx == 0:
            dt = dy/c_max
        elif dy == 0:
            dt = dx/c_max
        else:
            dt = (1/float(c_max))*(1/sqrt(1/dx**2 + 1/dy**2))
        print("dt = ", dt)
    return x, y, dx, dy, dt


@mlab.animate(delay=delay)
def solver(I, f, q, bc, Lx, Ly, Nx, Ny, dt, T, b, V,
           user_action=None, version='scalar',
           verbose=True, ue_const=False):
    """
    Solve the 2D wave equation u_tt = u_xx + u_yy + f(x,t) on (0,L) with
    u = bc(x, y, t) on the boundary and initial condition du/dt = 0.

    Nx and Ny are the total number of grid cells in the x and y
    directions. The grid points are numbered as (0,0), (1,0), (2,0),
    ..., (Nx,0), (0,1), (1,1), ..., (Nx, Ny).

    dt is the time step. If dt<=0, an optimal time step is used.
    T is the stop time for the simulation.

    I, f, bc are functions: I(x,y), f(x,y,t), bc(x,y,t)

    user_action: function of (u, x, y, t, n) called at each time
    level (x and y are one-dimensional coordinate vectors).
    This function allows the calling code to plot the solution,
    compute errors, etc.

    verbose: true if a message at each time step is written,
    false implies no output during the simulation.
    """
    if version == 'scalar':
        scheme = scheme_scalar_mesh
    elif version == "vectorized":
        scheme = scheme_vectorized_mesh
    else:
        raise ValueError("unknown scheme!")

    x, y, dx, dy, dt = get_dt(Lx, Ly, Nx, Ny, dt, q)
    Nt = int(round(T/float(dt)))
    t = linspace(0, T, Nt+1)   # mesh points in time

    dtdx2 = (dt/dx)**2 if dx > 0 else 0
    dtdy2 = (dt/dy)**2 if dy > 0 else 0
    dt2 = dt**2

    u_np1 = np.zeros((Nx+1, Ny+1), dtype=np.float)   # solution array
    u_n = zeros((Nx+1, Ny+1), dtype=np.float)   # solution at t-dt
    u_nm1 = zeros((Nx+1, Ny+1), dtype=np.float)   # solution at t-2*dt
    u_np1.fill(np.nan)
    u_n.fill(np.nan)
    u_nm1.fill(np.nan)

    Ix = range(0, Nx+1)
    Iy = range(0, Ny+1)

    # Load initial condition into u_n
    for i in Ix:
        for j in Iy:
            u_np1[i, j] = I(dx*i, dy*j)
    if user_action is not None:
        user_action(u_np1, x, y, t, 0)
    u_n = u_np1.copy()

    # Special formula for first time step
    u_np1 = scheme(u_np1, u_n, u_nm1, q, b, dx, dy, dt,
                   f, t[0], Nx, Ny, 0, V,
                   dtdx2, dtdy2, dt2, bc, ue_const)
    if user_action is not None:
        user_action(u_np1, x, y, t, 1)

    for n in range(1, Nt):
        u_nm1 = u_n.copy()
        u_n = u_np1.copy()
        u_np1 = scheme(u_np1, u_n, u_nm1, q, b, dx, dy, dt,
                       f, t[n], Nx, Ny, n, V,
                       dtdx2, dtdy2, dt2, bc, ue_const)
        if user_action is not None:
            if user_action(u_np1, x, y, t, n+1):
                break
            yield
    return dt  # dt might be computed in this function


def test_Gaussian(plot_u=1, version='scalar'):
    """
    Initial Gaussian bell in the middle of the domain.
    plot: not plot: 0; mesh: 1, surf: 2.
    """
    # Clean up plot files
    for name in glob('tmp_*.png'):
        os.remove(name)

    def V(x, y):
        return 0.0

    def I(x, y):
        return exp(-(x-Lx/2.0)**2/2.0 - (y-Ly/2.0)**2/2.0)

    def f(x, y, t):
        return 0.0

    Nx = 40
    Ny = 40
    T = 15
    Lx = 10
    Ly = 10
    q = 1.0
    b = 1.0

    # initialize plot
    # https://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html#surf
    gauss_extent = (0, Lx, 0, Ly, 0, 1)
    x = np.linspace(0, Lx, Nx+1)  # mesh points in x dir
    y = np.linspace(0, Ly, Ny+1)  # mesh points in y dir
    X, Y = np.meshgrid(x, y)
    Z = I(X, Y)
    mlab.figure(1, size=(1920, 1080), fgcolor=(1, 1, 1),
                bgcolor=(0.5, 0.5, 0.5))
    ms1 = mlab.surf(X.T, Y.T, Z.T, colormap='Spectral')
    ms2 = mlab.surf(X.T, Y.T, Z.T, color=(0.0, .0, .0),
                    representation="wireframe")
    mlab.outline(ms1, color=(0.7, .7, .7), extent=gauss_extent)
    ax = mlab.axes(ms1, color=(.7, .7, .7), extent=gauss_extent,
                   ranges=gauss_extent, xlabel='X', ylabel='Y',
                   zlabel='u(t)')
    ax.axes.label_format = '%.0f'
    mlab.view(142, -72, 32)

    bc = {'N': None, 'W': None, 'E': None, 'S': None}

    def action(u, x, y, t, n):
        ms1.mlab_source.set(z=u, scalars=u)
        ms2.mlab_source.set(z=u, scalars=u)
        mlab.title('Gaussian bell, t = {:.2f}'.format(t[n]))
    solver(I, f, q, bc, Lx, Ly, Nx, Ny, 0, T, b, V,
           user_action=action, version='vectorized')  # vectorized or scalar
    mlab.show()


def test_3_1(plot_u=1, version='scalar'):
    """
    """
    _c_ = 0.25

    def u_e(x, y, t):
        return x*0 + _c_

    def f(x, y, t):
        return 0.0

    def I(x, y):
        return x*0 + _c_

    def V(x, y):
        return 0.0

    def q(x, y):
        return x*0 + 1  # np.sin(x)*np.cos(y)

    bc = {'N': None, 'W': None, 'E': None, 'S': None}

    b = 1.0
    Nx = 8
    Ny = 15
    T = 15
    Lx = 50
    Ly = 50

    # initialize plot
    # https://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html#surf
    test_extent = (0, Lx, 0, Ly, 0, 1)
    x = np.linspace(0, Lx, Nx+1)  # mesh points in x dir
    y = np.linspace(0, Ly, Ny+1)  # mesh points in y dir
    X, Y = np.meshgrid(x, y)
    Z = I(X, Y)
    mlab.figure(1, size=(1920, 1080), fgcolor=(1, 1, 1),
                bgcolor=(0.5, 0.5, 0.5))
    ms1 = mlab.surf(X.T, Y.T, Z.T, colormap='Spectral')
    ms2 = mlab.surf(X.T, Y.T, Z.T, color=(0.0, .0, .0),
                    representation="wireframe")
    mlab.outline(ms1, color=(0.7, .7, .7), extent=test_extent)
    ax = mlab.axes(ms1, color=(.7, .7, .7), extent=test_extent,
                   ranges=test_extent, xlabel='X', ylabel='Y',
                   zlabel='u(t)')
    ax.axes.label_format = '%.0f'
    mlab.view(142, -72, 32)

    def action(u, x, y, t, n):
        print(n)
        print(u)
#        print('action, t=', t, 'u=', u, 'x=', x, 'y=', y)
        u_exakt = u_e(X.T, Y.T, t[n])
        ms1.mlab_source.set(z=u_exakt, scalars=u_exakt)
        ms2.mlab_source.set(z=u, scalars=u)
        mlab.title('Test, t = {:.2f}'.format(t[n]))
        return
    solver(I, f, q, bc, Lx, Ly, Nx, Ny, 0, T, b, V,
           user_action=action, version='vectorized', ue_const=_c_)
    mlab.show()


def test_3_4(dx, dy, dt):
    """
    """
    bc = {'N': None, 'W': None, 'E': None, 'S': None}
    A = 2.3
    mx = 3
    my = 4
    c = 1  #
    b = 1.0
    Lx = 10
    Ly = 10
    T = 2*10/2.**.5
    w = 2*np.pi/T

    # # # #
    sin = np.sin
    cos = np.cos
    pi = np.pi

    def u_e(x, y, t):
        kx = mx*np.pi/Lx
        ky = my*np.pi/Ly
        return A*np.cos(kx*x)*np.cos(ky*y)*np.cos(w*t)

    def f(x, y, t):
        res = A*(-Lx**2*Ly**2*w*(b*sin(t*w) + w*cos(t*w)) +
                 pi**2*Lx**2*c**2*my**2*cos(t*w) +
                 pi**2*Ly**2*c**2*mx**2*cos(t*w)) * \
            cos(pi*mx*x/Lx)*cos(pi*my*y/Ly)/(Lx**2*Ly**2)
        return res

    def I(x, y):
        return A*cos(pi*mx*x/Lx)*cos(pi*my*y/Ly)

    def V(x, y):
        return 0.0

    def q(x, y):
        return c**2

    Nt = int(round(T/float(dt)))
    Nx = int(round(Lx/float(dx)))
    Ny = int(round(Ly/float(dy)))
    x, y, dx, dy, dt = get_dt(Lx, Ly, Nx, Ny, dt, q)

    e = np.zeros((Nx+1, Ny+1, Nt+1))

    # https://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html#surf
    test_extent = (0, Lx, 0, Ly, 0, 1)
    X, Y = np.meshgrid(x, y)
    Z = I(X, Y)
    mlab.figure(1, size=(1920, 1080), fgcolor=(1, 1, 1),
                bgcolor=(0.5, 0.5, 0.5))
    ms1 = mlab.surf(X.T, Y.T, Z.T, colormap='Spectral')
    ms2 = mlab.surf(X.T, Y.T, Z.T, color=(0.0, .0, .0),
                    representation="wireframe")
    mlab.outline(ms1, color=(0.7, .7, .7), extent=test_extent)
    ax = mlab.axes(ms1, color=(.7, .7, .7), extent=test_extent,
                   ranges=test_extent, xlabel='X', ylabel='Y',
                   zlabel='u(t)')
    ax.axes.label_format = '%.0f'
    mlab.view(142, -72, 32)

    def action(u, x, y, t, n):
        u_exakt = u_e(X.T, Y.T, t[n])
        e[:, :, n] = u_exakt-u

        ms1.mlab_source.set(z=u_exakt, scalars=u_exakt)
        ms2.mlab_source.set(z=u, scalars=u)
        mlab.title('standing undamped wave, t = {:.2f}'.format(t[n]))
        # concate filename with zero padded index number as suffix
        filename = os.path.join(out_path, '{}_{:06d}{}'.format(prefix, n, ext))
        print(filename)
        mlab.savefig(filename=filename)
        return

    solver(I, f, q, bc, Lx, Ly, Nx, Ny, dt, T, b, V,
           user_action=action, version='vectorized')
    mlab.show()
#    print(e.min(), e.max(), e.mean())
    return e, dx, dy, dt


def calc_err_34():
    #    C = c = 1
    h1 = 2
    dx1 = dy1 = h1
    dt1 = dx1/2.**.5
    print(dt1)
    e_1, dx1, dy1, dt1 = test_3_4(dx1, dy1, dt1)
    E1 = np.sqrt(dx1*dy1*dt1*np.sum(e_1**2))  # eq. 26
    for i in range(7):
        h2 = h1/2
        dx2 = dy2 = h2
        dt2 = dx2/2.**.5
        e_2, dx2, dy2, dt2 = test_3_4(dx2, dy2, dt2)
        E2 = np.sqrt(dx2*dy2*dt2*np.sum(e_2**2))  # eq. 26
        conv_rate = np.log(E2/E1)/np.log(h2/h1)
        print("E = ", E2)
        print("E1 = {:.4f}, E2 = {:.4f}".format(E1, E2))
        print("h1 = {:.4f}, h2 = {:.4f}".format(h1, h2))
        print("dx1 = {:.4f}, dx2 = {:.4f}".format(dx1, dx2))
        print("dy1 = {:.4f}, dy2 = {:.4f}".format(dy1, dy2))
        print("dt1 = {:.4f}, dt2 = {:.4f}".format(dt1, dt2))
        print("convergence rate: ", conv_rate)
        e_1, dx1, dy1, dt1 = e_2.copy(), dx2, dy2, dt2
        E1 = E2
        h1 = h2
    return


def test_3_6(dx, dy, dt):
    """
    """
    bc = {'N': None, 'W': None, 'E': None, 'S': None}
    mx = 2
    my = 2
    Lx = 5.
    Ly = 5.
    T = 10./2.**.5

    # # # #
    sin = np.sin
    cos = np.cos
    pi = np.pi
    exp = np.exp

    k = 0.9
    B = 1.0

    kx = mx*pi/Lx
    ky = my*pi/Ly
    A = B
    b = np.sqrt(2*k*kx**2 + 2*k*ky**2)
    c = b/2
    w = np.sqrt(kx**2*k + ky**2*k - c**2)

    def u_e(x, y, t):
        return (A*cos(w*t) + B*sin(w*t)) * \
            exp(-c*t) * cos(kx*x) * cos(ky*y)

    def f(x, y, t):
        return 0.0

    def q(x, y):
        return k

    def I(x, y):
        return B*cos(kx*x)*cos(ky*y)

    def V(x, y):
        return 0.0

    Nt = int(round(T/float(dt)))
    Nx = int(round(Lx/float(dx)))
    Ny = int(round(Ly/float(dy)))
    x, y, dx, dy, dt = get_dt(Lx, Ly, Nx, Ny, dt, q)

    e = np.zeros((Nx+1, Ny+1, Nt+1))

    # https://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html#surf
    test_extent = (0, Lx, 0, Ly, -1, 1)
    X, Y = np.meshgrid(x, y)
    Z = I(X, Y)
    mlab.figure(1, size=(1920, 1080), fgcolor=(1, 1, 1),
                bgcolor=(0.5, 0.5, 0.5))
    ms1 = mlab.surf(X.T, Y.T, Z.T, colormap='Spectral')
    ms2 = mlab.surf(X.T, Y.T, Z.T, color=(0.0, .0, .0),
                    representation="wireframe")
    mlab.outline(ms1, color=(0.7, .7, .7), extent=test_extent)
    ax = mlab.axes(ms1, color=(.7, .7, .7), extent=test_extent,
                   ranges=test_extent, xlabel='X', ylabel='Y',
                   zlabel='u(t)')
    ax.axes.label_format = '%.1f'
    mlab.view(142, -72, sqrt(Lx**2+Ly**2)*4)

    def action(u, x, y, t, n):
        u_exakt = u_e(X.T, Y.T, t[n])
        e[:, :, n] = u_exakt-u

        ms1.mlab_source.set(z=u_exakt, scalars=u_exakt)
        ms2.mlab_source.set(z=u, scalars=u)
        mlab.title('standing damped wave, t = {:.2f}'.format(t[n]))
        filename = os.path.join(out_path, '{}_{:06d}{}'.format(prefix, n, ext))
        print(filename)
        mlab.savefig(filename=filename)
        return

    solver(I, f, q, bc, Lx, Ly, Nx, Ny, dt, T, b, V,
           user_action=action, version='vectorized')
    mlab.show()
#    print(e.min(), e.max(), e.mean())
    return e, dx, dy, dt


def calc_err_36():
    #    C = c = 1
    h1 = .5
    dx1 = dy1 = h1
    dt1 = dx1/2.**.5
    print(dt1)
    e_1, dx1, dy1, dt1 = test_3_6(dx1, dy1, dt1)
    E1 = np.sqrt(dx1*dy1*dt1*np.sum(e_1**2))  # eq. 26
    for i in range(7):
        h2 = h1/2
        dx2 = dy2 = h2
        dt2 = dx2/2.**.5
        e_2, dx2, dy2, dt2 = test_3_6(dx2, dy2, dt2)
        E2 = np.sqrt(dx2*dy2*dt2*np.sum(e_2**2))  # eq. 26
        conv_rate = np.log(E2/E1)/np.log(h2/h1)
        print("E = ", E2)
        print("E1 = {:.4f}, E2 = {:.4f}".format(E1, E2))
        print("h1 = {:.4f}, h2 = {:.4f}".format(h1, h2))
        print("dx1 = {:.4f}, dx2 = {:.4f}".format(dx1, dx2))
        print("dy1 = {:.4f}, dy2 = {:.4f}".format(dy1, dy2))
        print("dt1 = {:.4f}, dt2 = {:.4f}".format(dt1, dt2))
        print("convergence rate: ", conv_rate)
        e_1, dx1, dy1, dt1 = e_2.copy(), dx2, dy2, dt2
        E1 = E2
        h1 = h2
        assert conv_rate>1.8, "convergence rate expected to be 2.0 +- 10%"
        assert conv_rate<2.2, "convergence rate expected to be 2.0 +- 10%"
    return


def task_33():
    pulse(Nx=100, Ny=0, pulse_tp='plug', T=5, medium=[-1, -1])
    pulse(Nx=0, Ny=100, pulse_tp='plug', T=5, medium=[-1, -1])
    return


def pulse(C=1, Nx=200, Ny=0, animate=True, version='vectorized', T=2,
          loc='center', pulse_tp='gaussian', slowness_factor=2,
          medium=[0.7, 0.9], every_frame=1, sigma=0.05):
    """
    Various peaked-shaped initial conditions on [0,1].
    Wave velocity is decreased by the slowness_factor inside
    medium. The loc parameter can be 'center' or 'left',
    depending on where the initial pulse is to be located.
    The sigma parameter governs the width of the pulse.
    """
    # Use scaled parameters: L=1 for domain length, c_0=1
    # for wave velocity outside the domain.
    Lx = 1.0
    Ly = 1.0
    c_0 = 1.0
    if loc == 'center':
        xc = Lx/2
    elif loc == 'left':
        xc = 0

    if pulse_tp in ('gaussian', 'Gaussian'):

        def I(x):
            return exp(-0.5*((x-xc)/sigma)**2)

    elif pulse_tp == 'plug':

        def I(x):
            _I_ = np.ones_like(x)
            _I_[abs(x-xc) > sigma] = 0
            return _I_

    elif pulse_tp == 'cosinehat':

        def I(x):
            # One period of a cosine
            w = 2
            a = w*sigma
            return (0.5*(1 + np.cos(np.pi*(x-xc)/a))
                    if xc - a <= x <= xc + a else 0)

    elif pulse_tp == 'half-cosinehat':

        def I(x):
            # Half a period of a cosine
            w = 4
            a = w*sigma
            return (np.cos(np.pi*(x-xc)/a)
                    if xc - 0.5*a <= x <= xc + 0.5*a else 0)
    else:
        raise ValueError('Wrong pulse_tp="%s"' % pulse_tp)

    def c(x):
        return c_0/slowness_factor \
               if medium[0] <= x <= medium[1] else c_0

    ##############
    def q(x, y):
        if Ny == 0:
            l = x
        elif Nx == 0:
            l = y

        if isinstance(l, np.ndarray):
            _q_ = np.zeros_like(l).ravel()
            for i, elem in enumerate(np.nditer(l)):
                _q_[i] = c(elem)**2
            _q_.shape = l.shape
            return _q_
        else:
            return c(x)**2

    def I_2D(x, y):
        if Ny == 0:
            l = x
        elif Nx == 0:
            l = y
        return I(l)

    def f(x, y, t):
        return 0

    def V(x, y):
        return 0

    dt = 0  # is calculated in the solver
    b = 0
    bc = {'N': None, 'W': None, 'E': None, 'S': None}
    ##############

    # initialize plot
    # https://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html#surf
    test_extent = (0, Lx, 0, Ly, 0, 1)
    x = np.linspace(0, Lx, Nx+1)  # mesh points in x dir
    y = np.linspace(0, Ly, Ny+1)  # mesh points in y dir
    X, Y = np.meshgrid(x, y)
    Z = I_2D(X, Y)
    mlab.figure(1, size=(1920, 1080), fgcolor=(1, 1, 1),
                bgcolor=(0.5, 0.5, 0.5))
#    ms1 = mlab.surf(X.T, Y.T, Z.T, colormap='Spectral')
    ms2 = mlab.points3d(X.ravel(), Y.ravel(), Z.ravel(), color=(0.0, .0, .0),
                        scale_factor=.025)
    mlab.outline(ms2, color=(0.7, .7, .7), extent=test_extent)
    ax = mlab.axes(ms2, color=(.7, .7, .7), extent=test_extent,
                   ranges=test_extent, xlabel='X', ylabel='Y',
                   zlabel='u(t)')
    ax.axes.label_format = '%.0f'
    mlab.view(142, -72, 5)
    last_solution = Z.T

    def action(u, x, y, t, n):
        if n > 1:
            # every timestep exactly 2 cells change. If the wave is at the
            # boundary, only two cells change.
            msg = "the plug should travel exactly one cell per time step"
            cells_with_new_value = (u-last_solution) != 0
            print(np.sum(cells_with_new_value))
            assert ((np.sum(cells_with_new_value) == 4) |
                    (np.sum(cells_with_new_value) == 2) |
                    (np.sum(cells_with_new_value) == 0)), msg

        ms2.mlab_source.set(z=u)
        mlab.title('pulse, t = {:.2f}'.format(t[n]))
        last_solution[:, :] = u[:, :]
        # concate filename with zero padded index number as suffix
        filename = os.path.join(out_path, '{}_{:06d}{}'.format(prefix, n, ext))
        print(filename)
        mlab.savefig(filename=filename)
        return
    solver(I_2D, f, q, bc, Lx, Ly, Nx, Ny, dt, T, b, V,
           user_action=action, version='vectorized')
    mlab.show()

def task4():
    """
    """
    h = 0.5
    dx = dy = h
    dt = 0

    I0 = 0
    Ia = 40  # height wave
    Im = 0
    Is = 5

    B0 = -30
    Ba = 28
    Bmx = 50
    Bmy = 50  # symmetrie Linie
    Bs = 100  # ~Breite
    b = 1

    Lx = 150
    Ly = 150
    T = 15

    g = 9.81

    Lx, Ly, dx, dy, X, Y, Z_HELEN = get_mt_st_helens(Lx, Ba, B0)

    def st_helens(x, y):
#        print(x.shape, Z_HELEN.shape)
        if x.ravel().shape != Y.ravel().shape:
            _z_ = np.ones_like(x)*Z_HELEN.min()  # the fake mesh for c_max pukes here
#            print(_z_)
            return _z_
        else:
            return Z_HELEN.reshape(x.shape)

    def H(x, y):
        z = -st_helens(x, y)
#        print("H: ", z.min(), z.max())
        return z

    def q(x, y):
        return g*H(x, y)

    def I(x, y):  # 7
        return I0 + Ia*np.exp(-((x-Im)/Is)**2)

    def B8(x, y):  # 8
        return B0 + Ba*np.exp(-((x-Bmx)/Bs)**2-((y-Bmy)/(b*Bs))**2)

    def B9(x, y):  # 9
        return B0+Ba*np.cos(np.pi*(x-Bmx)/(2*Bs))*np.cos(np.pi*(y-Bmy)/(2*Bs))

    def B10(x, y):  # 10
        res = np.ones_like(x) * B0
        l1 = ((Bmx-Bs) <= x) & (x <= (Bmx+Bs))
        l2 = ((Bmy-b*Bs) <= y) & (y <= (Bmy+b*Bs))
        res[l1 & l2] = B0 + Ba
        return res

    def V(x, y):
        return 0.0

    def f(x, y, t):
        return 0.0

    bc = {'N': None, 'W': None, 'E': None, 'S': None}

    Nx = int(round(Lx/float(dx)))
    Ny = int(round(Ly/float(dy)))

    x, y, dx, dy, dt = get_dt(Lx, Ly, Nx, Ny, dt, q)


    # https://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html#surf
    test_extent = (0, Lx, 0, Ly, B0, Ia)
    X, Y = np.meshgrid(x, y)
    Z = I(X, Y)
    print(Z.shape)
    mlab.figure(1, size=(1920/2, 1080/2), fgcolor=(0, 0, 0),
                bgcolor=(1., 1., 1.))
    bottom = mlab.surf(X.T, Y.T, -H(X, Y).T, colormap='gist_earth')
    ms2 = mlab.surf(X.T, Y.T, Z.T, colormap='jet', opacity=0.5,
                    transparent=True, vmin=0, vmax=Ia/15)
    mlab.outline(ms2, color=(0.7, .7, .7), extent=test_extent)
    ax = mlab.axes(ms2, color=(.7, .7, .7), extent=test_extent,
                   ranges=test_extent, xlabel='X', ylabel='Y',
                   zlabel='H')
    ax.axes.label_format = ''
    mlab.view(140, -72, np.sqrt(Lx**2+Ly**2)*1.5)

    def action(u, x, y, t, n):
#        u_exakt = u_e(X.T, Y.T, t[n])
#        e[:, :, n] = u_exakt-u

#        ms1.mlab_source.set(z=u_exakt, scalars=u_exakt)
        nn = 10
        if n%nn == 0:
            ms2.mlab_source.set(z=u, scalars=u)
            mlab.title('Tsunami, t = {:.2f}'.format(t[n]))
            # concate filename with zero padded index number as suffix
            filename = os.path.join(out_path, '{}_{:06d}{}'.format(prefix, n//nn, ext))
            print(filename)
            mlab.savefig(filename=filename)
        return

    solver(I, f, q, bc, Lx, Ly, Nx, Ny, dt, T, b, V,
           user_action=action, version='vectorized')
    mlab.show()
#    print(e.min(), e.max(), e.mean())
    return


if __name__ == '__main__':
#    e_1, dx1, dy1, dt1 = test_3_6(.5, .5, .5/2.**.5)
#    pulse(Nx=0, Ny=25, pulse_tp='plug', T=1.0, medium=[-1, -1])
#    e_1, dx1, dy1, dt1 = test_3_4(.5, .5, .5/2.**.5)
    task4()
    calc_err_36()
#    calc_err_34()
#    task_33()
#    test_3_1()
#    test_Gaussian()

#    import sys
#
#    if len(sys.argv) < 2:
#        print("""Usage [...] function arg1 arg2 arg3 ...""")
#        print(sys.argv[0])
#        sys.exit(0)
#    print("hi")
#    cmd = '%s(%s)' % (sys.argv[1], ', '.join(sys.argv[2:]))
#    print(cmd)
#    eval(cmd)
