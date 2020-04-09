# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 12:24:42 2019

@author: Florian
"""
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

i, ip1, im1, j, jp1, jm1, n, dt, dx, dy, u, u_n, u_m1, b, x, f, q = sym.symbols('i ip1 im1 j jp1 jm1 n dt dx dy u u_n u_m1 b x f q')  # global symbols
y, c, t, b, u_np1, V, Lx, Ly, w, A, B, mx, my, k = sym.symbols('y c t b u_np1 V Lx Ly w A B mx my k')  # global symbols
kx, ky = sym.symbols('kx ky')  # global symbols


def u_tt(i, j, n, dt):
    return (u(i, j, n+1) - 2*u_n(i, j) + u_m1(i, j)) / (dt**2)


def u_t(i, j, n, dt):
    return (u(i, j, n+1) - u_m1(i, j)) / (2*dt)


def qux_x(x, y):
    return (q(x+dx/2, y) * (u_n(ip1, j) - u_n(i, j)) -
            q(x-dx/2, y) * (u_n(i, j) - u_n(im1, j))) * (1/dx)**2


def quy_y(x, y):
    return (q(x, y+dy/2) * (u_n(i, jp1) - u_n(i, j)) -
            q(x, y-dy/2) * (u_n(i, j) - u_n(i, jm1))) * (1/dy)**2


def task_2_2():
    # we are looking for u(i, j, np1)
    # u(i, j, n) is saved in u_n
    # u(i, j, n-1) is saved in u_m1
    right_side = qux_x(i*dx, j*dy) + quy_y(i*dx, j*dy) + f(i*dx, j*dy, n*dt)
    g = right_side-((x - 2*u_n(i, j) + u_m1(i, j)) / (dt**2) + b* (x - u_m1(i, j)) / (2*dt))
    res_t = sym.simplify(sym.solve(g, x))[0]
    print("u[i, j, n+1] = ", res_t, "\n")
    g = res_t - u(i, j, n+1.0)
    g = g.subs(n, 0.0)
    g = g.subs(u_m1(i, j), u(i, j, 1.0) - 2*dt*V(i, j))  # discretized initial condition
    g = g.subs(u(i, j, 1.0), x)
    res_1 = sym.simplify(sym.solve(g, x))[0]
    print("u[i, j, 1] = ", res_1)

    rt = \
    1/(b*dt + 2)*((
        + b*dt*u_m1(i, j)
        + 2*dt**2*f(dx*i, dy*j, dt*n)
        + 4*u_n(i, j)
        - 2*u_m1(i, j)
    )
    + 2*(dt/dy)**2*(
        - q(dx*i, dy*(2*j - 1)/2)*u_n(i, j)
        + q(dx*i, dy*(2*j - 1)/2)*u_n(i, jm1)
        - q(dx*i, dy*(2*j + 1)/2)*u_n(i, j)
        + q(dx*i, dy*(2*j + 1)/2)*u_n(i, jp1)
    )
    + 2*(dt/dx)**2*(
        - q(dx*(2*i - 1)/2, dy*j)*u_n(i, j)
        + q(dx*(2*i - 1)/2, dy*j)*u_n(im1, j)
        - q(dx*(2*i + 1)/2, dy*j)*u_n(i, j)
        + q(dx*(2*i + 1)/2, dy*j)*u_n(ip1, j)
    ))

    r1 = 0.5 * (
        (2*dt - b*dt**2)*V(i, j)
        + dt**2*f(dx*i, dy*j, 0)
        + 2*u_n(i, j)
        + (dt/dy)**2*(
            - q(dx*i, dy*(2*j - 1)/2)*u_n(i, j)
            + q(dx*i, dy*(2*j - 1)/2)*u_n(i, jm1)
            - q(dx*i, dy*(2*j + 1)/2)*u_n(i, j)
            + q(dx*i, dy*(2*j + 1)/2)*u_n(i, jp1))
        + (dt/dx)**2*(
            - q(dx*(2*i - 1)/2, dy*j)*u_n(i, j)
            + q(dx*(2*i - 1)/2, dy*j)*u_n(im1, j)
            - q(dx*(2*i + 1)/2, dy*j)*u_n(i, j)
            + q(dx*(2*i + 1)/2, dy*j)*u_n(ip1, j))
    )

    asdr1 =  (
            + (b*dt - 2)*V(i, j)
            + (4*dt -2*b*dt**2 + 4)*u_n(i, j)
            + 2*dt**2*f(dx*i, dy*j, 0) +
            2*(dt/dy)**2*(-q(dx*i, dy*(2*j - 1)/2)*u_n(i, j) +
                           q(dx*i, dy*(2*j - 1)/2)*u_n(i, jm1) -
                           q(dx*i, dy*(2*j + 1)/2)*u_n(i, j) +
                           q(dx*i, dy*(2*j + 1)/2)*u_n(i, jp1)) +

            2*(dt/dx)**2*(-q(dx*(2*i - 1)/2, dy*j)*u_n(i, j) +
                           q(dx*(2*i - 1)/2, dy*j)*u_n(im1, j) -
                           q(dx*(2*i + 1)/2, dy*j)*u_n(i, j) +
                           q(dx*(2*i + 1)/2, dy*j)*u_n(ip1, j))
            )/((b*dt + 2))

    print("\nAre the simplified equations by hand correct? (0 means yes)")
    print(sym.simplify(res_t-rt))
    print(sym.simplify(res_1-r1))


#    u_ij = - k_2*u_m1[i,j] + k_1*2*u_n[i,j]
#    u_xx = k_3*Cx2*(u_n[im1,j] - 2*u_n[i,j] + u_n[ip1,j])
#    u_yy = k_3*Cx2*(u_n[i,jm1] - 2*u_n[i,j] + u_n[i,jp1])
#    f_term = k_4*dt2*f(x, y, t_1)
#    r =  u_ij + u_xx + u_yy + f_term
    return res_t, res_1


def task_35():
#    kx = mx*sym.pi/Lx
#    ky = my*sym.pi/Ly

    A = B
    b = sym.sqrt(2*k*kx**2 + 2*k*ky**2)
    c = b/2
    w = sym.sqrt(kx**2*k + ky**2*k - c**2)


    def u_e(x, y, t):
        return (A*sym.cos(w*t) + B*sym.sin(w*t)) * \
            sym.exp(-c*t) * sym.cos(kx*x) * sym.cos(ky*y)

    def q(x, y):
        return k

    ux = sym.simplify(sym.diff(u_e(x, y, t), x))
    uy = sym.simplify(sym.diff(u_e(x, y, t), y))
    ut = sym.simplify(sym.diff(u_e(x, y, t), t))
    utt = sym.simplify(sym.diff(u_e(x, y, t), t, t))
    qux = sym.simplify(q(x, y)*ux)
    quy = sym.simplify(q(x, y)*uy)
    print(ux)
    print(uy)
    print(qux)
    print(quy)
    quxx = sym.simplify(sym.diff(qux, x))
    quyy = sym.simplify(sym.diff(quy, y))
    print(quxx)
    print(quyy)
    print()
    f = sym.simplify(utt+b*ut-quxx-quyy)
    I = sym.simplify(u_e(x, y, 0))
    V = sym.simplify(ut.subs(t, 0))
#    res_b = sym.simplify(sym.solve(V, b))#[0]
#    print("b=", res_b)
    print()
#    print("b = b")
    print("f(x, y, t) = ", f)
    print("q(x, y) = ", q(x, y))
    print("I(x, y) = ", I)
    print("V(x, y) = ", V)
    print()
    print("\ndef u_e(x, y, t): \n    return", u_e(x, y, t))
    print("\ndef f(x, y, t): \n    return", f)
    print("\ndef q(x, y): \n    return", q(x, y))
    print("\ndef I(x, y): \n    return", I)
    print("\ndef V(x, y): \n    return", V)
#    V = 0 = (-A*b + B*sqrt(-b**2 + 4*k*kx**2 + 4*k*ky**2))*cos(kx*x)*cos(ky*y)/2
#    A*b = B*sqrt(-b**2 + 4*k*kx**2 + 4*k*ky**2)
#    b**2 = -b**2 + 4*k*kx**2 + 4*k*ky**2
#    2*b**2 = 4*k*kx**2 + 4*k*ky**2
#    b = sym.sqrt(2*k*kx**2 + 2*k*ky**2)


def task_36():
#    kx = mx*sym.pi/Lx
#    ky = my*sym.pi/Ly


    def u_e(x, y, t):
        return (A*sym.cos(w*t) + B*sym.sin(w*t)) * \
            sym.exp(-c*t) * sym.cos(kx*x) * sym.cos(ky*y)

    def q(x, y):
        return 1/sym.sin(kx*x) * 1/sym.sin(ky*y)

    ux = sym.simplify(sym.diff(u_e(x, y, t), x))
    uy = sym.simplify(sym.diff(u_e(x, y, t), y))
    ut = sym.simplify(sym.diff(u_e(x, y, t), t))
    utt = sym.simplify(sym.diff(u_e(x, y, t), t, t))
    qux = sym.simplify(q(x, y)*ux)
    quy = sym.simplify(q(x, y)*uy)
    print(ux)
    print(uy)
    print(qux)
    print(quy)
    quxx = sym.simplify(sym.diff(qux, x))
    quyy = sym.simplify(sym.diff(quy, y))
    print(quxx)
    print(quyy)
    print()
    f = sym.simplify(utt+b*ut-quxx-quyy)
    I = sym.simplify(u_e(x, y, 0))
    V = sym.simplify(ut.subs(t, 0))
#    res_b = sym.simplify(sym.solve(V, b))#[0]
#    print("b=", res_b)
    print()
#    print("b = b")
    print("f(x, y, t) = ", f)
    print("q(x, y) = ", q(x, y))
    print("I(x, y) = ", I)
    print("V(x, y) = ", V)
    print()
    print("\ndef u_e(x, y, t): \n    return", u_e(x, y, t))
    print("\ndef f(x, y, t): \n    return", f)
    print("\ndef q(x, y): \n    return", q(x, y))
    print("\ndef I(x, y): \n    return", I)
    print("\ndef V(x, y): \n    return", V)
#    V = 0 = (-A*b + B*sqrt(-b**2 + 4*k*kx**2 + 4*k*ky**2))*cos(kx*x)*cos(ky*y)/2
#    A*b = B*sqrt(-b**2 + 4*k*kx**2 + 4*k*ky**2)
#    b**2 = -b**2 + 4*k*kx**2 + 4*k*ky**2
#    2*b**2 = 4*k*kx**2 + 4*k*ky**2
#    b = sym.sqrt(2*k*kx**2 + 2*k*ky**2)

def task_33():

    def u_e(x, y, t):
        kx = mx*sym.pi/Lx
        ky = my*sym.pi/Ly
        return A*sym.cos(kx*x)*sym.cos(ky*y)*sym.cos(w*t)

    def q(x, y):
        return c**2

    ux = sym.simplify(sym.diff(u_e(x, y, t), x))
    uy = sym.simplify(sym.diff(u_e(x, y, t), y))
    ut = sym.simplify(sym.diff(u_e(x, y, t), t))
    utt = sym.simplify(sym.diff(u_e(x, y, t), t, t))
    qux = sym.simplify(q(x, y)*ux)
    quy = sym.simplify(q(x, y)*uy)
    quxx = sym.simplify(sym.diff(qux, x))
    quyy = sym.simplify(sym.diff(quy, y))
    f = sym.simplify(utt+b*ut-quxx-quyy)
    I = sym.simplify(u_e(x, y, 0))
    V = sym.simplify(ut.subs(t, 0))
    print("ut = ", ut)
    print("utt = ", utt)
    print("lhs = ", utt+b*ut)
    print()
    print("f = ", f)
    print("b = 0")
    print("q(x,y) = c**2")
    print("I = ", I)
    print("V = ", V)


def task_3_1():
    def u_e(x, y, t):
        return c

    ux = sym.simplify(sym.diff(u_e(x, y, t), x))
    uy = sym.simplify(sym.diff(u_e(x, y, t), y))
    ut = sym.simplify(sym.diff(u_e(x, y, t), t))
    utt = sym.simplify(sym.diff(u_e(x, y, t), t, t))
    qux = sym.simplify(q(x, y)*ux)
    quy = sym.simplify(q(x, y)*uy)
    quxx = sym.simplify(sym.diff(qux, x))
    quyy = sym.simplify(sym.diff(quy, y))
    f = sym.simplify(utt+b*ut-quxx-quyy)
    I = sym.simplify(u_e(x, y, 0))
    V = sym.simplify(ut.subs(t, 0))
    print("ut = ", ut)
    print("utt = ", utt)
    print("lhs = ", utt+b*ut)
    print("f = ", f)
    print("b = anything")
    print("q(x,y) = anything")
    print("I = ", I)
    print("V = ", V)

    return

if __name__ == """__main__""":
    task_36()
#    task_33()
#    res_t, res_1 = task_2_2()
#    task_3_1()


