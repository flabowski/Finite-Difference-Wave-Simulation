# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 22:13:30 2019

@author: Florian
"""
import os
import numpy as np
from mayavi import mlab

# Output path for you animation images
out_path = './'
out_path = os.path.abspath(out_path)+"\\..\\ani.tmp\\"
prefix = 'ani'
ext = '.png'


def calc(something):
    return something


@mlab.animate(delay=250)
def solver(user_action):
    for n in range(100):  # here is the iteration over t
        calc(123)  # do all the math ...
        if user_action(n):
            break
        yield
    return


def task():
    # here the specific PDE is handled

    def action(n):  # callback function
        Z = np.random.rand(*X.shape)
        ms.mlab_source.set(z=Z, scalars=Z)

        # concate filename with zero padded index number as suffix
        filename = os.path.join(out_path, '{}_{:06d}{}'.format(prefix, n, ext))
        mlab.savefig(filename=filename)
        return

    # set up the plot
    f = mlab.figure()
    f.scene.movie_maker.record = True
    X, Y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 25))
    Z = np.sin(X)
    ms = mlab.mesh(X, Y, Z)
    solver(action)  # the actual
    mlab.show()

if __name__ == "__main__":
    task()
    # run convert ani_*.png movie.gif after generating the png's
