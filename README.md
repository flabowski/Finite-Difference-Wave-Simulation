# Project 1: two-dimensional, standard, linear wave equation, with damping

### Overview
This project is a finite difference implementation in python/ numpy. It is used to compute two dimensional waves.
The behavior of waves as they pass through different mediums with different velocities are studied and reported, see [report_wave_project.pdf](doc/report_wave_project.pdf).
The implementation of the numerical solution can be found in [wave2D.py](src/wave2D.py).
There is a function for every task that computes and animates the result.
If you want to know how the animation works, see [test_animation.py](src/test_animation.py).

The following sections demonstrate some examples.

### Verification: Exact 1D plug-wave solution in 2D
There is a pulse function that is split into two identical 1D waves. They are moving in opposite direction, exactly one cell per time step. The discrete solution is then equal to the exact solution.

Pulse in x-direction       |  Pulse in y-direction
:-------------------------:|:-------------------------:
![](data/pulse_x.gif)      |  ![](data/pulse_y.gif)


### Verification: Standing, undamped waves
Exact solution as a colored surface, numerical solution visualized as a grid.

![](data/standing_wave.gif)

### Verification: Standing, damped waves (manufactured solution)
Exact solution as a colored surface, numerical solution visualized as a grid.

![](data/standing_damped.gif)

### Investigation of a physical problem: Tsunami over a subsea mountain
![](data/tsunami.gif)
