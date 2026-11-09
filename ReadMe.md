ReadMe

Advection assignment — Maiurie Rasakulasuriar
==================================================

This folder contains the original advection examples plus a cubic
semi-Lagrangian scheme using cubic Lagrange interpolation.

Files
-----
advection.py
    Compares snapshots from FTBS, CTCS, FTCS, and cubicSemiLagrangian.

schemesBasic.py
    FTBS, FTCS, CTCS, and linear, quadratic and cubic semi Lagrangian

advectionHelpers.py
    cosBell, cosBell_assignment, squarePulse, sineWave, l2error, exactSolution, solveSchemes, compareSchemeSnapshots, errorVsCourant, convergenceStudy, error_Time, plotErrorVsCourant, errors_SL, variance_conservation,

Requirements (packages and software)
------------
Python 3, NumPy, OS, and Matplotlib.

How to run
----------
Open a terminal in this folder and run:
    python advection.py

Plots are written to the plots/ folder. 
Plots are also shown, so you will need to exit each one.
Run scripts from this folder so that
Python can import the neighbouring modules and the relative plot paths work.

Method assumptions and limitations
----------------------------------
The new scheme (Semi Lagrangian) assumes constant velocity, a uniform grid, and periodic boundaries. It traces each arrival grid point backwards by udt, then uses four grid values and cubic Lagrange interpolation at the departure point.
It can take Courant numbers with magnitude greater than one, but the basic pointwise interpolation is not guaranteed to conserve 
total mass and may produce small overshoots 
near steep gradients.
