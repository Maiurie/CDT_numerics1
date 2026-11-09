# Numerical schemes for solving the 1D linear advection equation on a
# periodic, uniform grid.
# Authors: Hilary Weller <h.weller@reading.ac.uk> (FTBS, FTCS, CTCS)
#          Maiurie Rasakulasuriar (semi-Lagrangian schemes)
#
# Every scheme has the same interface:
#       scheme(phiLevels, c)
# where phiLevels is a list of 1D arrays (latest solution is phiLevels[-1])
# and c = u*dt/dx is the Courant number. phiLevels is updated in place and
# nothing is returned.

import numpy as np

# -----------------------------------
# Eulerian finite difference schemes
# -----------------------------------
def FTBS(phiLevels, c):
    """Forward in time, backward in space. First order, stable for 0 <= c <= 1."""
    phi = phiLevels[-1]
    phiNew = phi - c*(phi - np.roll(phi, 1))      # np.roll(phi, 1)[j] = phi[j-1]
    phiLevels.append(phiNew)
    phiLevels.pop(0)


def FTCS(phiLevels, c):
    """Forward in time, centred in space. Unconditionally unstable."""
    phi = phiLevels[-1]
    phiNew = phi - 0.5*c*(np.roll(phi, -1) - np.roll(phi, 1))
    phiLevels.append(phiNew)
    phiLevels.pop(0)


def CTCS(phiLevels, c):
    """Centred in time, centred in space (leapfrog). Stable for |c| <= 1.

    Needs two time levels. If only one is present, the first step is taken with
    FTCS and the list is extended to hold [phi^n, phi^(n+1)].
    """
    if len(phiLevels) == 1:
        phiOld = phiLevels[0].copy()
        FTCS(phiLevels, c)
        phiLevels.insert(0, phiOld)
        return

    phiOld, phiMid = phiLevels[0], phiLevels[1]
    phiNew = phiOld - c*(np.roll(phiMid, -1) - np.roll(phiMid, 1))
    phiLevels.append(phiNew)
    phiLevels.pop(0)


# ------------------------
# Semi-Lagrangian schemes 
# ------------------------
def _departurePoints(nx, c):
    q = (np.arange(nx) - c) % nx
    i = np.floor(q).astype(int)
    return i, q - i

def linearSemiLagrangian(phiLevels, c):
    #Semi-Lagrangian with linear interpolation (nodes i, i+1)
    phi = phiLevels[-1]
    nx = len(phi)
    i, a = _departurePoints(nx, c)

    phiNew = (1.0 - a)*phi[i % nx] + a*phi[(i + 1) % nx]
    phiLevels.append(phiNew)
    phiLevels.pop(0)

def quadraticSemiLagrangian(phiLevels, c):
    #Semi-Lagrangian with downwind quadratic interpolation (nodes i, i+1, i+2)
    phi = phiLevels[-1]
    nx = len(phi)
    i, a = _departurePoints(nx, c)

    # Quadratic Lagrange interpolation weights
    w0 = (1.0 - a) * (2.0 - a) / 2.0
    w1 = a * (2.0 - a)
    w2 = -a * (1.0 - a) / 2.0

    phiNew = (
        w0 * phi[i % nx]
        + w1 * phi[(i + 1) % nx]
        + w2 * phi[(i + 2) % nx]
    )

    phiLevels.append(phiNew)
    phiLevels.pop(0)

def cubicSemiLagrangian(phiLevels, c):
    #Semi-Lagrangian with cubic interpolation (nodes i-1, i, i+1, i+2)

    phi = phiLevels[-1]
    nx = len(phi)
    i, a = _departurePoints(nx, c)

    wm1 = -a*(a - 1.0)*(a - 2.0)/6.0
    w0 = (a + 1.0)*(a - 1.0)*(a - 2.0)/2.0
    w1 = -(a + 1.0)*a*(a - 2.0)/2.0
    w2 = (a + 1.0)*a*(a - 1.0)/6.0

    phiNew = (wm1*phi[(i - 1) % nx] + w0*phi[i % nx]
              + w1*phi[(i + 1) % nx] + w2*phi[(i + 2) % nx])
    phiLevels.append(phiNew)
    phiLevels.pop(0)
