###############################################################################
# Author: Hilary Weller, Maiurie Rasakulasuriar
# Description: Experiments comparing advection schemes against the analytic
#              solution.
#Date: 3rd November 2026
###############################################################################

from advectionHelpers import (cosBell_assignment, squarePulse, sineWave,
                              compareSchemeSnapshots, errorVsCourant,
                              convergenceStudy,plotErrorVsCourant, 
                              errors_SL, variance_conservation)

from schemesBasic import (FTBS, FTCS, CTCS, linearSemiLagrangian,
                          quadraticSemiLagrangian, cubicSemiLagrangian)

initialConditions = [(cosBell_assignment, "cosineBell"),
                     (squarePulse, "squarePulse")]

# Common set-up: nx = 40, endTime = 0.1, u = 1  ->  c = 4/nt
# Note: nt = 10 gives c = 0.4; nt = 3 gives c = 1.33,

# 1. FTBS, FTCS, CTCS against the analytic solution (c = 0.4)
basicSchemes = [FTBS, FTCS,CTCS, cubicSemiLagrangian]
for init, name in initialConditions:
    compareSchemeSnapshots(init, basicSchemes, f"basic_{name}.pdf",
                           nx=40, nt=10, endTime=0.1)

# 2. Error of FTBS, FTCS, CTCS as the Courant number increases
for init, name in initialConditions:
    errorVsCourant(init, basicSchemes, f"errorVsCourant_{name}.pdf",
                   nx=40, ntValues=range(1, 41), endTime=0.1)

# 5. Experiment X- Error against grid spacing (log-log): gradient = order of the scheme.
#    Fixed Courant number 0.4, one half revolution (endTime = 0.5).
convergenceSchemes = [FTBS, CTCS, linearSemiLagrangian, quadraticSemiLagrangian, cubicSemiLagrangian]
for init, name in [(sineWave, "sineWave"),(cosBell_assignment, "cosineBell"), (squarePulse, "squarePulse")]:
    convergenceStudy(init, convergenceSchemes, f"convergence_{name}.pdf",
                     nxValues=(20, 40, 80, 160, 320, 640), c=0.4,
                     endTime=1)

#6. Experiment Y- Checking the SL scheme with Courant number c >1, 

    #Compare the cubicSL with FTBS and CTCS to show that SL handles 
    #   larger time steps better
schemes=[FTBS, CTCS,cubicSemiLagrangian ]
for init, name in [
    (sineWave, "sineWave"),
    (cosBell_assignment, "cosineBell"),
    (squarePulse, "squarePulse")]: 
    plotErrorVsCourant(schemes,f"errorVsCourant_{name}.pdf",init, nx=40)

    #Checking order of convergence for all SL schemes, with varying dx and dt
slSchemes = [linearSemiLagrangian, quadraticSemiLagrangian, cubicSemiLagrangian]
for init, name in [
    (sineWave, "sineWave"),
    (cosBell_assignment, "cosineBell"),
    (squarePulse, "squarePulse")]:
    errors_SL(slSchemes, f"errors_SL_{name}.pdf", init)

#Appendix- Comparing interpolation of Semi Lagrangian schemes
for init, name in initialConditions:
    for nt in (10, 3):
        compareSchemeSnapshots(init, slSchemes, f"semiLagrangian_{name}_nt{nt}.pdf",
                               nx=40, nt=nt, endTime=0.1)

#Apendix- Checking variance over time. We expect CTCS to be closest to conservation
variance_schemes = [FTBS, FTCS, CTCS, linearSemiLagrangian,
              quadraticSemiLagrangian, cubicSemiLagrangian]
for init, name in [
    (sineWave, "sineWave"),
    (cosBell_assignment, "cosineBell"),
    (squarePulse, "squarePulse")]:
    variance_conservation(init, variance_schemes,f"variance_{name}.pdf", nx=40, nt=100, u=1.0, endTime=1)