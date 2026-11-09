# Helper functions for testing advection schemes.
# Authors: Hilary Weller <h.weller@reading.ac.uk>,
#          Maiurie Rasakulasuriar <m.rasakulasuriar@reading.ac.uk>

import numpy as np
import os
import matplotlib.pyplot as plt

#List of function in this file:
    #cosBell (initial condition written by Hilary)
    #cosBell_assignment (initial condition with parameters in report)
    #squarePulse (initial condition written for square pulse)
    #sineWave (initial condition for a continuous smooth profile)
    #l2error (calculating the error analysis for L2)
    #exactSolution (function to calculate the analytic solution)
    #solveSchemes (function to apply the scheme from schemesBasic.py)
    #compareSchemeSnapshots (function to plot phi against x at t=T 
        #after applying the scheme,so we can compare the schemes 
        # solution against analytic )
    #errorVsCourant (plots the l2 error of the scheme solution against 
        # analytic solution as Courant number increases, for fixed dx, 
        # we vary dt to change Courant number c)
    #convergenceStudy (Experiment X: function to plot log(error) 
        # against log(dx) as we vary dx, but keep Courant number fixed 
        # (dt changes according to dx))
    #error_Time (function to calculate the l2 error over time)
    #plotErrorVsCourant (function to plot the l2 error as Courant number 
        # increases, for fixed dx and varying dt)
    #errors_SL (function to calculate and plot log(error) against log(dx) 
        #or log(dt), and check gradient of line for convergence property)
    #variance_conservation (function to calculate phi^2 over time (variance)
        #and investigates which schemes conserve variance, oscillate, or 
        #diverge)

#Key for colours of schemes
SCHEME_COLOURS = {
    "FTBS": "blue",
    "FTCS": "firebrick",
    "CTCS": "forestgreen",
    "linearSemiLagrangian": "darkorange",
    "quadraticSemiLagrangian": "mediumpurple",
    "cubicSemiLagrangian": "orchid"}

# Initial conditions
def cosBell(x, a, b):
    """Cosine bell between a and b (zero elsewhere)."""
    return np.where((x > a) & (x < b),
                    0.5*(1 - np.cos(2*np.pi*(x - a)/(b - a))), 0.)

def cosBell_assignment(x):
    """Cosine bell with a=0.1, b=0.5, as specified in the assignment."""
    return cosBell(x, 0.1, 0.5)

def squarePulse(x):
    """Square pulse with a=0.1, b=0.5, as specified in the assignment."""
    return np.where((x >= 0.1) & (x < 0.5), 1.0, 0.0)

def sineWave(x):
    """Infinitely smooth periodic function on [0,1). Used for convergence tests."""
    return np.sin(2*np.pi*x)

# Error norm
def l2error(phi, phiA):
    """Relative l2 error norm of phi compared with the exact solution phiA."""
    return np.sqrt(np.sum((phi - phiA)**2)/np.sum(phiA**2))

#Exact analytic solution
def exactSolution(init, x, u, t, xLim=(0, 1)):
    """Analytic solution phi(x,t) = phi(x - u t, 0) on a periodic domain."""
    L = xLim[1] - xLim[0]
    return init((x - xLim[0] - u*t) % L + xLim[0]) #init is initial condition

#Applying the schemes from schemesBasic.py
def solveSchemes(init, schemes, nx, nt, u, endTime, xLim=(0, 1)):

    dx = (xLim[1] - xLim[0]) / nx
    dt = endTime / nt
    c = u * dt / dx
    x = xLim[0] + dx * np.arange(nx)

    phi0 = init(x)
    phiExact = exactSolution(init, x, u, endTime, xLim)

    solutions = {}

    for scheme in schemes:
        phiLevels = [phi0.copy()]
        for _ in range(nt):
            scheme(phiLevels, c)
        solutions[scheme.__name__] = phiLevels[-1]

    return x, phi0, phiExact, c, solutions

#Section 3 and 5.1
#Comparing FTBS, FTCS, CTCS, and cubicSL with analytic solution
def compareSchemeSnapshots(init, schemes, outFile, nx=40, nt=10,
                           u=1.0, endTime=0.1, xLim=(0, 1),
                           yLim=(None, None)):

    x, phi0, phiExact,c,solutions = solveSchemes(init,schemes, 
                                    nx,nt, u,endTime, xLim)

    #Plotting the initial condition at time t=0
    plt.plot(x, phi0, "k:", linewidth=1.5,label="Initial condition")

    #Plotting the analytic solution. We expect this to be a translation of 
    #the initial condition
    plt.plot(x, phiExact, "k-", linewidth=2,label="Analytic solution")

    #Iterating through each scheme 
    for scheme in schemes:
        name = scheme.__name__
        plt.plot(x, solutions[name],
                color=SCHEME_COLOURS[name], label=name)

    #Overall plot includes: initial condition, analytic solution, scheme solutions
    plt.title(f"nx = {nx}, nt = {nt}, Courant number c = {c:.3g}")
    plt.xlim(xLim)
    plt.ylim(yLim)
    plt.xlabel("x", fontsize=14)
    plt.ylabel(r"$\phi$", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    #Saving the files to the plots folder:
        #Saved as basic_cosineBell.pdf and basic_squarePulse.pdf
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{outFile}", bbox_inches="tight")
    plt.close()

#Error as a function of Courant number fot FTBS, FTCS, CTCS, cubicSL
def errorVsCourant(init, schemes, outFile, nx=40,
                  ntValues=range(1, 41), u=1.0,
                  endTime=0.1, xLim=(0, 1)):

    ntValues = list(ntValues) #Convert no of dt into list
    errors = {s.__name__: [] for s in schemes}
    cValues = []

    # Calculate errors for each time-step value
    for nt in ntValues:
        #Solve numerical scheme to get exact solution
        x, phi0, phiExact, c, solutions = solveSchemes(init, schemes, nx, nt, u, endTime, xLim)
        
        cValues.append(c) #Store c for each time step

        for scheme in schemes:
            name = scheme.__name__
            #Calculate l2 error for scheme
            errors[name].append(l2error(solutions[name], phiExact))

    cValues = np.array(cValues)

    # Plot errors against Courant number
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for scheme in schemes:
        name = scheme.__name__
        err = np.array(errors[name])
        ok = np.isfinite(err)

        ax.plot(cValues[ok], err[ok], "o-",
                markersize=4, color=SCHEME_COLOURS[name],
                label=name)

    ax.axvline(1.0, color="k", linestyle="--",linewidth=1, label="c = 1")
    ax.set_xlabel("Courant number c", fontsize=12)
    ax.set_ylabel(r"Relative $\ell_2$ error", fontsize=12)
    ax.set_title(
        f"Error against Courant number: {init.__name__} "
        f"(nx={nx}, t={endTime})")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{outFile}", bbox_inches="tight")
    plt.close()

    return cValues, errors



###########################
#Experiment X, Section 5.2
###########################
#Convergence with spatial resolution
def convergenceStudy(init, schemes, outFile,
                    nxValues=(20, 40, 80, 160, 320, 640),
                    c=0.4, u=1.0, endTime=0.5, xLim=(0, 1)):

    # Calculate the domain length and grid spacings
    L = xLim[1] - xLim[0]
    dxValues = np.array([L/nx for nx in nxValues])

    # Initialise a dictionary to store errors
    errors = {s.__name__: [] for s in schemes}

    # Loop through each grid resolution
    for nx, dx in zip(nxValues, dxValues):

        # Calculate the number of time steps
        nt = int(round(endTime*u/(c*dx)))

        # Calculate the actual Courant number
        cActual = u*(endTime/nt)/dx

        # Create the spatial grid and initial condition
        x = xLim[0] + dx*np.arange(nx)
        phi0 = init(x)

        # Calculate the exact solution
        phiExact = exactSolution(init, x, u, endTime, xLim)

        # Solve each numerical scheme
        for scheme in schemes:
            phiLevels = [phi0.copy()]

            for _ in range(nt):
                scheme(phiLevels, cActual)

            # Calculate and store the L2 error
            errors[scheme.__name__].append(
                l2error(phiLevels[-1], phiExact)
            )

    # Create the log-log convergence plot
    plt.figure(figsize=(8, 6))

    for scheme in schemes:
        name = scheme.__name__
        err = np.array(errors[name])

        slope, intercept = np.polyfit(
            np.log(dxValues), np.log(err), 1
        )

        col = SCHEME_COLOURS[name]

        # Plot the numerical errors
        plt.loglog(dxValues, err, "o", color=col,
                   label=f"{name} (order {slope:.2f})")

        # Plot the fitted convergence line
        plt.loglog(dxValues,
                   np.exp(intercept)*dxValues**slope,
                   "-", color=col)

        # Print the observed order
        print(f"{name}: observed order = {slope:.2f}")

    # Add plot labels and formatting
    plt.xlabel(r"Grid spacing $\Delta x$", fontsize=12)
    plt.ylabel(r"Relative $\ell_2$ error", fontsize=12)
    plt.title(f"Convergence: {init.__name__}, Courant number {c}")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)

    # Save and display the plot
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{outFile}", bbox_inches="tight")
    plt.close()

    # Return grid spacings and errors
    return dxValues, errors

###########################
#Experiment Y, Section 5.3
###########################
#For each scheme and initial condition, calculate the l2 error over time.
def error_time(scheme, init, nx, nt, T=1.0):
    """Advect once round the domain (time T, u = 1) with nx points and nt steps.
    Returns the relative l2 error against the exact solution."""
    dx = 1.0/nx
    x = dx*np.arange(nx)
    c = T/(nt*dx)                                # Courant number
    levels = [init(x)]
    with np.errstate(all="ignore"):              # unstable schemes may overflow
        for _ in range(nt):
            scheme(levels, c)
    return l2error(levels[-1], exactSolution(init, x, 1.0, T))
 

# Error against Courant number
def plotErrorVsCourant(schemes, outFile, init, nx=40):

    # Define the number of time steps and corresponding Courant numbers
    ntValues = np.arange(4, 401)
    cValues = nx/ntValues

    # Plot the error for each scheme
    plt.figure(figsize=(9, 5.5))

    for scheme in schemes:

        # Calculate errors for each Courant number
        errors = np.array([error_time(scheme, init, nx, nt) for nt in ntValues])

        # Exclude invalid and effectively zero errors
        ok = np.isfinite(errors) & (errors > 1e-12)

        # Plot the errors using logarithmic axes
        plt.loglog(cValues[ok], errors[ok], "o-",
                   markersize=4, color=SCHEME_COLOURS[scheme.__name__],
                   label=scheme.__name__)

    # Add reference line at c = 1
    plt.axvline(1, color="k", linestyle="--", label="c = 1")

    # Format and display the graph
    plt.xlabel("Courant number c", fontsize=12)
    plt.ylabel(r"Relative $\ell_2$ error", fontsize=12)
    plt.title(f"Error against Courant number: {init.__name__} (nx={nx})")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)

    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{outFile}", bbox_inches="tight")
    plt.close()

# Check the exponents in  error ~ dx^(d+1) / dt
def errors_SL(schemes, outFile, init):

    # Define expected polynomial degrees
    degrees = [1, 2, 3]

    # Experiment 1: Vary dx while keeping nt fixed
    nxValues = np.array([20, 60, 180, 540])
    dx = 1.0/nxValues

    # Experiment 2: Vary dt while keeping nx fixed
    ntValues = np.array([420, 140, 84, 60, 28, 20, 12, 4])
    dt = 1.0/ntValues

    # Create two side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loop through each scheme and its expected degree
    for scheme, d in zip(schemes, degrees):

        name = scheme.__name__
        colour = SCHEME_COLOURS[name]
        # Experiment 1: Calculate errors for different dx values
        err1 = np.array([error_time(scheme, init, nx, 40) for nx in nxValues])

        # Calculate the gradient of the log-log fit
        g1 = np.polyfit(np.log(dx), np.log(err1), 1)[0]

        # Plot errors against dx
        ax1.loglog(dx, err1, "o-", color=colour,label=f"{name} ({g1:.2f})")

        # Experiment 2: Calculate errors for different dt values
        err2 = np.array([error_time(scheme, init, 210, nt) for nt in ntValues])

        # Calculate the gradient of the log-log fit
        g2 = np.polyfit(np.log(dt), np.log(err2), 1)[0]

        # Plot errors against dt
        ax2.loglog(dt, err2, "o-", color=colour,label=f"{name} ({g2:.2f})")

        # Print expected and measured gradients
        print(f"{name}: dx gradient = {g1:.2f}")
        print(f"{name}: dt gradient = {g2:.2f}")

    # Add labels, titles and legends
    ax1.set(xlabel="Grid spacing dx", ylabel="Relative L2 error",
            title="Error against dx (expect gradient d + 1)")

    ax2.set(xlabel="Time step dt", ylabel="Relative L2 error",
            title="Error against dt (expect gradient -1)")

    # Format both plots
    for ax in (ax1, ax2):
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{outFile}", bbox_inches="tight")
    plt.close()

###########################
#Experiment Z, Appendix C
###########################
#Appendix- Checking variance conservation (phi ^2)
    #This experiment was carried out first, but disregarded from the report, 
    # since the results aren't the clearest and don't provide evidence that 
    # cubic SL is the best scheme. 

def variance_conservation(init, schemes, outFile,nx=40, nt=100, u=1.0,
                          endTime=0.5, xLim=(0, 1)):
    
    dx = (xLim[1] - xLim[0]) / nx
    dt = endTime / nt
    c = u * dt / dx
    x = xLim[0] + dx * np.arange(nx)
    times = np.linspace(0, endTime, nt + 1)

    phi0 = init(x)
    results = {}

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(1, color="black", linestyle="--", label="Analytic (= 1)")

    for scheme in schemes:
        levels = [phi0.copy()]
        variance = [dx * np.sum(phi0**2)]

        with np.errstate(all="ignore"):
            for _ in range(nt):
                scheme(levels, c)
                variance.append(dx * np.sum(levels[-1]**2))

        #Calculate the ratio of variance at time t, over initial variance
        ratio = np.array(variance) / variance[0] 
        results[scheme.__name__] = np.array(variance)

        ax.plot(times, ratio, color=SCHEME_COLOURS[scheme.__name__],
        label=scheme.__name__)
        
    ax.set_xlabel("Time, t")
    ax.set_ylabel(r"Variance / initial variance")
    ax.set_title(f"Variance conservation: {init.__name__}, c={c:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{outFile}", bbox_inches="tight")
    plt.close()

    return times, results