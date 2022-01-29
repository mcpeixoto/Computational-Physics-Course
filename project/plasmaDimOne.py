
import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt
import time
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from banded import banded

########## DEFINING PARAMETERS ##########

length = 100                # D=1 space length
n_particles = 100000        # Number of particles
grid_cells  = 1000          # Number of cells in the grid (to be used in the Poisson solver)
n0 = 1                      # average electron density
dx = length/grid_cells      # Distance between grid points
dt = 1                      # Timestep
t = 0                       # Initial time

##### Beam inicialization #####

## Defining Positions
# Initialize random positions distributed by the length of the simulation
positions = np.random.rand( n_particles) * length

# Adding a perturbation with a sin
positions += 0.5*np.sin(2*np.pi*positions/length)

## Defining Velocities
velocities = np.ones(n_particles)

## Velocities of the two beams
top_beam_velocity = 2
low_beam_velocity = -2

# Setting desired velocities
velocities[:int(n_particles/2)] = top_beam_velocity
velocities[int(n_particles/2):] = low_beam_velocity

# Adding a perturbation with a sin
velocities += 0.2*np.sin(2*np.pi*positions[0]/length)



# Defining potential matrix system
# for being used by banded
e = np.ones(grid_cells, dtype=np.float32)
potential_vals  = np.vstack((e,-2*e,e))
potential_vals[1][0] = 1
potential_vals[0][0] = 0
potential_vals[1][-1] = 1
potential_vals[2][-1] = 0
potential_vals  /= dx**2



# Defining the matrix for the 1st derivative
# computation of the eletric field
e = np.ones(grid_cells)
diags = np.array([-1,1])
vals  = np.vstack((-e,e))
first_derivative = sp.spdiags(vals, diags, grid_cells, grid_cells)
first_derivative = sp.lil_matrix(first_derivative)
first_derivative[0, -1] = -1
first_derivative[-1, 0] = 1
first_derivative /= (2*dx)
first_derivative = sp.csr_matrix(first_derivative)


def calcular_aceleração(positions):
    ###################################
    ####### Density Computation #######
    ###################################

    # Make bins and count
    density, _ = np.histogram(positions, bins=grid_cells, range=(0,length))
    density = density.astype(float)

    # Normalize
    density *= n0 * length / n_particles / dx

    ###################################
    ###### Potential Computation ######
    ###################################

    # Solve Poisson's Equation: laplacian(phi) = n-n0
    phi_grid = banded(potential_vals, density-n0, 1, 1)

    ###################################
    #### Eletric Field Computation ####
    ###################################

    # Apply Derivative to get the Electric field
    E_grid = - first_derivative @ phi_grid

    
    # Interpolate grid value onto particle locations
    xp = np.linspace(0, length, num=grid_cells)
    E = np.interp(positions, xp, E_grid)

    ####################################
    ##### Acceleration Computation #####
    ####################################

    a = - E

    return a



########## SIMULATION START ##########


# Begin figure for plots
fig = plt.figure(figsize=(15,12), dpi=100)


# Calculate initial acceleration and push velocity
# backwards by 1/2 the timestep
acc = calcular_aceleração(positions)
velocities -= acc * dt

# Simulation Main Loop
for i in range(100):
    # Plot
    if i % 2 == 0:
        # Particle distributions
        plt.subplot(4, 2, 1)
        plt.cla()
        plt.title("Blue Distribution")
        plt.hist(positions[:int(n_particles/2)], bins=50, range=(0,length), color='blue')
        plt.xlabel("x")
        plt.ylabel("Blue particles")

        plt.subplot(4, 2, 2)
        plt.cla()
        plt.title("Red Distribution")
        plt.hist(positions[int(n_particles/2):], bins=50, range=(0,length), color='red')
        plt.xlabel("x")
        plt.ylabel("Red particles")

        # Particle phase space (individually)
        plt.subplot(4, 2, 3)
        plt.cla()
        plt.title("Speed / Position")
        plt.scatter(positions[:int(n_particles/2)], velocities[:int(n_particles/2)], s=.4,color='blue', alpha=0.5)
        plt.xlabel("X Position")
        plt.ylabel("X Speed")

        plt.subplot(4, 2, 4)
        plt.cla()
        plt.title("Speed / Position")
        plt.scatter(positions[int(n_particles/2):], velocities[int(n_particles/2):], s=.4,color='red', alpha=0.5)
        plt.xlabel("X Position")
        plt.ylabel("X Speed")

        # Particle phase space (together)
        plt.subplot(2, 1, 2)
        plt.cla()
        plt.title("Speed / Position")
        plt.scatter(positions[:int(n_particles/2)], velocities[:int(n_particles/2)], s=.4,color='blue', alpha=0.5)
        plt.scatter(positions[int(n_particles/2):], velocities[int(n_particles/2):], s=.4,color='red', alpha=0.5)
        plt.xlabel("X Position")
        plt.ylabel("X Speed")

        # Pause plot
        plt.pause(0.01)


    # Update accelerations
    acc = calcular_aceleração(positions)

    # Update velocities & positions
    velocities += acc * dt
    positions += velocities * dt

    # Particle in the cell, this dosen't let
    # the particles run out of the box
    positions = np.mod(positions, length)


    # Update current timestep
    t += dt
    





